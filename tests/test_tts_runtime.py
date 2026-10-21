"""Tests for lector/tts_runtime.py.

PiperRunner/build_piper_runner aren't unit-tested here, matching how
llm_kit's own in-process backends (LlamaCppRunner, VLLMRunner, HFRunner)
aren't either: they wrap a real heavy/optional dependency (piper-tts +
an actual voice model file) with no meaningful fake to substitute
in-process. What IS tested: the pure config/interface pieces, and
OpenAITTSRunner, which - like ServerRunner/OpenRouterRunner - accepts
an injected `client` specifically so its request-building logic can be
tested without a real API key or network call. CometTTSRunner likewise
takes an injected `transport` (the three HTTP calls it makes), and its
ffmpeg decode step is monkeypatched out - so the submit/poll/download
flow is tested without the network or the ffmpeg binary.

OrpheusRunner's actual generation/decoding (llama_cpp.generate + SNAC's
neural encode/decode) isn't tested for the same reason - no torch/
llama_cpp/snac installed here, and no meaningful fake for a real GGUF
model either. What IS tested: _build_prompt_tokens, which only calls
self.llama.tokenize() - injectable with a trivial fake, so the actual
prompt-assembly logic (turn order, voice-name prefixing, where a
cloning reference goes) is checked directly. The token<->SNAC-codes
math itself lives in orpheus_codec.py and is tested there.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from lector.orpheus_codec import END_OF_AI, END_OF_HUMAN, END_OF_SPEECH, START_OF_AI, START_OF_HUMAN, START_OF_SPEECH
from lector.tts_runtime import (
    BaseTTSRunner,
    CachedTTSRunner,
    CometHTTPError,
    CometTTSRunner,
    OpenAITTSRunner,
    OrpheusRunner,
    SynthesisConfig,
    build_comet_runner,
    detect_language,
    reference_audio_data_uri,
)


# -- SynthesisConfig --------------------------------------------------------

def test_synthesis_config_defaults():
    config = SynthesisConfig()

    assert config.speed == 1.0
    assert config.volume == 1.0


def test_synthesis_config_rejects_non_positive_speed():
    with pytest.raises(ValidationError):
        SynthesisConfig(speed=0.0)


def test_synthesis_config_rejects_negative_volume():
    with pytest.raises(ValidationError):
        SynthesisConfig(volume=-1.0)


# -- BaseTTSRunner context manager contract ----------------------------------

class _ClosingRunner(BaseTTSRunner):
    def __init__(self):
        self.closed = False

    def synthesize(self, text):
        return b""

    def close(self):
        self.closed = True


def test_base_runner_calls_close_on_context_exit():
    runner = _ClosingRunner()

    with runner as ctx_runner:
        assert ctx_runner is runner
        assert runner.closed is False

    assert runner.closed is True


def test_base_runner_synthesize_is_not_implemented_by_default():
    with pytest.raises(NotImplementedError):
        BaseTTSRunner().synthesize("hello")


# -- OpenAITTSRunner --------------------------------------------------------

class _FakeSpeechResponse:
    def __init__(self, content: bytes):
        self.content = content


class _FakeSpeechEndpoint:
    def __init__(self, response: _FakeSpeechResponse):
        self._response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeAudio:
    def __init__(self, speech: _FakeSpeechEndpoint):
        self.speech = speech


class _FakeOpenAIClient:
    def __init__(self, response: _FakeSpeechResponse):
        self.audio = _FakeAudio(_FakeSpeechEndpoint(response))


def test_openai_tts_runner_requires_an_api_key_when_no_client_is_given(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError):
        OpenAITTSRunner()


def test_openai_tts_runner_synthesize_returns_the_response_content():
    client = _FakeOpenAIClient(_FakeSpeechResponse(b"RIFF...wav-bytes"))
    runner = OpenAITTSRunner(voice="coral", client=client)

    audio_bytes = runner.synthesize("Hello, this is a lecture.")

    assert audio_bytes == b"RIFF...wav-bytes"


def test_openai_tts_runner_passes_voice_model_input_and_speed():
    client = _FakeOpenAIClient(_FakeSpeechResponse(b""))
    config = SynthesisConfig(speed=1.5)
    runner = OpenAITTSRunner(voice="coral", model="gpt-4o-mini-tts", config=config, client=client)

    runner.synthesize("some text")

    call = client.audio.speech.calls[0]
    assert call["voice"] == "coral"
    assert call["model"] == "gpt-4o-mini-tts"
    assert call["input"] == "some text"
    assert call["speed"] == 1.5
    assert call["response_format"] == "wav"


# -- OrpheusRunner: prompt assembly (_build_prompt_tokens) -------------------

class _FakeLlama:
    """Stand-in for llama_cpp.Llama - only .tokenize() is exercised by
    _build_prompt_tokens. Returns the raw byte values as "tokens" so
    tests can check exactly what text got tokenized, and records every
    call for order/content assertions."""
    def __init__(self):
        self.tokenize_calls = []

    def tokenize(self, data: bytes, add_bos: bool = False):
        self.tokenize_calls.append(data)
        return list(data)


def test_orpheus_runner_requires_reference_transcript_alongside_reference_audio():
    with pytest.raises(ValueError):
        OrpheusRunner(llama=_FakeLlama(), snac_model=None, reference_audio=[0.0, 0.1])


def test_orpheus_runner_build_prompt_tokens_without_voice_or_reference():
    llama = _FakeLlama()
    runner = OrpheusRunner(llama, snac_model=None)

    tokens = runner._build_prompt_tokens("hi")

    assert llama.tokenize_calls == [b"hi"]
    assert tokens == [START_OF_HUMAN, *list(b"hi"), END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]


def test_orpheus_runner_build_prompt_tokens_prefixes_a_preset_voice_name():
    llama = _FakeLlama()
    runner = OrpheusRunner(llama, snac_model=None, voice="tara")

    runner._build_prompt_tokens("hi")

    assert llama.tokenize_calls == [b"tara: hi"]


def test_orpheus_runner_build_prompt_tokens_includes_the_reference_turn_first():
    llama = _FakeLlama()
    runner = OrpheusRunner(llama, snac_model=None)
    # Bypass __init__'s torch-dependent _encode_reference: set the
    # already-encoded reference directly, since what's under test here
    # is prompt assembly, not SNAC encoding.
    runner.reference_transcript = "ref text"
    runner._reference_frame_tokens = [1, 2, 3, 4, 5, 6, 7]

    tokens = runner._build_prompt_tokens("hi")

    assert llama.tokenize_calls == [b"ref text", b"hi"]
    assert tokens == [
        START_OF_HUMAN, *list(b"ref text"), END_OF_HUMAN,
        START_OF_AI, START_OF_SPEECH, 1, 2, 3, 4, 5, 6, 7, END_OF_SPEECH, END_OF_AI,
        START_OF_HUMAN, *list(b"hi"), END_OF_HUMAN, START_OF_AI, START_OF_SPEECH,
    ]


# -- CometTTSRunner -----------------------------------------------------------

class _FakeTransport:
    """Scripted CometAPI: returns `polls` one by one on get_json (an
    Exception instance is raised instead), records every request."""

    def __init__(self, polls, submit=None):
        self.polls = list(polls)
        self.submit = submit or {"id": "task-1"}
        self.posted = []
        self.downloaded = []

    def post_json(self, path, body):
        self.posted.append((path, body))
        return self.submit

    def get_json(self, path):
        poll = self.polls.pop(0)
        if isinstance(poll, Exception):
            raise poll
        return poll

    def download(self, url):
        self.downloaded.append(url)
        return b"mp3-bytes"


@pytest.fixture
def no_ffmpeg(monkeypatch):
    import lector.ffmpeg

    monkeypatch.setattr(lector.ffmpeg, "audio_to_wav", lambda data, rate: b"WAV:" + data)


def _comet(transport, **kwargs):
    kwargs.setdefault("voice", "Bernard")
    return CometTTSRunner(transport=transport, poll_interval=0, **kwargs)


def test_detect_language_by_dominant_alphabet():
    assert detect_language("Вероятностное пространство and P") == "ru"
    assert detect_language("Probability space, Ω") == "en"
    assert detect_language("123") == "en"


def test_comet_preset_request_carries_voice_language_and_clamped_speed():
    runner = _comet(_FakeTransport([]), config=SynthesisConfig(speed=2.0))

    body = runner.build_request("Привет, мир.")

    assert body == {
        "model": "eleven_v3", "promptText": "Привет, мир.",
        "voice": {"type": "runway-preset", "presetId": "Bernard"},
        "speed": 1.2, "languageCode": "ru",
    }


def test_comet_forced_language_wins_over_detection():
    runner = _comet(_FakeTransport([]), language="en")

    assert runner.build_request("Привет.")["languageCode"] == "en"


def test_comet_multilingual_v2_sends_no_language_code():
    runner = _comet(_FakeTransport([]), model="eleven_multilingual_v2")

    assert "languageCode" not in runner.build_request("Hello.")


def test_comet_cloning_request_inlines_the_reference_recording(tmp_path):
    reference = tmp_path / "me.mp3"
    reference.write_bytes(b"ID3fake")
    runner = CometTTSRunner(model="seed_audio", reference_audio=str(reference),
                            config=SynthesisConfig(speed=0.8), transport=_FakeTransport([]))

    body = runner.build_request("Текст.")

    assert body["voice"] == {"type": "reference-audio", "audioUri": "data:audio/mpeg;base64,SUQzZmFrZQ=="}
    assert body["speechRate"] == -20
    assert body["outputFormat"] == "wav"
    assert "languageCode" not in body


def test_comet_seed_without_reference_uses_default_voice():
    runner = CometTTSRunner(model="seed_audio", transport=_FakeTransport([]))

    assert "voice" not in runner.build_request("Text.")


def test_comet_rejects_inconsistent_voice_settings(tmp_path):
    reference = tmp_path / "me.wav"
    reference.write_bytes(b"RIFF")

    with pytest.raises(ValueError, match="seed_audio"):
        CometTTSRunner(model="eleven_v3", voice="Bernard", reference_audio=str(reference),
                       transport=_FakeTransport([]))
    with pytest.raises(ValueError, match="preset voice"):
        CometTTSRunner(model="eleven_v3", transport=_FakeTransport([]))
    with pytest.raises(ValueError, match="Unknown"):
        CometTTSRunner(model="nope", transport=_FakeTransport([]))


def test_comet_synthesize_submits_polls_downloads_and_decodes(no_ffmpeg):
    transport = _FakeTransport([
        CometHTTPError(400, '{"code":"task_not_exist"}'),
        {"id": "task-1", "status": "RUNNING"},
        {"id": "task-1", "status": "SUCCEEDED", "output": ["https://cdn/x.mp3"]},
    ])

    wav = _comet(transport).synthesize("Hello.")

    assert wav == b"WAV:mp3-bytes"
    assert transport.posted[0][0] == "/runwayml/v1/text_to_speech"
    assert transport.downloaded == ["https://cdn/x.mp3"]


def test_comet_failed_task_raises_with_its_reason(no_ffmpeg):
    transport = _FakeTransport([{"id": "task-1", "status": "FAILED", "failure": "moderation"}])

    with pytest.raises(RuntimeError, match="moderation"):
        _comet(transport).synthesize("Hello.")


def test_comet_client_error_while_polling_is_not_retried(no_ffmpeg):
    transport = _FakeTransport([CometHTTPError(401, "bad key")])

    with pytest.raises(CometHTTPError):
        _comet(transport).synthesize("Hello.")


def test_comet_rejected_submission_raises(no_ffmpeg):
    transport = _FakeTransport([], submit={"error": "insufficient balance"})

    with pytest.raises(RuntimeError, match="insufficient balance"):
        _comet(transport).synthesize("Hello.")


def test_comet_task_that_never_finishes_times_out(no_ffmpeg):
    transport = _FakeTransport([{"id": "task-1", "status": "RUNNING"}] * 1000)
    runner = CometTTSRunner(voice="Bernard", transport=transport, poll_interval=0, task_timeout=0.01)

    with pytest.raises(TimeoutError):
        runner.synthesize("Hello.")


def test_reference_audio_data_uri_rejects_unknown_format_and_oversize(tmp_path):
    odd = tmp_path / "voice.xyz"
    odd.write_bytes(b"x")
    big = tmp_path / "voice.wav"
    big.write_bytes(b"x" * (5 * 1024 * 1024 + 1))

    with pytest.raises(ValueError, match="Unsupported"):
        reference_audio_data_uri(str(odd))
    with pytest.raises(ValueError, match="5 MB"):
        reference_audio_data_uri(str(big))


def test_build_comet_runner_routes_openai_models_to_the_speech_endpoint(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("COMET_API_KEY", "test-key")

    runner = build_comet_runner(model="gpt-4o-mini-tts", voice="onyx")

    assert isinstance(runner, OpenAITTSRunner)
    assert str(runner._client.base_url).rstrip("/") == "https://api.cometapi.com/v1"


def test_build_comet_runner_accepts_cometapi_key_spelling(monkeypatch):
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    monkeypatch.setenv("COMETAPI_KEY", "test-key")

    assert isinstance(build_comet_runner(model="eleven_v3", voice="Bernard"), CometTTSRunner)


def test_build_comet_runner_without_any_key_fails(monkeypatch):
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    monkeypatch.delenv("COMETAPI_KEY", raising=False)

    with pytest.raises(ValueError, match="COMET_API_KEY"):
        build_comet_runner(model="eleven_v3", voice="Bernard")


# -- CachedTTSRunner ----------------------------------------------------------

class _CountingRunner(BaseTTSRunner):
    def __init__(self):
        self.calls = []
        self.closed = False

    def synthesize(self, text):
        self.calls.append(text)
        return f"wav:{text}".encode()

    def close(self):
        self.closed = True


def test_cached_runner_synthesizes_each_text_once_per_namespace(tmp_path):
    inner = _CountingRunner()
    runner = CachedTTSRunner(inner, str(tmp_path), namespace="voice-a")

    assert runner.synthesize("one") == b"wav:one"
    assert runner.synthesize("one") == b"wav:one"
    assert inner.calls == ["one"]
    assert runner.hits == 1

    other_voice = CachedTTSRunner(inner, str(tmp_path), namespace="voice-b")
    other_voice.synthesize("one")
    assert inner.calls == ["one", "one"]


def test_cached_runner_closes_the_wrapped_runner(tmp_path):
    inner = _CountingRunner()

    with CachedTTSRunner(inner, str(tmp_path), namespace="n"):
        pass

    assert inner.closed
