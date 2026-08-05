"""Tests for lector/tts_runtime.py.

PiperRunner/build_piper_runner aren't unit-tested here, matching how
llm_kit's own in-process backends (LlamaCppRunner, VLLMRunner, HFRunner)
aren't either: they wrap a real heavy/optional dependency (piper-tts +
an actual voice model file) with no meaningful fake to substitute
in-process. What IS tested: the pure config/interface pieces, and
OpenAITTSRunner, which - like ServerRunner/OpenRouterRunner - accepts
an injected `client` specifically so its request-building logic can be
tested without a real API key or network call.

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
from lector.tts_runtime import BaseTTSRunner, OpenAITTSRunner, OrpheusRunner, SynthesisConfig


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
