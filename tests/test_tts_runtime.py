"""Tests for lector/tts_runtime.py.

PiperRunner/build_piper_runner aren't unit-tested here, matching how
llm_kit's own in-process backends (LlamaCppRunner, VLLMRunner, HFRunner)
aren't either: they wrap a real heavy/optional dependency (piper-tts +
an actual voice model file) with no meaningful fake to substitute
in-process. What IS tested: the pure config/interface pieces, and
OpenAITTSRunner, which - like ServerRunner/OpenRouterRunner - accepts
an injected `client` specifically so its request-building logic can be
tested without a real API key or network call.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from lector.tts_runtime import BaseTTSRunner, OpenAITTSRunner, SynthesisConfig


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
