"""Universal TTS synthesis runner: the "how to synthesize" layer only.

One interface regardless of backend: `BaseTTSRunner.synthesize(text: str)
-> bytes`, always the bytes of a single WAV file - callers (lector.audio)
never need to know or care whether that came from a local ONNX model or
an HTTP call to a hosted API.

Local-first, cloud as an explicit opt-in: PiperRunner (in-process, CPU
or CUDA via onnxruntime-gpu, no network dependency once a voice model
is on disk) is the runner meant for regular use - narrating a lecture
shouldn't require paying per character or sending the text to a third
party by default. OpenAITTSRunner exists for when local quality/speed
isn't enough for a particular run; nothing here selects it automatically,
the caller constructs it directly, the same separation llm_kit draws
between build_runner (local) and OpenRouterRunner (hosted).

Heavy/optional dependencies (piper, openai) are imported lazily inside
whichever class actually needs them, so importing this module never
requires every backend's library to be installed.
"""
from __future__ import annotations

import io
import os
import wave
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SynthesisConfig(BaseModel):
    """Framework-agnostic voice parameters, translated per-backend."""
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    speed: float = Field(default=1.0, gt=0.0,
                          description="Speaking-rate multiplier: >1.0 faster, <1.0 slower.")
    volume: float = Field(default=1.0, ge=0.0, description="Output volume scale.")


class BaseTTSRunner:
    """Common interface for every TTS backend. Use as a context manager
    to guarantee any loaded model / client is released:
        with build_piper_runner(voice_path) as runner:
            wav_bytes = runner.synthesize(text)
    """

    def synthesize(self, text: str) -> bytes:
        """Return one WAV file's bytes for `text`, synthesized in a
        single call - callers with long text should chunk it first
        (see lector.audio.synthesize_long_text) rather than relying on
        a particular backend's tolerance for large inputs."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "BaseTTSRunner":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Local backend (default) - piper-tts, in-process ONNX model.
# ---------------------------------------------------------------------------

class PiperRunner(BaseTTSRunner):
    """Wraps an in-process piper.PiperVoice. No network calls once the
    voice model (a .onnx file plus its matching .onnx.json) is on disk."""

    def __init__(self, voice, config: Optional[SynthesisConfig] = None):
        self.voice = voice
        self.config = config or SynthesisConfig()

    def synthesize(self, text: str) -> bytes:
        from piper import SynthesisConfig as PiperSynthesisConfig

        # Piper's length_scale stretches phoneme duration, so it's the
        # inverse of a speed multiplier: bigger scale = slower speech.
        syn_config = PiperSynthesisConfig(
            length_scale=1.0 / self.config.speed, volume=self.config.volume,
        )
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        return buffer.getvalue()

    def close(self) -> None:
        self.voice = None


def build_piper_runner(model_path: str, config: Optional[SynthesisConfig] = None,
                        use_cuda: bool = False) -> PiperRunner:
    """Load a Piper voice from a local .onnx model file (its matching
    .onnx.json config must sit alongside it) and wrap it as a runner."""
    from piper import PiperVoice

    voice = PiperVoice.load(model_path, use_cuda=use_cuda)
    return PiperRunner(voice, config)


# ---------------------------------------------------------------------------
# Cloud backend (explicit opt-in) - OpenAI TTS.
# ---------------------------------------------------------------------------

class OpenAITTSRunner(BaseTTSRunner):
    """Hosted option for when local voice quality/speed isn't enough for
    a particular run. Never selected implicitly - constructed directly
    by whichever caller decided to opt into it."""

    def __init__(self, voice: str = "alloy", model: str = "gpt-4o-mini-tts",
                 config: Optional[SynthesisConfig] = None,
                 api_key: Optional[str] = None, client=None):
        self.voice = voice
        self.model = model
        self.config = config or SynthesisConfig()
        self._client = client  # allows injecting a fake client for testing
        if self._client is None:
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)

    def synthesize(self, text: str) -> bytes:
        response = self._client.audio.speech.create(
            model=self.model, voice=self.voice, input=text,
            response_format="wav", speed=self.config.speed,
        )
        return response.content
