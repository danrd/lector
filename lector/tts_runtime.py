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

OrpheusRunner is a second local option, for when voice cloning matters
more than raw speed: it wraps a GGUF speech-LLM through llama_cpp.Llama
- the exact same class llm_kit's own LlamaCppRunner wraps, just a
different model - so CPU vs GPU is the same n_gpu_layers knob already
used there, not a separate code path. See lector/orpheus_codec.py for
the (fully unit-tested, torch-free) token<->SNAC-codes math this relies
on. UNVERIFIED AGAINST A REAL MODEL: built from a researched reference
implementation (github.com/Zuellni/Orpheus-GGUF), not yet run end to
end against real Orpheus GGUF + SNAC checkpoints - treat as a starting
point to validate once those are downloaded, not as confirmed-working.

Heavy/optional dependencies (piper, openai, llama_cpp, torch, snac) are
imported lazily inside whichever class actually needs them, so
importing this module never requires every backend's library to be
installed.
"""
from __future__ import annotations

import io
import os
import wave
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from lector.orpheus_codec import (
    END_OF_AI,
    END_OF_HUMAN,
    END_OF_SPEECH,
    START_OF_AI,
    START_OF_HUMAN,
    START_OF_SPEECH,
    frame_tokens_to_snac_codes,
    snac_codes_to_frame_tokens,
)


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


# ---------------------------------------------------------------------------
# Local backend (voice cloning) - Orpheus GGUF via llama_cpp + SNAC codec.
# ---------------------------------------------------------------------------

class OrpheusRunner(BaseTTSRunner):
    """Wraps an in-process Orpheus GGUF model (llama_cpp.Llama) plus a
    SNAC codec to turn its output tokens into audio. Same architecture
    as llm_kit's LlamaCppRunner - just a different GGUF model, with a
    codec step where plain text decoding would be - so CPU vs GPU is
    the n_gpu_layers knob already used there, not a separate path.

    Voice is fixed per instance, same as PiperRunner is locked to one
    loaded voice model: either a preset speaker name (pass
    `voice="tara"` etc, per Orpheus's published "name: text" prompt
    convention), or a cloned voice - pass `reference_audio` (a mono
    float32 waveform at 24kHz, matching SNAC's sample rate) and
    `reference_transcript` (the words spoken in it). Every
    synthesize() call after that continues in the reference voice.
    """

    def __init__(self, llama, snac_model, snac_device: str = "cpu",
                 voice: Optional[str] = None,
                 reference_audio=None, reference_transcript: Optional[str] = None,
                 config: Optional[SynthesisConfig] = None, max_tokens: int = 4096):
        if reference_audio is not None and reference_transcript is None:
            raise ValueError("reference_transcript is required alongside reference_audio")

        self.llama = llama
        self.snac_model = snac_model
        self.snac_device = snac_device
        self.voice = voice
        self.reference_transcript = reference_transcript
        self.config = config or SynthesisConfig()
        self.max_tokens = max_tokens
        self._reference_frame_tokens = (
            self._encode_reference(reference_audio) if reference_audio is not None else None
        )

    def _encode_reference(self, audio) -> List[int]:
        import torch

        with torch.no_grad():
            waveform = torch.as_tensor(audio, dtype=torch.float32, device=self.snac_device).reshape(1, 1, -1)
            codes = self.snac_model.encode(waveform)
        layer0 = codes[0][0].tolist()
        layer1 = codes[1][0].tolist()
        layer2 = codes[2][0].tolist()
        return snac_codes_to_frame_tokens(layer0, layer1, layer2)

    def _build_prompt_tokens(self, text: str) -> List[int]:
        """Assemble the Llama-token prompt: a plain human/AI turn
        (optionally prefixed with a preset voice name), preceded - when
        a reference voice was given - by that reference as a prior
        human-text-then-AI-speech turn, so the model continues in the
        same voice for the new turn. Only calls self.llama.tokenize(),
        so this is testable with a fake llama stub - no real GGUF model
        needed to check the prompt is assembled correctly."""
        if self.reference_transcript is not None:
            reference_tokens = self.llama.tokenize(self.reference_transcript.encode("utf-8"), add_bos=False)
            prior_turn = [
                START_OF_HUMAN, *reference_tokens, END_OF_HUMAN,
                START_OF_AI, START_OF_SPEECH, *self._reference_frame_tokens, END_OF_SPEECH, END_OF_AI,
            ]
        else:
            prior_turn = []

        prompt_text = f"{self.voice}: {text}" if self.voice else text
        text_tokens = self.llama.tokenize(prompt_text.encode("utf-8"), add_bos=False)
        new_turn = [START_OF_HUMAN, *text_tokens, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
        return [*prior_turn, *new_turn]

    def synthesize(self, text: str) -> bytes:
        import torch

        prompt_tokens = self._build_prompt_tokens(text)
        generated: List[int] = []
        for token in self.llama.generate(
            prompt_tokens, top_k=40, top_p=0.9, temp=0.6, repeat_penalty=1.1,
        ):
            if token == END_OF_SPEECH or len(generated) >= self.max_tokens:
                break
            generated.append(token)

        # SNAC frames are 7 tokens each - drop any trailing partial
        # frame rather than letting it corrupt the un-interleaving.
        usable_len = len(generated) - (len(generated) % 7)
        layer0, layer1, layer2 = frame_tokens_to_snac_codes(generated[:usable_len])

        with torch.no_grad():
            codes = [
                torch.tensor([layer0], device=self.snac_device),
                torch.tensor([layer1], device=self.snac_device),
                torch.tensor([layer2], device=self.snac_device),
            ]
            waveform = self.snac_model.decode(codes)

        samples = (waveform.squeeze().clamp(-1, 1).cpu().numpy() * 32767).astype("int16")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)  # SNAC's native rate for the 24kHz checkpoint
            wav_file.writeframes(samples.tobytes())
        return buffer.getvalue()

    def close(self) -> None:
        self.llama = None
        self.snac_model = None
        import gc
        gc.collect()


def build_orpheus_runner(model_path: str, device: str = "cpu",
                          voice: Optional[str] = None, reference_audio=None,
                          reference_transcript: Optional[str] = None,
                          config: Optional[SynthesisConfig] = None) -> OrpheusRunner:
    """Load an Orpheus GGUF model plus the SNAC codec, on CPU or GPU per
    `device` ("cpu" | "gpu") - mirrors llm_kit.llm_setup's own device
    split (n_gpu_layers for the GGUF model, a torch device for SNAC),
    not a separate implementation per device."""
    from llama_cpp import Llama
    from snac import SNAC

    n_gpu_layers = -1 if device == "gpu" else 0
    llama = Llama(model_path=model_path, n_ctx=8192, n_gpu_layers=n_gpu_layers, verbose=False)

    snac_device = "cuda" if device == "gpu" else "cpu"
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(snac_device).eval()

    return OrpheusRunner(llama, snac_model, snac_device=snac_device, voice=voice,
                          reference_audio=reference_audio, reference_transcript=reference_transcript,
                          config=config)
