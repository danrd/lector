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

CometTTSRunner is the hosted option that does both quality and cloning:
CometAPI's Runway-format speech route, with ElevenLabs preset voices
(the most natural prosody tried so far) or seed_audio cloning a voice
from a short local recording. Verified end to end against the live API
on Russian vault notes. CachedTTSRunner wraps any backend so a re-run
only pays for chunks whose text changed.

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
                 api_key: Optional[str] = None, client=None, base_url: Optional[str] = None):
        self.voice = voice
        self.model = model
        self.config = config or SynthesisConfig()
        self._client = client  # allows injecting a fake client for testing
        if self._client is None:
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            from openai import OpenAI
            # base_url: any OpenAI-compatible /audio/speech host - e.g.
            # CometAPI (see build_comet_runner) - not just OpenAI itself.
            self._client = OpenAI(api_key=api_key, base_url=base_url)

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


# ---------------------------------------------------------------------------
# Hosted backend (voice cloning) - CometAPI's /runwayml/v1/text_to_speech.
# ---------------------------------------------------------------------------

COMET_BASE_URL = "https://api.cometapi.com"

# OpenAI's own speech models on CometAPI go through the synchronous,
# OpenAI-compatible /v1/audio/speech instead - see build_comet_runner.
COMET_OPENAI_MODELS = ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")
COMET_PRESET_MODELS = ("eleven_v3", "eleven_v4", "eleven_multilingual_v2")
COMET_CLONING_MODEL = "seed_audio"

# Longest text each model accepts per request is 1000 (multilingual_v2),
# 5000 (v3), unlimited (v4) and 2048 (seed_audio) characters; these sit
# well below that - shorter chunks also mean more of them run in
# parallel, and each request takes about a minute whatever its length.
# seed_audio's is much lower than its documented limit for a measured
# reason: its output stops at roughly a minute of speech and silently
# drops the rest of the text (1500-character Russian chunks came back as
# ~54 s clips missing a third of their words), so a chunk has to stay
# well under a minute of reading - ~500 characters is ~30 s.
COMET_MAX_CHUNK_CHARS = {
    "eleven_multilingual_v2": 900, "eleven_v3": 1500, "eleven_v4": 1500, "seed_audio": 500,
}

_REFERENCE_MAX_BYTES = 5 * 1024 * 1024
_REFERENCE_MIME = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
                   ".m4a": "audio/mp4", ".flac": "audio/flac", ".webm": "audio/webm"}


def comet_api_key(api_key: Optional[str] = None) -> str:
    """CometAPI's own docs call the variable COMETAPI_KEY; accept that
    spelling too, not only COMET_API_KEY."""
    api_key = api_key or os.environ.get("COMET_API_KEY") or os.environ.get("COMETAPI_KEY")
    if not api_key:
        raise ValueError("COMET_API_KEY is not set")
    return api_key


def detect_language(text: str) -> str:
    """"ru" or "en" by which alphabet dominates `text` - all the vault is
    written in, and enough for ElevenLabs' languageCode hint."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "en"
    cyrillic = sum("Ѐ" <= ch <= "ӿ" for ch in letters)
    return "ru" if cyrillic / len(letters) >= 0.3 else "en"


def reference_audio_data_uri(path: str) -> str:
    """A local recording as the base64 data URI seed_audio accepts as a
    voice reference - so cloning needs no hosting of the sample."""
    import base64
    from pathlib import Path

    suffix = Path(path).suffix.lower()
    if suffix not in _REFERENCE_MIME:
        raise ValueError(f"Unsupported reference audio format {suffix!r} "
                         f"(expected one of {', '.join(sorted(_REFERENCE_MIME))})")
    data = Path(path).read_bytes()
    if len(data) > _REFERENCE_MAX_BYTES:
        raise ValueError(f"Reference audio is {len(data) / 2**20:.1f} MB; the API accepts at most 5 MB "
                         "- a 15-30 second clip is plenty.")
    return f"data:{_REFERENCE_MIME[suffix]};base64,{base64.b64encode(data).decode('ascii')}"


class CometHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"CometAPI returned HTTP {status}: {body[:500]}")


class _UrllibTransport:
    """The three HTTP calls CometTTSRunner makes, on the stdlib alone -
    swapped for a fake in tests."""

    def __init__(self, api_key: str, base_url: str = COMET_BASE_URL, timeout: float = 120.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, url: str, body: Optional[dict] = None, auth: bool = True) -> bytes:
        import json
        import urllib.error
        import urllib.request

        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url, method=method, headers=headers,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise CometHTTPError(exc.code, exc.read().decode("utf-8", errors="replace")) from exc

    def post_json(self, path: str, body: dict) -> dict:
        import json
        return json.loads(self._request("POST", self.base_url + path, body))

    def get_json(self, path: str) -> dict:
        import json
        return json.loads(self._request("GET", self.base_url + path))

    def download(self, url: str) -> bytes:
        # Result URLs are pre-signed storage links - no API key goes there.
        return self._request("GET", url, auth=False)


class CometTTSRunner(BaseTTSRunner):
    """Speech through CometAPI's Runway-format TTS route: submit a task,
    poll it, download the result, decode it to WAV.

    Two kinds of voice, by model:
    - `eleven_v3` / `eleven_v4` / `eleven_multilingual_v2` (ElevenLabs) -
      one of Runway's preset voices, `voice="Bernard"` etc. Best prosody;
      no cloning on this route.
    - `seed_audio` - the default voice, or a cloned one: pass
      `reference_audio` (path to a short clean recording of the target
      voice, sent inline as base64; no transcript needed).

    Every chunk is decoded to the same mono 24 kHz WAV (lector.ffmpeg),
    whatever container the service returned, so lector.audio's
    concatenation never sees mismatched formats. synthesize() keeps no
    per-call state, so lector.audio may call it from several threads -
    which matters here, since one request takes about a minute.
    """

    def __init__(self, model: str = "eleven_v3", voice: Optional[str] = None,
                 reference_audio: Optional[str] = None, language: Optional[str] = None,
                 config: Optional[SynthesisConfig] = None, api_key: Optional[str] = None,
                 transport=None, poll_interval: float = 3.0, task_timeout: float = 600.0,
                 sample_rate: int = 24000):
        if model not in (*COMET_PRESET_MODELS, COMET_CLONING_MODEL):
            raise ValueError(f"Unknown CometAPI speech model {model!r}")
        if reference_audio is not None and model != COMET_CLONING_MODEL:
            raise ValueError(f"Voice cloning needs model={COMET_CLONING_MODEL!r}; {model} only has preset voices")
        if model in COMET_PRESET_MODELS and not voice:
            raise ValueError(f"{model} needs a preset voice name (e.g. 'Bernard')")

        self.model = model
        self.voice = voice
        self.language = language
        self.config = config or SynthesisConfig()
        self.poll_interval = poll_interval
        self.task_timeout = task_timeout
        self.sample_rate = sample_rate
        self._reference_uri = reference_audio_data_uri(reference_audio) if reference_audio else None
        self._transport = transport or _UrllibTransport(comet_api_key(api_key))

    def build_request(self, text: str) -> dict:
        body: dict = {"model": self.model, "promptText": text}
        speed = self.config.speed
        if self.model == COMET_CLONING_MODEL:
            if self._reference_uri:
                body["voice"] = {"type": "reference-audio", "audioUri": self._reference_uri}
            # speechRate is a -50..100 percentage offset from normal speed.
            rate = max(-50, min(100, round((speed - 1.0) * 100)))
            if rate:
                body["speechRate"] = rate
            body["outputFormat"] = "wav"
            body["sampleRate"] = self.sample_rate
        else:
            body["voice"] = {"type": "runway-preset", "presetId": self.voice}
            if speed != 1.0:
                body["speed"] = max(0.7, min(1.2, speed))
            if self.model != "eleven_multilingual_v2":
                body["languageCode"] = self.language or detect_language(text)
        return body

    def _wait_for(self, task_id: str) -> dict:
        import time

        deadline = time.monotonic() + self.task_timeout
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            try:
                task = self._transport.get_json(f"/runwayml/v1/tasks/{task_id}")
            except CometHTTPError as exc:
                # A just-submitted task can briefly 400 as task_not_exist,
                # and the proxy occasionally 5xx's - both are worth waiting out.
                if exc.status >= 500 or "task_not_exist" in exc.body:
                    continue
                raise
            status = task.get("status")
            if status == "SUCCEEDED":
                return task
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"CometAPI speech task {task_id} {status.lower()}: "
                                   f"{task.get('failure') or task.get('failureCode') or 'no reason given'}")
        raise TimeoutError(f"CometAPI speech task {task_id} didn't finish within {self.task_timeout:.0f}s")

    def synthesize(self, text: str) -> bytes:
        from lector.ffmpeg import audio_to_wav

        submitted = self._transport.post_json("/runwayml/v1/text_to_speech", self.build_request(text))
        if "id" not in submitted:
            raise RuntimeError(f"CometAPI didn't accept the speech task: {submitted}")
        task = self._wait_for(submitted["id"])
        if not task.get("output"):
            raise RuntimeError(f"CometAPI speech task {submitted['id']} succeeded with no output")
        return audio_to_wav(self._transport.download(task["output"][0]), self.sample_rate)


def build_comet_runner(model: str = "eleven_v3", voice: Optional[str] = None,
                        reference_audio: Optional[str] = None, language: Optional[str] = None,
                        config: Optional[SynthesisConfig] = None,
                        api_key: Optional[str] = None) -> BaseTTSRunner:
    """Any CometAPI speech model behind one constructor: OpenAI's models
    through OpenAITTSRunner pointed at CometAPI (synchronous, no cloning),
    everything else through CometTTSRunner."""
    if model in COMET_OPENAI_MODELS:
        if reference_audio is not None:
            raise ValueError(f"{model} can't clone a voice - use model={COMET_CLONING_MODEL!r}")
        return OpenAITTSRunner(voice=voice or "alloy", model=model, config=config,
                               api_key=comet_api_key(api_key), base_url=f"{COMET_BASE_URL}/v1")
    return CometTTSRunner(model=model, voice=voice, reference_audio=reference_audio,
                          language=language, config=config, api_key=api_key)


# ---------------------------------------------------------------------------
# Chunk cache - wraps any backend.
# ---------------------------------------------------------------------------

class CachedTTSRunner(BaseTTSRunner):
    """Keeps every synthesized chunk on disk, keyed by the text and
    `namespace` (whatever identifies the voice: backend, model, voice,
    reference file). A run that fails halfway, or a re-run after fixing
    one line of the script, then only pays for chunks it hasn't already
    got - which matters for hosted backends that bill per request."""

    def __init__(self, inner: BaseTTSRunner, cache_dir: str, namespace: str):
        from pathlib import Path

        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.namespace = namespace
        self.hits = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, text: str):
        import hashlib

        key = hashlib.sha256(f"{self.namespace}\0{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.wav"

    def synthesize(self, text: str) -> bytes:
        path = self._path(text)
        if path.exists():
            self.hits += 1
            return path.read_bytes()
        data = self.inner.synthesize(text)
        partial = path.with_suffix(".partial")
        partial.write_bytes(data)
        partial.replace(path)  # atomic: a killed run never leaves a truncated chunk behind
        return data

    def close(self) -> None:
        self.inner.close()
