"""The two audio-format conversions lector needs, both through the ffmpeg
binary rather than a Python audio library: decoding whatever a hosted TTS
backend returned (mp3, wav at an arbitrary rate) into the one PCM format
lector.audio concatenates, and encoding the final WAV into a listenable
format (mp3, m4a, ...).

Decoding goes to raw s16le PCM, not to WAV, on purpose: ffmpeg writing a
WAV to a pipe can't seek back to fill in the header's length fields, so
the stdlib `wave` module would read a bogus frame count. Raw PCM wrapped
in a header written here has no such ambiguity.
"""
from __future__ import annotations

import io
import shutil
import subprocess
import wave
from pathlib import Path

DEFAULT_SAMPLE_RATE = 24000


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH - it's needed to decode hosted "
                           "TTS output and to encode the final audio file.")
    return path


def audio_to_wav(data: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Decode any ffmpeg-readable audio bytes into a mono 16-bit WAV at
    `sample_rate` - so chunks from one backend always share a format,
    whatever container/rate the backend happened to return them in."""
    result = subprocess.run(
        [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"],
        input=data, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode the audio: {result.stderr.decode(errors='replace').strip()}")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(result.stdout)
    return buffer.getvalue()


def encode_audio(wav_path: "str | Path", output_path: "str | Path") -> None:
    """Encode a WAV file into whatever format `output_path`'s extension
    names (.mp3, .m4a, .ogg, ...), letting ffmpeg pick the codec."""
    result = subprocess.run(
        [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_path),
         "-q:a", "2", str(output_path)],
        capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not encode {output_path}: "
                           f"{result.stderr.decode(errors='replace').strip()}")
