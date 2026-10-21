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
from typing import Dict, List, Optional

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


def _metadata_flags(metadata: Optional[Dict[str, str]]) -> List[str]:
    flags: List[str] = []
    for key, value in (metadata or {}).items():
        flags += ["-metadata", f"{key}={value}"]
    if metadata:
        flags += ["-id3v2_version", "3"]  # ID3v2.3: what Android players read most reliably
    return flags


def encode_audio(wav_path: "str | Path", output_path: "str | Path",
                 metadata: Optional[Dict[str, str]] = None) -> None:
    """Encode a WAV file into whatever format `output_path`'s extension
    names (.mp3, .m4a, .ogg, ...), letting ffmpeg pick the codec.
    `metadata` becomes tags (title, album, track, ...) - what audiobook
    players use to show a folder as a book with chapters."""
    result = subprocess.run(
        [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_path),
         "-q:a", "2", *_metadata_flags(metadata), str(output_path)],
        capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not encode {output_path}: "
                           f"{result.stderr.decode(errors='replace').strip()}")


def retag_audio(source: "str | Path", target: "str | Path", metadata: Dict[str, str]) -> None:
    """Copy `source` to `target` with new tags, without re-encoding - for a
    section that only moved (new track number) or was renamed."""
    target = Path(target)
    partial = target.with_name(target.stem + ".partial" + target.suffix)
    result = subprocess.run(
        [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-map_metadata", "-1", "-c", "copy", *_metadata_flags(metadata), str(partial)],
        capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not retag {source}: {result.stderr.decode(errors='replace').strip()}")
    partial.replace(target)
