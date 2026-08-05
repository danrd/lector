"""Turn arbitrary-length narration text into a single WAV file.

No backend in tts_runtime.py is asked to synthesize an entire lecture
in one call: Piper has no hard input limit but a single multi-minute
call is slow and fragile to interrupt/retry, and hosted APIs like
OpenAI TTS cap `input` well below that. So text is split into
sentence-bounded chunks first, each chunk is synthesized separately,
and the resulting WAV chunks are concatenated into one output file.

Concatenation trusts that every chunk came from the same runner/voice
and therefore shares one sample rate/width/channel count - it doesn't
attempt to resample mismatched chunks, it raises instead.
"""
from __future__ import annotations

import io
import re
import wave
from dataclasses import dataclass
from typing import List

from lector.tts_runtime import BaseTTSRunner

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class AudioResult:
    output_path: str
    num_chunks: int
    duration_seconds: float


def split_into_chunks(text: str, max_chunk_chars: int = 1000) -> List[str]:
    """Pack sentences into chunks no longer than `max_chunk_chars`,
    never splitting a sentence across two chunks. A single sentence
    longer than the limit becomes its own oversized chunk rather than
    being cut mid-word - correctness of the audio matters more than
    strictly respecting the cap in that rare case."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        added_len = len(sentence) + (1 if current else 0)  # +1 for the joining space
        if current and current_len + added_len > max_chunk_chars:
            chunks.append(" ".join(current))
            current, current_len = [], 0
            added_len = len(sentence)
        current.append(sentence)
        current_len += added_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def _concatenate_wavs(wav_bytes_list: List[bytes], output_path: str) -> float:
    """Write every chunk's PCM frames into one WAV file at
    `output_path`, in order. Returns the total duration in seconds."""
    params = None
    total_frames = 0

    with wave.open(output_path, "wb") as out_file:
        for wav_bytes in wav_bytes_list:
            with wave.open(io.BytesIO(wav_bytes), "rb") as chunk_file:
                chunk_params = chunk_file.getparams()
                if params is None:
                    params = chunk_params
                    out_file.setparams((
                        params.nchannels, params.sampwidth, params.framerate,
                        0, params.comptype, params.compname,
                    ))
                elif (chunk_params.nchannels, chunk_params.sampwidth, chunk_params.framerate) != \
                        (params.nchannels, params.sampwidth, params.framerate):
                    raise ValueError(
                        "Audio chunks have mismatched format (channels/sample width/rate): "
                        f"{chunk_params} does not match the first chunk's {params}. "
                        "All chunks must come from the same runner/voice."
                    )
                out_file.writeframes(chunk_file.readframes(chunk_file.getnframes()))
                total_frames += chunk_file.getnframes()

    return total_frames / params.framerate if params else 0.0


def synthesize_long_text(text: str, runner: BaseTTSRunner, output_path: str,
                          max_chunk_chars: int = 1000) -> AudioResult:
    """Synthesize arbitrary-length `text` to a single WAV file at
    `output_path`, chunking internally so no backend ever gets a single
    oversized synthesis call."""
    chunks = split_into_chunks(text, max_chunk_chars=max_chunk_chars)
    if not chunks:
        raise ValueError("Nothing to synthesize: text is empty after stripping/splitting.")

    wav_bytes_list = [runner.synthesize(chunk) for chunk in chunks]
    duration = _concatenate_wavs(wav_bytes_list, output_path)
    return AudioResult(output_path=output_path, num_chunks=len(chunks), duration_seconds=duration)
