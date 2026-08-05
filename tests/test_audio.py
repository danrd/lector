"""Tests for lector/audio.py.

Chunking is pure text logic (no fixture needed). Concatenation is
tested against real, small WAV byte-strings built with the stdlib
`wave` module directly - no need for piper/openai or any actual TTS
call, since what's under test is the chunk-splitting and PCM-stitching
logic, not synthesis quality.
"""
from __future__ import annotations

import io
import wave

import pytest

from lector.audio import AudioResult, split_into_chunks, synthesize_long_text
from lector.tts_runtime import BaseTTSRunner


def _make_wav_bytes(num_frames: int, framerate: int = 16000, nchannels: int = 1, sampwidth: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(nchannels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(framerate)
        wav_file.writeframes(b"\x00" * (num_frames * nchannels * sampwidth))
    return buffer.getvalue()


class _FakeTTSRunner(BaseTTSRunner):
    """Returns one second of silence per chunk, at a framerate keyed off
    the chunk's length so tests can tell chunks apart if needed; records
    every chunk of text it was asked to synthesize."""
    def __init__(self, framerate: int = 16000):
        self.framerate = framerate
        self.calls = []

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return _make_wav_bytes(self.framerate, framerate=self.framerate)


class _MismatchedTTSRunner(BaseTTSRunner):
    """Simulates a backend whose output format changes between calls -
    the failure mode _concatenate_wavs must refuse to paper over."""
    def __init__(self):
        self.call_count = 0

    def synthesize(self, text: str) -> bytes:
        self.call_count += 1
        framerate = 16000 if self.call_count == 1 else 22050
        return _make_wav_bytes(framerate, framerate=framerate)


# -- split_into_chunks --------------------------------------------------------

def test_split_into_chunks_packs_short_sentences_into_one_chunk():
    text = "First sentence. Second sentence. Third sentence."

    chunks = split_into_chunks(text, max_chunk_chars=1000)

    assert chunks == ["First sentence. Second sentence. Third sentence."]


def test_split_into_chunks_starts_a_new_chunk_once_the_limit_would_be_exceeded():
    text = "AAAAAAAAAA. BBBBBBBBBB. CCCCCCCCCC."

    chunks = split_into_chunks(text, max_chunk_chars=15)

    assert chunks == ["AAAAAAAAAA.", "BBBBBBBBBB.", "CCCCCCCCCC."]


def test_split_into_chunks_never_splits_a_single_sentence_even_if_oversized():
    text = "This one sentence alone is longer than the tiny limit given below."

    chunks = split_into_chunks(text, max_chunk_chars=10)

    assert chunks == [text]


def test_split_into_chunks_returns_empty_list_for_empty_text():
    assert split_into_chunks("   ", max_chunk_chars=1000) == []


# -- synthesize_long_text ---------------------------------------------------

def test_synthesize_long_text_calls_the_runner_once_per_chunk(tmp_path):
    text = "First sentence. Second sentence. Third sentence."
    runner = _FakeTTSRunner()
    output_path = str(tmp_path / "out.wav")

    synthesize_long_text(text, runner, output_path, max_chunk_chars=15)

    assert len(runner.calls) == 3


def test_synthesize_long_text_writes_a_single_playable_wav(tmp_path):
    text = "First sentence. Second sentence."
    runner = _FakeTTSRunner(framerate=16000)
    output_path = str(tmp_path / "out.wav")

    result = synthesize_long_text(text, runner, output_path, max_chunk_chars=1000)

    assert isinstance(result, AudioResult)
    assert result.output_path == output_path
    with wave.open(output_path, "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 16000  # one chunk -> one second of silence


def test_synthesize_long_text_concatenates_frames_across_chunks(tmp_path):
    text = "One. Two. Three. Four."
    runner = _FakeTTSRunner(framerate=8000)
    output_path = str(tmp_path / "out.wav")

    result = synthesize_long_text(text, runner, output_path, max_chunk_chars=4)

    assert result.num_chunks == 4
    assert result.duration_seconds == pytest.approx(4.0)
    with wave.open(output_path, "rb") as wav_file:
        assert wav_file.getnframes() == 8000 * 4


def test_synthesize_long_text_raises_on_mismatched_chunk_formats(tmp_path):
    runner = _MismatchedTTSRunner()
    output_path = str(tmp_path / "out.wav")

    with pytest.raises(ValueError, match="mismatched format"):
        synthesize_long_text("First sentence. Second sentence.", runner, output_path, max_chunk_chars=5)


def test_synthesize_long_text_raises_on_empty_text(tmp_path):
    with pytest.raises(ValueError):
        synthesize_long_text("   ", _FakeTTSRunner(), str(tmp_path / "out.wav"))
