"""Tests for lector/pipeline.py.

No real LLM/TTS - a queue-based fake LLM runner and a fake TTS runner
stand in, since what's under test is the wiring between stages (does
verification actually run before narration, does a soft failure at one
stage stop the pipeline instead of limping forward with garbage) not
generation/synthesis quality.
"""
from __future__ import annotations

import io
import wave

from lector.pipeline import LectureAudioResult, PipelineConfig, PipelineStage, generate_lecture_audio
from lector.lecture import LectureConfig
from lector.narration import NarrationConfig
from lector.tts_runtime import BaseTTSRunner


class _FakeTokenizer:
    def tokenize(self, text):
        return text.split()


class _FakeRunner:
    """Returns queued responses in order; records every prompt it was
    called with."""
    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self._responses.pop(0)


def _make_wav_bytes(num_frames=8000, framerate=8000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(b"\x00" * (num_frames * 2))
    return buffer.getvalue()


class _FakeTTSRunner(BaseTTSRunner):
    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return _make_wav_bytes()


def test_generate_lecture_audio_stops_at_drafting_when_prompt_does_not_fit(tmp_path):
    config = PipelineConfig(lecture=LectureConfig(token_limit=1))
    runner = _FakeRunner()
    tts_runner = _FakeTTSRunner()

    result = generate_lecture_audio(
        "A topic way too long for a token_limit of one", [], _FakeTokenizer(), runner,
        tts_runner, str(tmp_path / "out.wav"), config,
    )

    assert result.stage_reached == PipelineStage.DRAFTING
    assert result.draft is None
    assert runner.calls == []
    assert tts_runner.calls == []


def test_generate_lecture_audio_stops_at_narration_when_prompt_does_not_fit(tmp_path):
    config = PipelineConfig(narration=NarrationConfig(token_limit=1))
    runner = _FakeRunner(responses=["A clean draft with no flagged claims."])
    tts_runner = _FakeTTSRunner()

    result = generate_lecture_audio(
        "Some topic", [], _FakeTokenizer(), runner, tts_runner, str(tmp_path / "out.wav"), config,
    )

    assert result.stage_reached == PipelineStage.NARRATION
    assert result.draft == "A clean draft with no flagged claims."
    assert result.narration is not None
    assert result.narration.script == ""
    assert result.audio is None
    assert tts_runner.calls == []


def test_generate_lecture_audio_runs_the_full_pipeline_and_writes_audio(tmp_path):
    runner = _FakeRunner(responses=[
        "A clean draft with no flagged claims.",
        "A spoken-friendly version of the draft.",
    ])
    tts_runner = _FakeTTSRunner()
    output_path = str(tmp_path / "out.wav")

    result = generate_lecture_audio(
        "Some topic", [], _FakeTokenizer(), runner, tts_runner, output_path,
    )

    assert isinstance(result, LectureAudioResult)
    assert result.stage_reached == PipelineStage.DONE
    assert result.verification_results == []
    assert result.annotated_draft == "A clean draft with no flagged claims."
    assert result.narration.script == "A spoken-friendly version of the draft."
    assert result.audio is not None
    assert result.audio.output_path == output_path
    assert len(runner.calls) == 2
    assert tts_runner.calls == ["A spoken-friendly version of the draft."]


def test_generate_lecture_audio_verifies_claims_before_narrating(tmp_path):
    draft_with_claim = "Intro. [UNVERIFIED: an obscure fact] Outro."
    runner = _FakeRunner(responses=[
        draft_with_claim,
        "VERDICT: UNCERTAIN\nNOTE: can't confirm.",
        "Narrated version, review note excluded.",
    ])
    tts_runner = _FakeTTSRunner()

    result = generate_lecture_audio(
        "Some topic", [], _FakeTokenizer(), runner, tts_runner, str(tmp_path / "out.wav"),
    )

    assert result.stage_reached == PipelineStage.DONE
    assert len(result.verification_results) == 1
    assert "NEEDS REVIEW" in result.annotated_draft
    assert "UNVERIFIED" not in result.annotated_draft
    # the narration prompt must never see the still-unresolved marker
    assert "NEEDS REVIEW" not in runner.calls[-1]
    assert result.narration.excluded_claims == ["an obscure fact - can't confirm."]


def test_generate_lecture_audio_passes_max_chunk_chars_through(tmp_path):
    config = PipelineConfig(max_chunk_chars=5)
    runner = _FakeRunner(responses=[
        "Draft.",
        "One. Two. Three.",
    ])
    tts_runner = _FakeTTSRunner()

    result = generate_lecture_audio(
        "Some topic", [], _FakeTokenizer(), runner, tts_runner, str(tmp_path / "out.wav"), config,
    )

    assert result.audio.num_chunks == 3
    assert len(tts_runner.calls) == 3
