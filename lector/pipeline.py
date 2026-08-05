"""Wire the full topic -> narrated audio pipeline together: draft a
lecture grounded in the vault (lecture.py), verify its
[UNVERIFIED: ...] claims (verification.py), rewrite the result for
speech (narration.py), then synthesize it to a single WAV file
(audio.py / tts_runtime.py).

Every stage up to narration can already fail "softly" - a prompt that
doesn't fit its token_limit returns None/empty rather than raising.
This module propagates that instead of masking it: a caller gets back
exactly which stage the run stopped at (LectureAudioResult.stage_reached),
not a confusing empty audio file with no explanation.

config.lecture.language and config.narration.language are independent -
if you want the lecture drafted and narrated in the same language, set
both; they are not synced automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from lector.audio import AudioResult, synthesize_long_text
from lector.knowledge_base import Note
from lector.lecture import LectureConfig, draft_lecture
from lector.narration import NarrationConfig, NarrationResult, rewrite_for_narration
from lector.tts_runtime import BaseTTSRunner
from lector.verification import VerificationConfig, VerificationResult, annotate_draft, verify_draft


class PipelineStage(str, Enum):
    DRAFTING = "drafting"
    NARRATION = "narration"
    DONE = "done"


class PipelineConfig(BaseModel):
    """Bundles each stage's own config under one object, so a caller
    tunes one thing instead of threading four configs through by hand."""
    model_config = ConfigDict(extra="forbid")

    lecture: LectureConfig = Field(default_factory=LectureConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    narration: NarrationConfig = Field(default_factory=NarrationConfig)
    max_chunk_chars: int = Field(default=1000, ge=1)


@dataclass
class LectureAudioResult:
    """What actually happened, stage by stage - stopping early (e.g. at
    DRAFTING because the topic + vault content didn't fit the token
    budget) is a legitimate outcome, not swallowed into a generic
    failure. Fields past `stage_reached` are populated only as far as
    the run actually got."""
    stage_reached: PipelineStage
    topic: str
    draft: Optional[str] = None
    verification_results: List[VerificationResult] = field(default_factory=list)
    annotated_draft: Optional[str] = None
    narration: Optional[NarrationResult] = None
    audio: Optional[AudioResult] = None


def generate_lecture_audio(topic: str, notes: List[Note], tokenizer, runner,
                            tts_runner: BaseTTSRunner, output_path: str,
                            config: Optional[PipelineConfig] = None) -> LectureAudioResult:
    """Run the full pipeline for `topic`, writing the final narration to
    a WAV file at `output_path`. `runner` handles every LLM call
    (drafting, verification, narration-rewrite); `tts_runner` handles
    speech synthesis - pass whichever lector.tts_runtime backend you
    want (PiperRunner by default, OpenAITTSRunner as the cloud option).
    """
    config = config or PipelineConfig()

    draft = draft_lecture(topic, notes, tokenizer, runner, config.lecture)
    if draft is None:
        return LectureAudioResult(stage_reached=PipelineStage.DRAFTING, topic=topic)

    verification_results = verify_draft(draft, notes, tokenizer, runner, config.verification)
    annotated_draft = annotate_draft(draft, verification_results)

    narration_result = rewrite_for_narration(annotated_draft, tokenizer, runner, config.narration)
    if not narration_result.script:
        return LectureAudioResult(
            stage_reached=PipelineStage.NARRATION, topic=topic, draft=draft,
            verification_results=verification_results, annotated_draft=annotated_draft,
            narration=narration_result,
        )

    audio_result = synthesize_long_text(
        narration_result.script, tts_runner, output_path, max_chunk_chars=config.max_chunk_chars,
    )

    return LectureAudioResult(
        stage_reached=PipelineStage.DONE, topic=topic, draft=draft,
        verification_results=verification_results, annotated_draft=annotated_draft,
        narration=narration_result, audio=audio_result,
    )
