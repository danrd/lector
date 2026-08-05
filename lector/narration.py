"""Rewrite a (verified/annotated) lecture draft into a script meant to be
read aloud: verbalize formulas/math notation in words, turn markdown
structure (headers, bullet lists) into connected spoken prose, and drop
anything still flagged [NEEDS REVIEW: ...] by verification.py rather than
reading unresolved uncertainty aloud.

Known limitation, not silently accepted: formula verbalization goes
through the model, not a rule-based LaTeX-to-speech converter (a genuinely
hard, separate problem on its own). That means a formula's spoken
restatement is itself an unverified transformation - the model could
subtly get an exponent or operator wrong while "just" rewording it. The
prompt instructs it to preserve meaning exactly, but nothing here checks
that it did. If formula fidelity turns out to matter more than expected
in practice, add a rule-based verbalizer for recognizable notation and
only fall back to the model for the rest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from llm_kit.prompt_builder import PromptBuilder, PromptingConfig

PROMPTS_DIR = str(Path(__file__).parent / "prompts")

_NEEDS_REVIEW_RE = re.compile(r"\[NEEDS REVIEW:\s*(.*?)\]", re.DOTALL)
# Cheap structural sanity check after rewriting - the model was told to
# strip markdown/list structure and never reproduce a review marker; this
# just confirms it actually did, without a second LLM call.
_LEFTOVER_MARKDOWN_RE = re.compile(
    r"^#{1,6}\s|^[-*]\s|^\d+\.\s|\[UNVERIFIED:|\[NEEDS REVIEW:", re.MULTILINE
)


@dataclass
class NarrationResult:
    script: str
    excluded_claims: List[str] = field(default_factory=list)
    has_leftover_markdown: bool = False


class NarrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocks_dir: str = PROMPTS_DIR
    token_limit: int = Field(default=8000, ge=1)
    language: Optional[str] = None


def _strip_needs_review(draft: str) -> Tuple[str, List[str]]:
    """Remove every [NEEDS REVIEW: ...] span entirely - unresolved
    uncertainty must not be read aloud - and return what was cut so the
    caller can surface it separately (e.g. "these N points still need
    manual review before this script is ready")."""
    excluded = [m.group(1).strip() for m in _NEEDS_REVIEW_RE.finditer(draft)]
    cleaned = _NEEDS_REVIEW_RE.sub("", draft)
    return cleaned, excluded


def build_narration_prompt(draft: str, tokenizer, config: Optional[NarrationConfig] = None) -> Optional[str]:
    """Build (but don't generate) the narration-rewrite prompt for
    `draft` as-is - callers that need [NEEDS REVIEW: ...] spans stripped
    first should use rewrite_for_narration instead, which does that
    before ever building a prompt."""
    config = config or NarrationConfig()
    prompting_config = PromptingConfig(
        blocks_dir=config.blocks_dir, blocks=["narration_instruction", "draft"],
        token_limit=config.token_limit,
    )
    builder = PromptBuilder(prompting_config, tokenizer)
    return builder.build(task=draft, context={"draft": draft, "language": config.language})


def rewrite_for_narration(draft: str, tokenizer, runner,
                           config: Optional[NarrationConfig] = None) -> NarrationResult:
    """Strip unresolved [NEEDS REVIEW: ...] spans first, then rewrite the
    rest for speech via `runner`. The model never even sees a still-under-
    review span - it isn't given the chance to narrate it by mistake."""
    cleaned, excluded = _strip_needs_review(draft)

    prompt = build_narration_prompt(cleaned, tokenizer, config)
    if prompt is None:
        return NarrationResult(script="", excluded_claims=excluded)

    script = runner.generate(prompt)
    has_leftover = bool(_LEFTOVER_MARKDOWN_RE.search(script))
    return NarrationResult(script=script, excluded_claims=excluded, has_leftover_markdown=has_leftover)
