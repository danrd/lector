"""Draft a lecture on a topic: build a prompt grounded in whatever the
vault retrieval finds for that topic (possibly nothing - see
knowledge_base.py), then generate a draft via whichever llm_kit Runner
the caller configured.

The prompt instructs the model to mark claims it's drawing from its own
knowledge (rather than the provided material) with `[UNVERIFIED: ...]` -
see prompts/role_instruction/v1.j2. That marker is what a later
verification step is meant to key off of, so it only has to scrutinize
what's actually ungrounded instead of re-checking the whole draft.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from lector.knowledge_base import Note, VaultSearchConfig, make_knowledge_base_resolver
from llm_kit.prompt_builder import PromptBuilder, PromptingConfig

PROMPTS_DIR = str(Path(__file__).parent / "prompts")

LECTURE_BLOCKS = ["role_instruction", "knowledge_base", "topic", "output_format"]


class LectureConfig(BaseModel):
    """Config for drafting one lecture."""
    model_config = ConfigDict(extra="forbid")

    blocks_dir: str = PROMPTS_DIR
    token_limit: int = Field(default=8000, ge=1)
    language: Optional[str] = None
    vault_search: VaultSearchConfig = Field(default_factory=VaultSearchConfig)


def build_lecture_prompt(topic: str, notes: List[Note], tokenizer,
                          config: Optional[LectureConfig] = None) -> Optional[str]:
    """Build (but don't generate) the lecture prompt for `topic`, grounded
    in whatever of `notes` is relevant - possibly nothing, in which case
    the knowledge_base block just contributes no content, not an error.
    Returns None if the prompt doesn't fit token_limit (same convention as
    PromptBuilder.build)."""
    config = config or LectureConfig()
    prompting_config = PromptingConfig(
        blocks_dir=config.blocks_dir,
        blocks=LECTURE_BLOCKS,
        token_limit=config.token_limit,
        resolvers=["knowledge_base"],
    )
    builder = PromptBuilder(
        prompting_config, tokenizer,
        resolver_registry={"knowledge_base": make_knowledge_base_resolver(notes, config.vault_search)},
    )
    return builder.build(task=topic, context={"topic": topic, "language": config.language})


def draft_lecture(topic: str, notes: List[Note], tokenizer, runner,
                   config: Optional[LectureConfig] = None) -> Optional[str]:
    """Build the prompt and generate a draft via `runner`
    (runner.generate(prompt) -> str - any llm_kit Runner). Returns None
    without calling the runner if the prompt itself didn't fit
    token_limit."""
    prompt = build_lecture_prompt(topic, notes, tokenizer, config)
    if prompt is None:
        return None
    return runner.generate(prompt)
