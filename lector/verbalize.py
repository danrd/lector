"""Make formulas and notation speakable while leaving the rest of a note's
text exactly as written.

The rest of the reading path is verbatim on purpose (see
lector/obsidian.py), so this is the one place a model touches the text,
and it touches as little as possible:

- Only lines that actually contain notation ($...$, `...`, Greek
  letters, arrows, operators) are sent to the model at all; every other
  line passes through byte-for-byte.
- The model gets whole lines, not isolated formulas: in Russian a
  formula's spoken form has to agree in case with the sentence around it
  ("для набора $A_1$" -> "для набора А один", not a bare nominative), and
  that's impossible to get right without the sentence.
- Its output is checked: every ordinary word of the original line must
  still be there. A line where the model paraphrased instead of only
  verbalizing is kept (it's usually still right) but reported as
  drifted, so the caller can point the user at it rather than have it
  slip through unnoticed.

Verbalization goes through the model, not a rule-based LaTeX-to-speech
converter: rules can't do grammatical agreement, and a model given the
whole sentence can. A documented, accepted trade-off, same as
narration.py's.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from lector.obsidian import FORMULA_SPAN_RE
from llm_kit.prompt_builder import PromptBuilder, PromptingConfig

PROMPTS_DIR = str(Path(__file__).parent / "prompts")

# Greek, arrows, math operators/symbols, super/subscript digits, and the
# few ASCII/Latin-1 characters that only show up in prose as notation.
_NOTATION_RE = re.compile(r"[$`Ͱ-Ͽ←-⇿∀-⋿⁰-₟⟰-⟿²³¹×÷±^=<>]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_LEFTOVER_RE = re.compile(r"[$`]|\\[a-zA-Z]+")
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


class VerbalizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocks_dir: str = PROMPTS_DIR
    token_limit: int = Field(default=8000, ge=1)
    batch_size: int = Field(default=8, ge=1)
    max_workers: int = Field(default=4, ge=1)
    # Share of a line's ordinary words that must survive the rewrite for it
    # to count as verbatim. Not 1.0: a word glued to a formula ("$x$-ось")
    # legitimately changes form along with it.
    min_prose_overlap: float = Field(default=0.9, ge=0.0, le=1.0)


@dataclass
class VerbalizationResult:
    lines: List[str]
    rewritten: List[Tuple[str, str]] = field(default_factory=list)  # (original, spoken) per changed line
    drifted: List[Tuple[str, str]] = field(default_factory=list)    # rewrites that lost ordinary words
    failed: List[str] = field(default_factory=list)                 # lines the model never returned usably

    @property
    def has_leftover_notation(self) -> bool:
        return any(_LEFTOVER_RE.search(line) for line in self.lines)


def needs_verbalization(line: str) -> bool:
    return bool(_NOTATION_RE.search(line))


def strip_notation_markup(line: str) -> str:
    """The no-model fallback: drop the $ and ` delimiters and keep their
    content as-is. Far from good speech for real formulas, but it never
    invents anything - used for --formulas raw and for batches the model
    failed on."""
    return re.sub(r"\s+", " ", line.replace("$", "").replace("`", "")).strip()


def _prose_words(line: str) -> List[str]:
    """The ordinary words of a line: everything outside formula/code spans
    that isn't itself notation (a lone Greek letter, "x²")."""
    prose = FORMULA_SPAN_RE.sub(" ", line)
    return [w.casefold() for w in _WORD_RE.findall(prose)
            if len(w) > 1 and not _NOTATION_RE.search(w) and not w.isdigit()]


def preserves_prose(original: str, rewritten: str, min_overlap: float = 0.9) -> bool:
    words = _prose_words(original)
    if not words:
        return True
    kept = {w.casefold() for w in _WORD_RE.findall(rewritten)}
    return sum(w in kept for w in words) / len(words) >= min_overlap


def build_verbalize_prompt(lines: List[str], tokenizer,
                            config: Optional[VerbalizeConfig] = None) -> Optional[str]:
    config = config or VerbalizeConfig()
    prompting_config = PromptingConfig(
        blocks_dir=config.blocks_dir, blocks=["verbalize_instruction", "verbalize_lines"],
        token_limit=config.token_limit,
    )
    builder = PromptBuilder(prompting_config, tokenizer)
    lines_json = json.dumps(lines, ensure_ascii=False, indent=0)
    return builder.build(task=lines_json, context={"lines_json": lines_json, "count": len(lines)})


def _parse_lines(output: str, expected: int) -> Optional[List[str]]:
    try:
        parsed = json.loads(_JSON_FENCE_RE.sub("", output or ""))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or len(parsed) != expected or not all(isinstance(s, str) for s in parsed):
        return None
    return parsed


def _verbalize_batch(batch: List[str], tokenizer, runner, config: VerbalizeConfig) -> Optional[List[str]]:
    """One model call for `batch`, retried once on an unusable answer (bad
    JSON, wrong line count). None if both attempts fail or the prompt
    doesn't fit token_limit."""
    prompt = build_verbalize_prompt(batch, tokenizer, config)
    if prompt is None:
        return None
    for _ in range(2):
        parsed = _parse_lines(runner.generate(prompt), len(batch))
        if parsed is not None:
            return parsed
    return None


def verbalize_lines(lines: List[str], tokenizer, runner,
                     config: Optional[VerbalizeConfig] = None) -> VerbalizationResult:
    """Return `lines` with every formula/notation span turned into words,
    via `runner` (any llm_kit Runner). Lines with no notation are never
    sent to the model. Batches run concurrently, up to
    config.max_workers at a time - the runner must tolerate that (the
    hosted OpenAI-compatible ones do)."""
    config = config or VerbalizeConfig()
    indices = [i for i, line in enumerate(lines) if needs_verbalization(line)]
    batches = [indices[i:i + config.batch_size] for i in range(0, len(indices), config.batch_size)]

    def run(batch_indices: List[int]) -> Optional[List[str]]:
        return _verbalize_batch([lines[i] for i in batch_indices], tokenizer, runner, config)

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        outputs = list(pool.map(run, batches))

    result = VerbalizationResult(lines=list(lines))
    for batch_indices, output in zip(batches, outputs):
        for position, index in enumerate(batch_indices):
            original = lines[index]
            if output is None:
                result.failed.append(original)
                result.lines[index] = strip_notation_markup(original)
                continue
            spoken = strip_notation_markup(output[position])
            result.lines[index] = spoken
            if spoken != original:
                result.rewritten.append((original, spoken))
            if not preserves_prose(original, spoken, config.min_prose_overlap):
                result.drifted.append((original, spoken))
    return result
