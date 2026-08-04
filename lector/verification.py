"""Check the claims a lecture draft marked as [UNVERIFIED: ...] (see
lecture.py / prompts/role_instruction/v1.j2) - the only ones worth
spending verification effort on, since everything else was either
grounded in the vault already or uncontroversial background knowledge.

No external search yet (deferred - see lecture.py's commit message), so
this is two cheap passes rather than real fact-checking against outside
sources:

    1. A second, claim-targeted vault search - catches claims the
       topic-level retrieval in knowledge_base.py missed. A coarse
       topic-level query doesn't always surface a specific fact that a
       more precise, claim-specific one would.
    2. Failing that, ask the model to assess the claim in isolation, out
       of the context of writing a whole lecture. A plain confidence
       check, not independent verification - meant to catch overconfident
       phrasing the model would itself walk back if asked directly, not
       to replace a real external source.

Anything that doesn't clear either pass comes back UNCERTAIN and stays
visibly flagged (see annotate_draft) rather than being silently accepted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from lector.knowledge_base import Note, VaultSearchConfig, search_vault
from llm_kit.prompt_builder import PromptBuilder, PromptingConfig

PROMPTS_DIR = str(Path(__file__).parent / "prompts")

# Non-greedy: a claim containing its own "]" (a citation, an array
# literal) will truncate early at that bracket. Accepted limitation for a
# marker convention this simple - not worth a real parser over.
_CLAIM_RE = re.compile(r"\[UNVERIFIED:\s*(.*?)\]", re.DOTALL)

_VERDICT_RE = re.compile(r"VERDICT:\s*(CONFIRMED|UNCERTAIN)", re.IGNORECASE)
_NOTE_RE = re.compile(r"NOTE:\s*(.*)", re.DOTALL)


@dataclass
class FlaggedClaim:
    text: str
    span: Tuple[int, int]  # (start, end) of the full "[UNVERIFIED: ...]" marker in the draft


class ClaimVerdict(str, Enum):
    GROUNDED_IN_VAULT = "grounded_in_vault"
    CONFIRMED_BY_MODEL = "confirmed_by_model"
    UNCERTAIN = "uncertain"


@dataclass
class VerificationResult:
    claim: FlaggedClaim
    verdict: ClaimVerdict
    note: str = ""


class VerificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocks_dir: str = PROMPTS_DIR
    token_limit: int = Field(default=2000, ge=1)
    vault_search: VaultSearchConfig = Field(default_factory=VaultSearchConfig)


def extract_unverified_claims(draft: str) -> List[FlaggedClaim]:
    """Find every [UNVERIFIED: ...] marker in `draft`, in order."""
    return [FlaggedClaim(text=m.group(1).strip(), span=m.span()) for m in _CLAIM_RE.finditer(draft)]


def _parse_claim_check_response(response: str) -> Tuple[ClaimVerdict, str]:
    verdict_match = _VERDICT_RE.search(response)
    note_match = _NOTE_RE.search(response)
    # Fail-safe: anything we can't parse as an explicit CONFIRMED stays
    # UNCERTAIN - an unparseable response must never silently pass as verified.
    is_confirmed = bool(verdict_match and verdict_match.group(1).upper() == "CONFIRMED")
    verdict = ClaimVerdict.CONFIRMED_BY_MODEL if is_confirmed else ClaimVerdict.UNCERTAIN
    note = note_match.group(1).strip() if note_match else response.strip()
    return verdict, note


def verify_claim(claim: FlaggedClaim, notes: List[Note], tokenizer, runner,
                  config: Optional[VerificationConfig] = None) -> VerificationResult:
    """Check one flagged claim: a targeted vault search first (cheap, no
    model call), then - only if that finds nothing - a model self-check
    of the claim in isolation."""
    config = config or VerificationConfig()

    vault_hits = search_vault(notes, claim.text, config=config.vault_search)
    if vault_hits:
        matched_titles = ", ".join(r.note.title for r in vault_hits)
        return VerificationResult(claim=claim, verdict=ClaimVerdict.GROUNDED_IN_VAULT,
                                   note=f"Matches vault note(s): {matched_titles}")

    prompting_config = PromptingConfig(
        blocks_dir=config.blocks_dir, blocks=["claim_check"], token_limit=config.token_limit,
    )
    builder = PromptBuilder(prompting_config, tokenizer)
    prompt = builder.build(task=claim.text, context={"claim": claim.text})
    if prompt is None:
        return VerificationResult(claim=claim, verdict=ClaimVerdict.UNCERTAIN,
                                   note="Claim-check prompt didn't fit token_limit")

    response = runner.generate(prompt)
    verdict, note = _parse_claim_check_response(response)
    return VerificationResult(claim=claim, verdict=verdict, note=note)


def verify_draft(draft: str, notes: List[Note], tokenizer, runner,
                  config: Optional[VerificationConfig] = None) -> List[VerificationResult]:
    """Extract every [UNVERIFIED: ...] claim from `draft` and check each
    one. Returns [] if the draft flagged nothing - a clean draft is a
    valid, common outcome, not something to report on."""
    claims = extract_unverified_claims(draft)
    return [verify_claim(c, notes, tokenizer, runner, config) for c in claims]


def annotate_draft(draft: str, results: List[VerificationResult]) -> str:
    """Replace each [UNVERIFIED: ...] marker with the outcome of checking
    it: resolved claims (grounded or confirmed) become plain text again -
    their uncertainty was resolved, no reason to keep flagging them -
    while claims that stayed UNCERTAIN become [NEEDS REVIEW: ...], so a
    human (or a later narration step) still sees exactly what's actually
    in doubt, not everything that merely started out unverified.

    `results` must come from verify_draft(draft, ...) called on this
    exact `draft` - spans are matched by position, not by claim text.
    """
    # Back-to-front so replacing one span doesn't shift the positions of
    # markers still to be processed.
    for result in sorted(results, key=lambda r: r.claim.span[0], reverse=True):
        start, end = result.claim.span
        if result.verdict == ClaimVerdict.UNCERTAIN:
            replacement = f"[NEEDS REVIEW: {result.claim.text} - {result.note}]"
        else:
            replacement = result.claim.text
        draft = draft[:start] + replacement + draft[end:]
    return draft
