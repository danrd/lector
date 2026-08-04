"""Tests for lector/verification.py.

No real LLM - a fake runner (queue of canned responses, records what it
was called with) stands in, since what's under test is the two-pass
logic (vault-first, model-only-if-needed) and the marker
parsing/annotation, not generation quality.
"""
from __future__ import annotations

from lector.knowledge_base import load_vault
from lector.verification import (
    ClaimVerdict,
    VerificationResult,
    annotate_draft,
    extract_unverified_claims,
    verify_claim,
    verify_draft,
)


class _FakeTokenizer:
    def tokenize(self, text):
        return text.split()


class _FakeRunner:
    """Returns queued responses in order; records every prompt it was
    called with, so tests can assert whether (and with what) it was
    actually invoked."""
    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self._responses.pop(0)


def _write_note(vault_dir, filename, *, title=None, tags=None, body=""):
    lines = ["---"]
    if title is not None:
        lines.append(f"title: {title}")
    if tags is not None:
        lines.append("tags:")
        lines.extend(f"  - {t}" for t in tags)
    lines.append("---")
    (vault_dir / filename).write_text("\n".join(lines) + "\n" + body, encoding="utf-8")


# -- extract_unverified_claims -----------------------------------------------

def test_extract_unverified_claims_finds_a_single_claim():
    draft = "Some text. [UNVERIFIED: Gödel proved this in 1931]. More text."

    claims = extract_unverified_claims(draft)

    assert len(claims) == 1
    assert claims[0].text == "Gödel proved this in 1931"


def test_extract_unverified_claims_finds_multiple_in_order():
    draft = "[UNVERIFIED: first claim] middle text [UNVERIFIED: second claim]"

    claims = extract_unverified_claims(draft)

    assert [c.text for c in claims] == ["first claim", "second claim"]


def test_extract_unverified_claims_returns_empty_list_for_a_clean_draft():
    assert extract_unverified_claims("Nothing flagged in this text at all.") == []


def test_extract_unverified_claims_span_matches_the_full_marker():
    draft = "before [UNVERIFIED: the claim] after"

    claim = extract_unverified_claims(draft)[0]

    start, end = claim.span
    assert draft[start:end] == "[UNVERIFIED: the claim]"


# -- verify_claim: vault-first, model-only-if-needed --------------------------

def test_verify_claim_grounded_in_vault_never_calls_the_runner(tmp_path):
    _write_note(tmp_path, "n.md", title="Gödel", body="Gödel proved incompleteness in 1931.")
    notes = load_vault(tmp_path)
    claim = extract_unverified_claims("[UNVERIFIED: Gödel proved incompleteness in 1931]")[0]
    runner = _FakeRunner()

    result = verify_claim(claim, notes, _FakeTokenizer(), runner)

    assert result.verdict == ClaimVerdict.GROUNDED_IN_VAULT
    assert "Gödel" in result.note
    assert runner.calls == []  # cheap vault check short-circuited the model call


def test_verify_claim_falls_back_to_model_when_not_in_vault():
    claim = extract_unverified_claims("[UNVERIFIED: some obscure fact]")[0]
    runner = _FakeRunner(responses=["VERDICT: CONFIRMED\nNOTE: this is well-established."])

    result = verify_claim(claim, [], _FakeTokenizer(), runner)

    assert result.verdict == ClaimVerdict.CONFIRMED_BY_MODEL
    assert result.note == "this is well-established."
    assert len(runner.calls) == 1
    assert "some obscure fact" in runner.calls[0]


def test_verify_claim_model_says_uncertain():
    claim = extract_unverified_claims("[UNVERIFIED: a shaky claim]")[0]
    runner = _FakeRunner(responses=["VERDICT: UNCERTAIN\nNOTE: the exact date is disputed."])

    result = verify_claim(claim, [], _FakeTokenizer(), runner)

    assert result.verdict == ClaimVerdict.UNCERTAIN
    assert "disputed" in result.note


def test_verify_claim_unparseable_model_response_fails_safe_to_uncertain():
    """An unparseable response must never silently count as verified."""
    claim = extract_unverified_claims("[UNVERIFIED: something]")[0]
    runner = _FakeRunner(responses=["I'm not going to answer in that format."])

    result = verify_claim(claim, [], _FakeTokenizer(), runner)

    assert result.verdict == ClaimVerdict.UNCERTAIN


# -- verify_draft ---------------------------------------------------------

def test_verify_draft_returns_empty_list_for_a_clean_draft():
    assert verify_draft("nothing flagged here", [], _FakeTokenizer(), _FakeRunner()) == []


def test_verify_draft_checks_every_claim_in_order(tmp_path):
    # Title matches the claim text verbatim so the vault hit is
    # unambiguous regardless of exact scoring-weight tuning.
    _write_note(tmp_path, "n.md", title="X is definitely true", body="Elaboration here.")
    notes = load_vault(tmp_path)
    draft = "Intro. [UNVERIFIED: X is definitely true] then [UNVERIFIED: an unrelated guess]."
    runner = _FakeRunner(responses=["VERDICT: UNCERTAIN\nNOTE: can't confirm."])

    results = verify_draft(draft, notes, _FakeTokenizer(), runner)

    assert len(results) == 2
    assert results[0].verdict == ClaimVerdict.GROUNDED_IN_VAULT
    assert results[1].verdict == ClaimVerdict.UNCERTAIN


# -- annotate_draft ---------------------------------------------------------

def test_annotate_draft_strips_the_marker_for_resolved_claims():
    draft = "before [UNVERIFIED: the claim] after"
    claim = extract_unverified_claims(draft)[0]
    results = [VerificationResult(claim=claim, verdict=ClaimVerdict.CONFIRMED_BY_MODEL, note="fine")]

    annotated = annotate_draft(draft, results)

    assert annotated == "before the claim after"


def test_annotate_draft_keeps_a_visible_flag_for_uncertain_claims():
    draft = "before [UNVERIFIED: the claim] after"
    claim = extract_unverified_claims(draft)[0]
    results = [VerificationResult(claim=claim, verdict=ClaimVerdict.UNCERTAIN, note="disputed date")]

    annotated = annotate_draft(draft, results)

    assert "NEEDS REVIEW" in annotated
    assert "the claim" in annotated
    assert "disputed date" in annotated
    assert "UNVERIFIED" not in annotated


def test_annotate_draft_handles_multiple_claims_without_corrupting_surrounding_text():
    draft = "AAA [UNVERIFIED: first] BBB [UNVERIFIED: second] CCC"
    claims = extract_unverified_claims(draft)
    results = [
        VerificationResult(claim=claims[0], verdict=ClaimVerdict.CONFIRMED_BY_MODEL, note=""),
        VerificationResult(claim=claims[1], verdict=ClaimVerdict.UNCERTAIN, note="unsure"),
    ]

    annotated = annotate_draft(draft, results)

    assert annotated.startswith("AAA first BBB")
    assert annotated.endswith("CCC")
    assert "NEEDS REVIEW: second" in annotated
