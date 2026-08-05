"""Tests for lector/narration.py.

No real LLM or TTS - a fake runner (queue of canned responses, records
what it was called with) stands in, since what's under test is the
[NEEDS REVIEW: ...]-stripping-before-prompting order of operations and
the leftover-markdown sanity check, not narration quality.
"""
from __future__ import annotations

from lector.narration import (
    NarrationConfig,
    _strip_needs_review,
    build_narration_prompt,
    rewrite_for_narration,
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


# -- _strip_needs_review ------------------------------------------------

def test_strip_needs_review_removes_a_single_marker():
    draft = "before [NEEDS REVIEW: shaky claim - no source found] after"

    cleaned, excluded = _strip_needs_review(draft)

    assert cleaned == "before  after"
    assert excluded == ["shaky claim - no source found"]


def test_strip_needs_review_removes_multiple_markers_in_order():
    draft = "[NEEDS REVIEW: first] middle [NEEDS REVIEW: second]"

    cleaned, excluded = _strip_needs_review(draft)

    assert "NEEDS REVIEW" not in cleaned
    assert excluded == ["first", "second"]


def test_strip_needs_review_is_a_no_op_for_a_clean_draft():
    draft = "Nothing flagged in this text at all."

    cleaned, excluded = _strip_needs_review(draft)

    assert cleaned == draft
    assert excluded == []


# -- build_narration_prompt ----------------------------------------------

def test_build_narration_prompt_includes_the_draft_content():
    prompt = build_narration_prompt("The Pythagorean theorem states x^2 + y^2 = z^2.", _FakeTokenizer())

    assert prompt is not None
    assert "Pythagorean theorem" in prompt


def test_build_narration_prompt_includes_language_instruction_when_set():
    config = NarrationConfig(language="French")

    prompt = build_narration_prompt("some text", _FakeTokenizer(), config)

    assert "French" in prompt


def test_build_narration_prompt_omits_language_instruction_when_unset():
    prompt = build_narration_prompt("some text", _FakeTokenizer())

    assert "Write the narration in" not in prompt


def test_build_narration_prompt_returns_none_when_it_does_not_fit():
    config = NarrationConfig(token_limit=1)

    prompt = build_narration_prompt("way more text than the tiny budget allows", _FakeTokenizer(), config)

    assert prompt is None


# -- rewrite_for_narration -------------------------------------------------

def test_rewrite_for_narration_happy_path_returns_the_generated_script():
    runner = _FakeRunner(responses=["Spoken script with no markup at all."])

    result = rewrite_for_narration("## Heading\n\nSome text.", _FakeTokenizer(), runner)

    assert result.script == "Spoken script with no markup at all."
    assert result.excluded_claims == []
    assert result.has_leftover_markdown is False
    assert len(runner.calls) == 1


def test_rewrite_for_narration_strips_needs_review_before_building_the_prompt():
    draft = "Solid intro. [NEEDS REVIEW: disputed fact - can't confirm] Solid outro."
    runner = _FakeRunner(responses=["Solid intro. Solid outro, spoken."])

    result = rewrite_for_narration(draft, _FakeTokenizer(), runner)

    assert len(runner.calls) == 1
    assert "NEEDS REVIEW" not in runner.calls[0]
    assert "disputed fact" not in runner.calls[0]
    assert result.excluded_claims == ["disputed fact - can't confirm"]


def test_rewrite_for_narration_detects_leftover_markdown_in_the_response():
    runner = _FakeRunner(responses=["## Still a heading\n\n- and a bullet"])

    result = rewrite_for_narration("some draft", _FakeTokenizer(), runner)

    assert result.has_leftover_markdown is True


def test_rewrite_for_narration_clean_prose_has_no_leftover_markdown():
    runner = _FakeRunner(responses=["Just flowing spoken prose, nothing structural."])

    result = rewrite_for_narration("some draft", _FakeTokenizer(), runner)

    assert result.has_leftover_markdown is False


def test_rewrite_for_narration_returns_empty_script_without_calling_the_runner_when_prompt_does_not_fit():
    config = NarrationConfig(token_limit=1)
    runner = _FakeRunner()

    result = rewrite_for_narration("way more text than the tiny budget allows", _FakeTokenizer(), runner, config)

    assert result.script == ""
    assert runner.calls == []
