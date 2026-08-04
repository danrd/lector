"""Tests for lector/lecture.py: prompt construction and draft generation.
No real LLM involved - a fake tokenizer/runner (same pattern as llm_kit's
own tests) stand in, since what's under test here is the wiring (does the
prompt actually include vault content when relevant, does it still build
when there's none, does draft_lecture call the runner correctly) not
generation quality.
"""
from __future__ import annotations

from lector.knowledge_base import load_vault
from lector.lecture import LectureConfig, build_lecture_prompt, draft_lecture


class _FakeTokenizer:
    def tokenize(self, text):
        return text.split()


class _FakeRunner:
    def __init__(self, response="GENERATED LECTURE"):
        self.response = response
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.response


def _write_note(vault_dir, filename, *, title=None, tags=None, body=""):
    lines = ["---"]
    if title is not None:
        lines.append(f"title: {title}")
    if tags is not None:
        lines.append("tags:")
        lines.extend(f"  - {t}" for t in tags)
    lines.append("---")
    (vault_dir / filename).write_text("\n".join(lines) + "\n" + body, encoding="utf-8")


def test_build_lecture_prompt_includes_relevant_vault_content(tmp_path):
    _write_note(tmp_path, "prolog.md", title="Логическое программирование",
                tags=["prolog"], body="Prolog использует unification и resolution.")
    notes = load_vault(tmp_path)

    prompt = build_lecture_prompt("Логическое программирование", notes, _FakeTokenizer())

    assert prompt is not None
    assert "unification" in prompt


def test_build_lecture_prompt_succeeds_with_no_relevant_notes(tmp_path):
    """The core case this whole design is meant to handle gracefully:
    nothing in the vault on this topic - the prompt still builds, just
    without a knowledge_base section, and the model is still told how to
    mark ungrounded claims."""
    _write_note(tmp_path, "unrelated.md", title="Something Else", body="irrelevant content")
    notes = load_vault(tmp_path)

    prompt = build_lecture_prompt("quantum chromodynamics", notes, _FakeTokenizer())

    assert prompt is not None
    assert "quantum chromodynamics" in prompt
    assert "UNVERIFIED" in prompt


def test_build_lecture_prompt_works_with_an_empty_vault():
    prompt = build_lecture_prompt("any topic", [], _FakeTokenizer())

    assert prompt is not None
    assert "any topic" in prompt


def test_build_lecture_prompt_includes_language_instruction_when_set():
    config = LectureConfig(language="Russian")

    prompt = build_lecture_prompt("some topic", [], _FakeTokenizer(), config=config)

    assert "Russian" in prompt


def test_build_lecture_prompt_omits_language_instruction_when_unset():
    prompt = build_lecture_prompt("some topic", [], _FakeTokenizer())

    assert "Write the lecture in" not in prompt


def test_draft_lecture_generates_via_the_runner():
    runner = _FakeRunner(response="the actual lecture text")

    result = draft_lecture("some topic", [], _FakeTokenizer(), runner)

    assert result == "the actual lecture text"
    assert len(runner.calls) == 1
    assert "some topic" in runner.calls[0]


def test_draft_lecture_returns_none_without_calling_runner_when_prompt_does_not_fit():
    runner = _FakeRunner()
    config = LectureConfig(token_limit=1)  # far too small for the role_instruction block alone

    result = draft_lecture("some topic", [], _FakeTokenizer(), runner, config=config)

    assert result is None
    assert runner.calls == []
