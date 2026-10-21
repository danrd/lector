"""Tests for lector/verbalize.py.

The model is a fake runner that answers from a lookup table, so what's
checked is everything around the model call: which lines get sent at
all, how answers map back to lines, what happens when the model returns
something unusable, and whether a paraphrase is caught as drift.
"""
from __future__ import annotations

import json
import threading

from llm_kit.prompt_builder import ApproxTokenizer

from lector.verbalize import (
    VerbalizeConfig,
    build_verbalize_prompt,
    needs_verbalization,
    preserves_prose,
    strip_notation_markup,
    verbalize_lines,
)


class _TableRunner:
    """Answers each prompt by rewriting the lines it contains through
    `table` (unknown lines come back unchanged); records every prompt."""

    def __init__(self, table=None, raw_answers=None):
        self.table = table or {}
        self.raw_answers = list(raw_answers or [])
        self.prompts = []
        self._lock = threading.Lock()

    def generate(self, prompt: str) -> str:
        with self._lock:
            self.prompts.append(prompt)
            if self.raw_answers:
                return self.raw_answers.pop(0)
        start = prompt.index("[")
        lines = json.JSONDecoder().raw_decode(prompt[start:])[0]
        return json.dumps([self.table.get(line, line) for line in lines], ensure_ascii=False)


def _verbalize(lines, runner, **config):
    return verbalize_lines(lines, ApproxTokenizer(), runner, VerbalizeConfig(**config))


def test_needs_verbalization_spots_latex_code_greek_and_arrows():
    assert needs_verbalization("тройка $(\\Omega, F, P)$")
    assert needs_verbalization("call `fit()` first")
    assert needs_verbalization("σ-алгебра событий")
    assert needs_verbalization("pipeline → network")
    assert needs_verbalization("счётных объединений/пересечений")
    assert not needs_verbalization("Обычное предложение, без формул.")


def test_lines_without_notation_never_reach_the_model():
    runner = _TableRunner()

    result = _verbalize(["Plain line.", "Another plain one."], runner)

    assert runner.prompts == []
    assert result.lines == ["Plain line.", "Another plain one."]


def test_rewrites_land_on_their_own_lines_in_order():
    lines = ["Intro.", "нормировка: $P(\\Omega)=1$;", "Middle.", "неотрицательность: $P(A)\\ge 0$;"]
    table = {
        lines[1]: "нормировка: пэ от омеги равно единице;",
        lines[3]: "неотрицательность: пэ от а не меньше нуля;",
    }

    result = _verbalize(lines, _TableRunner(table), batch_size=1)

    assert result.lines == [
        "Intro.", "нормировка: пэ от омеги равно единице;",
        "Middle.", "неотрицательность: пэ от а не меньше нуля;",
    ]
    assert [original for original, _ in result.rewritten] == [lines[1], lines[3]]
    assert result.drifted == []


def test_lines_are_batched():
    lines = [f"line {i} with $x_{i}$." for i in range(5)]
    runner = _TableRunner()

    _verbalize(lines, runner, batch_size=2)

    assert len(runner.prompts) == 3


def test_prompt_states_how_many_lines_to_return():
    prompt = build_verbalize_prompt(["a $x$.", "b $y$."], ApproxTokenizer())

    assert "exactly 2 strings" in prompt


def test_fenced_json_answer_is_accepted():
    line = "Sum $\\sum_i x_i$."
    runner = _TableRunner(raw_answers=['```json\n["Sum over i of x i."]\n```'])

    assert _verbalize([line], runner).lines == ["Sum over i of x i."]


def test_unusable_answer_is_retried_once_then_falls_back_to_stripped_markup():
    line = "Value $x^2$ here."
    runner = _TableRunner(raw_answers=["not json", '["one", "too many"]'])

    result = _verbalize([line], runner)

    assert len(runner.prompts) == 2
    assert result.failed == [line]
    assert result.lines == ["Value x^2 here."]


def test_retry_that_succeeds_is_used():
    line = "Value $x^2$ here."
    runner = _TableRunner(raw_answers=["garbage", '["Value x squared here."]'])

    result = _verbalize([line], runner)

    assert result.failed == []
    assert result.lines == ["Value x squared here."]


def test_paraphrase_is_reported_as_drift_but_kept():
    line = "Формула полной вероятности: $P(A) = \\sum_i P(A|B_i)P(B_i)$ — основа теоремы Байеса."
    paraphrase = "Вероятность события равна сумме произведений."

    result = _verbalize([line], _TableRunner({line: paraphrase}))

    assert result.drifted == [(line, paraphrase)]
    assert result.lines == [paraphrase]


def test_leftover_dollar_or_backtick_in_an_answer_is_stripped():
    line = "Value $x$."

    result = _verbalize([line], _TableRunner({line: "Value $икс$."}))

    assert result.lines == ["Value икс."]
    assert not result.has_leftover_notation


def test_preserves_prose_ignores_the_formula_words_themselves():
    original = "для набора $A_1, A_2$ выполняется равенство"

    assert preserves_prose(original, "для набора а один, а два выполняется равенство")
    assert not preserves_prose(original, "выполняется")


def test_strip_notation_markup_keeps_span_content():
    assert strip_notation_markup("use `numpy` and $x+1$  now") == "use numpy and x+1 now"
