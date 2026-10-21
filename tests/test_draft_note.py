"""Tests for lector/draft_note.py - the draft's frontmatter, that review
state stays visible (counts, NEEDS REVIEW markers, per-claim outcomes,
source links), and that an existing draft is never overwritten."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml

from lector.draft_note import draft_note_path, render_draft_note, write_draft_note
from lector.knowledge_base import Note
from lector.verification import ClaimVerdict, FlaggedClaim, VerificationResult

TODAY = dt.date(2026, 10, 21)


def _result(text, verdict, note=""):
    return VerificationResult(claim=FlaggedClaim(text=text, span=(0, 0)), verdict=verdict, note=note)


def _note(path):
    return Note(path=Path(path), title=Path(path).stem, tags=frozenset(), frontmatter={}, body="")


def test_render_has_frontmatter_callout_draft_claims_and_sources():
    text = render_draft_note(
        'Теорема "Байеса"', "## Идея\nТекст [NEEDS REVIEW: дата - неясно].",
        [_result("дата", ClaimVerdict.UNCERTAIN, "неясно"), _result("формула", ClaimVerdict.GROUNDED_IN_VAULT)],
        sources=[_note("/vault/Math/02_Probability.md")], vault=Path("/vault"), model="m", today=TODAY,
    )

    frontmatter = yaml.safe_load(text.split("---")[1])
    assert frontmatter == {"status": "draft", "source": "lector", "topic": 'Теорема "Байеса"',
                           "created": TODAY, "model": "m", "tags": []}
    assert "# Теорема \"Байеса\"" in text
    assert "Проверено утверждений: 2, осталось под вопросом: 1" in text
    assert "[NEEDS REVIEW: дата - неясно]" in text
    assert "- ❓ требует проверки: дата — неясно" in text
    assert "- [[Math/02_Probability|02_Probability]]" in text


def test_render_without_sources_or_claims_says_so():
    text = render_draft_note("Тема", "Текст.", [], today=TODAY)

    assert "без опоры на заметки хранилища" in text
    assert "## Проверка утверждений" not in text
    assert "## Опора на заметки" not in text


def test_existing_draft_is_never_overwritten(tmp_path):
    first = write_draft_note(tmp_path, "Topic: A/B", "One.", [])
    first.write_text("edited by hand", encoding="utf-8")

    second = write_draft_note(tmp_path, "Topic: A/B", "Two.", [])

    assert first.name == "Topic A B.md"
    assert second != first
    assert first.read_text(encoding="utf-8") == "edited by hand"
    assert second.name == f"Topic A B ({dt.date.today().isoformat()}).md"
    assert draft_note_path(tmp_path, "Topic: A/B", dt.date.today()).name.endswith(" 2).md")


def test_the_models_own_h1_is_dropped():
    text = render_draft_note("Тема", "# Тема лекции\n\n## Раздел\nТекст.", [], today=TODAY)

    assert text.count("\n# ") == 1
    assert "## Раздел" in text
