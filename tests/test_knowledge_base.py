"""Tests for lector/knowledge_base.py: keyword retrieval over a synthetic
Obsidian-style vault (frontmatter + tags + body, built by hand in tmp_path
so each test controls exactly what's "in the vault"). The behavior most
worth pinning down: an irrelevant query must return an EMPTY result, not
just a low-ranked one - "no data for this topic" is the common case for a
personal vault, not an edge case.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lector.knowledge_base import (
    VaultSearchConfig,
    format_results,
    load_vault,
    make_knowledge_base_resolver,
    parse_note,
    search_vault,
)


def _write_note(vault_dir: Path, filename: str, *, title=None, tags=None, body="") -> Path:
    frontmatter_lines = ["---"]
    if title is not None:
        frontmatter_lines.append(f"title: {title}")
    if tags is not None:
        frontmatter_lines.append("tags:")
        frontmatter_lines.extend(f"  - {t}" for t in tags)
    frontmatter_lines.append("---")
    content = "\n".join(frontmatter_lines) + "\n" + body
    path = vault_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def vault(tmp_path):
    _write_note(
        tmp_path, "prolog.md",
        title="Логическое программирование",
        tags=["method/symbolic-reasoning", "prolog"],
        body="Prolog использует unification и resolution для вывода.",
    )
    _write_note(
        tmp_path, "gradient-descent.md",
        title="Gradient Descent",
        tags=["method/optimization"],
        body="Basic first-order optimization: follow the negative gradient.",
    )
    _write_note(
        tmp_path, "no-frontmatter.md",
        body="Just a plain note about #chess openings, no frontmatter block.",
    )
    return load_vault(tmp_path)


# -- parse_note ---------------------------------------------------------

def test_parse_note_reads_frontmatter_title_and_tags(tmp_path):
    path = _write_note(tmp_path, "n.md", title="My Title", tags=["a", "b/c"], body="hello")

    note = parse_note(path)

    assert note.title == "My Title"
    assert note.tags == frozenset({"a", "b/c"})
    assert "hello" in note.body


def test_parse_note_without_frontmatter_falls_back_to_filename(tmp_path):
    path = tmp_path / "untitled-note.md"
    path.write_text("no frontmatter here", encoding="utf-8")

    note = parse_note(path)

    assert note.title == "untitled-note"
    assert note.frontmatter == {}


def test_parse_note_picks_up_inline_body_tags(tmp_path):
    path = _write_note(tmp_path, "n.md", body="talking about #chess and #openings/sicilian today")

    note = parse_note(path)

    assert "chess" in note.tags
    assert "openings/sicilian" in note.tags


# -- load_vault -----------------------------------------------------------

def test_load_vault_returns_empty_list_for_missing_directory(tmp_path):
    assert load_vault(tmp_path / "does-not-exist") == []


def test_load_vault_skips_unparseable_files_instead_of_crashing(tmp_path):
    _write_note(tmp_path, "good.md", title="Good", body="fine")
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"---\ntags: [\xff\xfe invalid yaml\n---\nbroken")

    notes = load_vault(tmp_path)

    assert any(n.title == "Good" for n in notes)


# -- search_vault: the core "often there's no data" behavior --------------

def test_search_vault_finds_the_relevant_note_by_title(vault):
    results = search_vault(vault, "Логическое программирование")

    assert results
    assert results[0].note.title == "Логическое программирование"


def test_search_vault_finds_by_tag_even_without_title_overlap(vault):
    results = search_vault(vault, "symbolic-reasoning")

    assert any(r.note.title == "Логическое программирование" for r in results)


def test_search_vault_returns_empty_for_an_unrelated_topic(vault):
    """The behavior this module exists to get right: a topic the vault
    has nothing on must come back empty, not as a padded top-K of
    barely-related notes."""
    results = search_vault(vault, "quantum chromodynamics")

    assert results == []


def test_search_vault_returns_empty_for_empty_vault():
    assert search_vault([], "anything") == []


def test_search_vault_respects_top_k(vault):
    """Query deliberately matches both notes' tags (method/symbolic-reasoning,
    method/optimization) - verified uncapped this scores 2 qualifying hits,
    so top_k=1 actually has something to cut down, not just an empty result
    that happens to satisfy "<= 1"."""
    query = "method optimization symbolic reasoning"
    uncapped = search_vault(vault, query, config=VaultSearchConfig(top_k=100, min_score=0.1))
    assert len(uncapped) == 2  # sanity check the test's own premise

    capped = search_vault(vault, query, config=VaultSearchConfig(top_k=1, min_score=0.1))

    assert len(capped) == 1
    assert capped[0].note.title == uncapped[0].note.title  # keeps the best-scoring one


def test_search_vault_min_score_filters_weak_matches(vault):
    lenient = search_vault(vault, "resolution", config=VaultSearchConfig(min_score=0.1))
    strict = search_vault(vault, "resolution", config=VaultSearchConfig(min_score=1000.0))

    assert lenient  # a single body-text hit clears a near-zero bar
    assert strict == []  # ...but not an unreasonably high one


# -- format_results ---------------------------------------------------------

def test_format_results_empty_list_returns_empty_string():
    assert format_results([], count_tokens=len, token_budget=1000) == ""


def test_format_results_stops_before_exceeding_the_token_budget(vault):
    results = search_vault(vault, "Логическое программирование")
    assert results

    def count_tokens(text):
        return len(text.split())

    full = format_results(results, count_tokens, token_budget=10_000)
    truncated = format_results(results, count_tokens, token_budget=1)

    assert full != ""
    assert truncated == ""  # can't even fit the first block in a 1-token budget


# -- make_knowledge_base_resolver: PromptBuilder integration ----------------

class _FakeBuilder:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_resolver_returns_formatted_text_for_a_relevant_topic(vault):
    resolver = make_knowledge_base_resolver(vault)

    rendered = resolver("Логическое программирование", 10_000, {}, _FakeBuilder())

    assert "Prolog" in rendered or "unification" in rendered


def test_resolver_returns_empty_string_not_none_for_an_irrelevant_topic(vault):
    """Critical distinction: PromptBuilder treats a resolver returning
    None as "abort the whole prompt build" - an empty knowledge base for
    this topic must not do that, since the topic is meant to be usable
    without any vault content at all."""
    resolver = make_knowledge_base_resolver(vault)

    rendered = resolver("quantum chromodynamics", 10_000, {}, _FakeBuilder())

    assert rendered == ""
    assert rendered is not None


def test_resolver_integrates_end_to_end_with_prompt_builder_when_topic_has_no_notes(tmp_path):
    """Full round-trip: a PromptBuilder build() call must succeed (not
    return None) even when the knowledge_base resolver finds nothing."""
    from llm_kit.prompt_builder import PromptBuilder, PromptingConfig

    block_dir = tmp_path / "blocks" / "intro"
    block_dir.mkdir(parents=True)
    (block_dir / "v1.j2").write_text("Lecture on: {{ topic }}")

    config = PromptingConfig(
        blocks_dir=str(tmp_path / "blocks"),
        blocks=["intro", "knowledge_base"],
        token_limit=10_000,
        resolvers=["knowledge_base"],
    )

    class _Tok:
        def tokenize(self, text):
            return text.split()

    builder = PromptBuilder(
        config, _Tok(),
        resolver_registry={"knowledge_base": make_knowledge_base_resolver([])},  # empty vault
    )

    result = builder.build(task="quantum chromodynamics", context={"topic": "quantum chromodynamics"})

    assert result is not None
    assert "Lecture on: quantum chromodynamics" in result
