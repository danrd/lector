"""Keyword-based retrieval over a local Obsidian vault: given a lecture
topic, find whatever personal notes are actually relevant - and, just as
importantly, say so honestly when nothing is. Personal vaults are sparse
and uneven by nature (write-for-yourself notes, not encyclopedia
coverage), so unlike typical RAG top-K retrieval, this deliberately
returns an EMPTY result when nothing clears the relevance bar, rather than
padding the response with the "best" of a bunch of irrelevant notes.

No embeddings/vector index here on purpose: title/tag/body keyword
overlap is index-free, dependency-free (beyond PyYAML for frontmatter),
and good enough for "does this vault say anything about X" - which is
the actual question being asked. If it turns out too many genuinely
relevant notes get missed on vocabulary mismatch alone, a semantic
reranking pass could layer on top of this later without changing the
public contract (search_vault's signature, empty-is-valid semantics).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_INLINE_TAG_RE = re.compile(r"(?<!\w)#([\w/-]+)")


@dataclass
class Note:
    """One parsed vault note."""
    path: Path
    title: str
    tags: frozenset
    frontmatter: dict
    body: str


@dataclass
class SearchResult:
    note: Note
    score: float


class VaultSearchConfig(BaseModel):
    """How search_vault (and the resolver built on it) scores and filters
    notes. Defaults are deliberately conservative - min_score is a real
    bar, not a formality, since "nothing relevant" has to be a genuine,
    common outcome for a personal vault."""
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=5, ge=1)
    min_score: float = Field(default=2.0, ge=0.0)
    title_weight: float = 3.0
    tag_weight: float = 2.0
    body_term_weight: float = 0.3
    body_term_cap: int = Field(default=10, ge=1)  # caps how much one over-repeated term can inflate a note's score


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def parse_note(path: Path) -> Note:
    """Parse one .md file: YAML frontmatter (if present), tags (frontmatter
    `tags:` list + inline `#tags` in the body), title (frontmatter `title`,
    else the filename), and the body text."""
    text = path.read_text(encoding="utf-8")

    frontmatter: dict = {}
    body = text
    match = _FRONTMATTER_RE.match(text)
    if match:
        frontmatter = yaml.safe_load(match.group(1)) or {}
        if not isinstance(frontmatter, dict):
            frontmatter = {}
        body = text[match.end():]

    tags = set()
    fm_tags = frontmatter.get("tags")
    if isinstance(fm_tags, list):
        tags.update(str(t).lstrip("#").lower() for t in fm_tags)
    elif isinstance(fm_tags, str):
        tags.update(t.strip().lstrip("#").lower() for t in fm_tags.split(","))
    tags.update(t.lower() for t in _INLINE_TAG_RE.findall(body))

    title = str(frontmatter.get("title") or path.stem)

    return Note(path=path, title=title, tags=frozenset(tags), frontmatter=frontmatter, body=body)


def load_vault(vault_path: "str | Path") -> List[Note]:
    """Parse every .md file under vault_path (recursively). Files that
    fail to parse (bad encoding, malformed frontmatter) are skipped, not
    fatal - a single broken note shouldn't take down retrieval for every
    other topic."""
    root = Path(vault_path)
    if not root.exists():
        return []

    notes = []
    for md_path in sorted(root.rglob("*.md")):
        try:
            notes.append(parse_note(md_path))
        except (UnicodeDecodeError, yaml.YAMLError):
            continue
    return notes


def _score(note: Note, query_terms: set, config: VaultSearchConfig) -> float:
    title_terms = set(_tokenize(note.title))
    tag_terms = {t for tag in note.tags for t in _tokenize(tag)}
    body_terms = _tokenize(note.body)

    score = 0.0
    score += config.title_weight * len(query_terms & title_terms)
    score += config.tag_weight * len(query_terms & tag_terms)

    body_counts = Counter(body_terms)
    body_hits = sum(min(body_counts[term], config.body_term_cap) for term in query_terms)
    score += config.body_term_weight * body_hits

    return score


def search_vault(notes: List[Note], query: str, config: Optional[VaultSearchConfig] = None) -> List[SearchResult]:
    """Score every note against `query` and return the ones clearing
    config.min_score, best first, capped at config.top_k. Returns []
    when nothing does - a valid, expected outcome, not a failure."""
    config = config or VaultSearchConfig()
    query_terms = set(_tokenize(query))
    if not query_terms:
        return []

    scored = [SearchResult(note=note, score=_score(note, query_terms, config)) for note in notes]
    scored = [r for r in scored if r.score >= config.min_score]
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[: config.top_k]


def format_results(results: List[SearchResult], count_tokens, token_budget: int) -> str:
    """Render search results as text blocks, stopping before token_budget
    is exceeded (count_tokens: e.g. PromptBuilder.count_tokens). Returns
    "" for an empty result list - the caller decides what that means for
    the rest of the prompt, this function just doesn't fabricate content."""
    if not results:
        return ""

    parts: List[str] = []
    used = 0
    for result in results:
        block = f"## {result.note.title}\n{result.note.body.strip()}\n"
        cost = count_tokens(block)
        if used + cost > token_budget:
            break
        parts.append(block)
        used += cost

    return "\n".join(parts)


def make_knowledge_base_resolver(notes: List[Note], config: Optional[VaultSearchConfig] = None):
    """Build a llm_kit.prompt_builder resolver: `topic` (the resolver's
    `task` argument) is searched against `notes`, formatted to fit the
    remaining token budget. Returns "" - never None - when nothing is
    found or fits, since an empty knowledge-base block is meant to be
    skippable, not a reason to fail the whole prompt (PromptBuilder
    treats a resolver returning None as "abort the build entirely",
    which retrieval finding nothing is not)."""
    config = config or VaultSearchConfig()

    def resolver(topic: str, remaining_tokens: int, context: dict, builder) -> str:
        results = search_vault(notes, topic, config=config)
        return format_results(results, builder.count_tokens, remaining_tokens)

    return resolver
