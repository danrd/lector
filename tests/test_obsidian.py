"""Tests for lector/obsidian.py.

All pure text processing, so every test is a small inline note - the
cases are the Obsidian constructs this vault actually uses (numbered
file names in links, a "map" navigation callout, "🔗 Sources" link lists
under each subtopic, LaTeX with subscripts that look like markdown
italics).
"""
from __future__ import annotations

import pytest

from lector.obsidian import (
    SectionNotFoundError,
    clean_inline,
    clean_markdown,
    list_headings,
    normalize_heading,
    select_sections,
)

NOTE = """# Probability

## 📚 Theory

### Bayes theorem
Bayes body.

### 🔗 Sources
- [[Some paper]]

### Bayesian inference
Inference body.

## 📰 Research
Research body.
"""


# -- headings ----------------------------------------------------------------

def test_normalize_heading_ignores_emoji_case_and_punctuation():
    assert normalize_heading("📚 Theory") == "theory"
    assert normalize_heading("Условная вероятность, независимость") == "условная вероятность независимость"


def test_list_headings_skips_hash_lines_inside_code_fences():
    body = "# Real\n```python\n# not a heading\n```\n## Also real\n"

    assert [(h.level, h.text) for h in list_headings(body)] == [(1, "Real"), (2, "Also real")]


# -- select_sections --------------------------------------------------------

def test_select_sections_takes_a_section_with_its_subsections_minus_sources():
    selected = select_sections(NOTE, ["Theory"])

    assert "Bayes body." in selected
    assert "Inference body." in selected
    assert "Some paper" not in selected      # the "🔗 Sources" subsection is cut out
    assert "Research body." not in selected  # the next level-2 section ends it


def test_select_sections_exact_match_beats_substring_match():
    selected = select_sections(NOTE, ["Bayes theorem"])

    assert "Bayes body." in selected
    assert "Inference body." not in selected


def test_select_sections_substring_matches_every_containing_heading():
    selected = select_sections(NOTE, ["bayes"])

    assert "Bayes body." in selected
    assert "Inference body." in selected


def test_select_sections_does_not_repeat_a_subsection_selected_twice():
    selected = select_sections(NOTE, ["Theory", "Bayes theorem"])

    assert selected.count("Bayes body.") == 1


def test_select_sections_keeps_note_order_whatever_the_request_order():
    selected = select_sections(NOTE, ["Research", "Bayes theorem"])

    assert selected.index("Bayes body.") < selected.index("Research body.")


def test_select_sections_without_names_returns_whole_note_minus_sources():
    selected = select_sections(NOTE)

    assert "Research body." in selected
    assert "Some paper" not in selected


def test_select_sections_can_keep_sources():
    assert "Some paper" in select_sections(NOTE, exclude=())


def test_select_sections_unknown_name_lists_available_headings():
    with pytest.raises(SectionNotFoundError) as excinfo:
        select_sections(NOTE, ["Galois"])

    assert "### Bayes theorem" in str(excinfo.value)


# -- clean_inline -----------------------------------------------------------

def test_wikilink_reads_alias_heading_or_bare_file_name():
    assert clean_inline("see [[06_NLP]]") == "see NLP"
    assert clean_inline("see [[10_Multimodal#Audio and Speech|10_Multimodal]]") == "see Multimodal"
    assert clean_inline("via [[../Math/03_Calculus#Ряды Фурье|Math/03_Calculus]]") == "via Calculus"
    assert clean_inline("see [[File#Some heading]]") == "see Some heading"
    assert clean_inline("as [[Note|in my notes]]") == "as in my notes"


def test_embeds_urls_and_footnotes_are_dropped():
    assert clean_inline("A ![[diagram.png]] B ![alt](x.png) C https://x.org/y D[^1]") == "A B C D"


def test_markdown_link_keeps_its_text():
    assert clean_inline("read [the paper](https://arxiv.org/abs/1) now") == "read the paper now"


def test_emphasis_is_stripped_but_formulas_and_code_are_untouched():
    line = "**Bold** and *it* with $A_1, A_2$ and `snake_case_name` and _under_"

    assert clean_inline(line) == "Bold and it with $A_1, A_2$ and `snake_case_name` and under"


def test_inline_tags_and_emoji_are_removed():
    assert clean_inline("Some text #method/ssl 📚 here") == "Some text here"


# -- clean_markdown ---------------------------------------------------------

def test_headings_become_sentences_and_list_markers_go():
    cleaned = clean_markdown("## 📚 Теория\n- первый пункт\n1. второй пункт;\n")

    assert cleaned.lines == ["Теория.", "первый пункт.", "второй пункт;"]


def test_map_callout_is_skipped_but_other_callouts_are_read():
    markdown = (
        "> [!map]- Place in the architecture\n> **Core:** [[A]] · [[B]]\n\n"
        "> [!tip] Remember\n> The body counts.\n"
    )

    assert clean_markdown(markdown).lines == ["Remember.", "The body counts."]


def test_code_blocks_and_tables_are_skipped_and_counted():
    markdown = "Before.\n```python\nx = 1\n```\n| a | b |\n|---|---|\n| 1 | 2 |\nAfter.\n"

    cleaned = clean_markdown(markdown)

    assert cleaned.lines == ["Before.", "After."]
    assert cleaned.skipped_code_blocks == 1
    assert cleaned.skipped_tables == 1


def test_multiline_display_math_becomes_one_line():
    markdown = "$$\nP(A) = \\sum_i P(A|B_i)\n$$\n"

    assert clean_markdown(markdown).lines == ["$$ P(A) = \\sum_i P(A|B_i) $$."]


def test_rules_tag_lines_and_comments_are_dropped():
    markdown = "Text.\n---\n#method/self-supervised-learning\n%% private %%\n<!-- hidden -->\n"

    assert clean_markdown(markdown).lines == ["Text."]


def test_symbol_only_lines_are_dropped():
    assert clean_markdown("📚\n-\nReal.\n").lines == ["Real."]
