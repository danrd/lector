"""Turn an Obsidian note into plain lines ready to be read aloud verbatim:
pick the requested sections, then strip every piece of Obsidian/markdown
syntax a TTS engine would otherwise read out literally or stumble over.

What happens to each construct:

- Headings become their own short sentence ("Вероятностное пространство.")
  - they carry structure a listener needs, the "#" marks don't.
- [[wikilinks]] become their visible text: the alias if there is one,
  else the heading they point at, else the file name without its numeric
  prefix ("../Math/03_Calculus" -> "Calculus"). ![[embeds]] and images
  are dropped - there's nothing to read.
- Callouts lose their "> [!type]" markup; their body is read as normal
  text, except for the types in `skip_callouts` (by default "map", this
  vault's navigation box, which is a list of links, not content, and
  "lector", the banner of a generated draft note).
- `[NEEDS REVIEW: ...]` claims of a generated draft are dropped: unverified
  statements aren't read out as fact.
- Code blocks and tables are skipped entirely - read aloud they are noise
  - and counted, so the caller can say what was left out.
- Bold/italic/highlight/strikethrough markers, inline #tags, footnote
  refs, HTML tags, comments and emoji are removed.

Formulas ($...$, $$...$$) and inline code spans are deliberately left in
place, untouched: turning them into words is lector.verbalize's job, and
it needs the surrounding sentence intact to do it with the right grammar.
Emphasis stripping is applied only between those spans, so a LaTeX
subscript like $A_1$ never gets mistaken for markdown italics.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_CALLOUT_RE = re.compile(r"^\[!([\w-]+)\][+-]?\s*(.*)$")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?")
_RULE_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_TAG_ONLY_RE = re.compile(r"^(?:\s*#[\w/-]+)+\s*$")
_INLINE_TAG_RE = re.compile(r"(?<![\w&])#[^\s#\d][\w/-]*")
FORMULA_SPAN_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$|`[^`\n]+`")
_EMBED_RE = re.compile(r"!\[\[[^\]]*\]\]|!\[[^\]]*\]\([^)]*\)")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_MDLINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+")
_FOOTNOTE_RE = re.compile(r"\[\^[^\]]+\]")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_COMMENT_RE = re.compile(r"%%.*?%%|<!--.*?-->")
# lector's own draft markers (lector/draft_note.py): an unresolved claim is
# exactly what must not be read aloud as if it were fact.
_NEEDS_REVIEW_RE = re.compile(r"\s*\[NEEDS REVIEW:.*?\]", re.DOTALL)
_EMPHASIS_RES = [
    re.compile(r"\*\*(.+?)\*\*"), re.compile(r"__(.+?)__"),
    re.compile(r"==(.+?)=="), re.compile(r"~~(.+?)~~"),
    re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"),
    re.compile(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])"),
]
_NUMERIC_PREFIX_RE = re.compile(r"^\d+[_\s-]+")
_TERMINAL_PUNCT = ".!?…:;"


class SectionNotFoundError(ValueError):
    def __init__(self, name: str, available: Sequence[str]):
        self.name = name
        self.available = list(available)
        listing = "\n".join(f"  {heading}" for heading in self.available) or "  (the note has no headings)"
        super().__init__(f"No section matching '{name}'. Headings in this note:\n{listing}")


@dataclass
class Heading:
    level: int
    text: str
    line_index: int


@dataclass
class CleanedNote:
    lines: List[str]
    skipped_code_blocks: int = 0
    skipped_tables: int = 0


def _strip_symbols(text: str) -> str:
    """Drop emoji and other pictographic symbols (Unicode category "So"),
    plus the joiners/variation selectors they come wrapped in. Math
    symbols (category "Sm": ≤, →, ∑) are kept - verbalize reads them."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) != "So" and ch not in "️‍"
    )


def normalize_heading(text: str) -> str:
    """Casefolded, emoji-free, punctuation-insensitive form of a heading,
    so "theory" matches "## 📚 Theory" and "условная вероятность" matches
    "### Условная вероятность, независимость"."""
    words = re.findall(r"\w+", _strip_symbols(text).casefold())
    return " ".join(words)


def list_headings(body: str) -> List[Heading]:
    headings = []
    in_fence = False
    for index, line in enumerate(body.splitlines()):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if match:
            headings.append(Heading(level=len(match.group(1)), text=match.group(2), line_index=index))
    return headings


def _match_headings(headings: List[Heading], name: str) -> List[Heading]:
    """Exact (normalized) match wins; otherwise every heading containing
    the name. Taking all substring matches rather than guessing one keeps
    "Bayes" from silently picking whichever of two Bayes sections came
    first."""
    wanted = normalize_heading(name)
    if not wanted:
        return []
    exact = [h for h in headings if normalize_heading(h.text) == wanted]
    return exact or [h for h in headings if wanted in normalize_heading(h.text)]


# Per-section link lists in this vault ("### 🔗 Sources" under nearly every
# subtopic) - read aloud they're a string of titles and URLs, not content.
DEFAULT_EXCLUDED_SECTIONS = ("источники", "sources", "ресурсы", "resources",
                             # the bookkeeping tail of a lector draft note
                             "проверка утверждений", "опора на заметки")


def _section_end(headings: List[Heading], heading: Heading, total_lines: int) -> int:
    return next((h.line_index for h in headings
                 if h.line_index > heading.line_index and h.level <= heading.level), total_lines)


def select_sections(body: str, names: Sequence[str] = (),
                     exclude: Sequence[str] = DEFAULT_EXCLUDED_SECTIONS) -> str:
    """Return the text of every section matching any of `names` (a section
    runs from its heading to the next heading of the same or higher
    level, subsections included), in note order, each line at most once -
    a subsection that's already inside another selected section isn't
    repeated. No names means the whole body. Sections whose heading
    matches one of `exclude` exactly are cut out either way.

    Raises SectionNotFoundError for a name that matches nothing, rather
    than quietly reading less than was asked for."""
    lines = body.splitlines()
    headings = list_headings(body)

    if names:
        keep = [False] * len(lines)
        for name in names:
            matched = _match_headings(headings, name)
            if not matched:
                raise SectionNotFoundError(name, [f"{'#' * h.level} {h.text}" for h in headings])
            for heading in matched:
                for index in range(heading.line_index, _section_end(headings, heading, len(lines))):
                    keep[index] = True
    else:
        keep = [True] * len(lines)

    excluded = {normalize_heading(name) for name in exclude}
    for heading in headings:
        if normalize_heading(heading.text) in excluded:
            for index in range(heading.line_index, _section_end(headings, heading, len(lines))):
                keep[index] = False

    return "\n".join(line for line, kept in zip(lines, keep) if kept)


def _link_label(target: str, alias: Optional[str]) -> str:
    if alias:
        label = alias.strip()
    elif "#" in target:
        label = target.split("#", 1)[1].strip()
    else:
        label = target.strip()
    # A label with no spaces is a file name/path (this vault's aliases are
    # often just "10_Multimodal" or "Math/03_Calculus"), not prose: read
    # only the name itself, without the folder and the ordering prefix.
    if " " not in label:
        label = label.rstrip("/").rsplit("/", 1)[-1]
        label = _NUMERIC_PREFIX_RE.sub("", label).replace("_", " ")
    return label


def _clean_prose(text: str) -> str:
    """Inline cleanup for a stretch of text with no formula/code span in it."""
    text = _EMBED_RE.sub("", text)
    text = _WIKILINK_RE.sub(lambda m: _link_label(m.group(1), m.group(2)), text)
    text = _MDLINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("", text)
    text = _FOOTNOTE_RE.sub("", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _INLINE_TAG_RE.sub("", text)
    for pattern in _EMPHASIS_RES:
        text = pattern.sub(r"\1", text)
    return _strip_symbols(text)


def clean_inline(line: str) -> str:
    """Clean one line's inline markup, leaving $formula$ and `code` spans
    exactly as they were."""
    pieces = []
    position = 0
    for match in FORMULA_SPAN_RE.finditer(line):
        pieces.append(_clean_prose(line[position:match.start()]))
        pieces.append(match.group(0))
        position = match.end()
    pieces.append(_clean_prose(line[position:]))
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def _finish_sentence(text: str) -> str:
    """Make sure a line ends a sentence - headings and list items usually
    don't, and both the chunker (lector.audio splits on .!?) and the TTS
    engine's pausing key off sentence-final punctuation."""
    text = re.sub(r"\s+([,.;:!?])", r"\1", text).strip(" ,—-")
    if text and text[-1] not in _TERMINAL_PUNCT:
        text += "."
    return text


def clean_markdown(markdown: str, skip_callouts: Sequence[str] = ("map", "lector")) -> CleanedNote:
    """Strip `markdown` (a note body, frontmatter already removed) down to
    speakable lines - see the module docstring for what happens to what."""
    markdown = _NEEDS_REVIEW_RE.sub("", _COMMENT_RE.sub("", markdown))
    skip = {name.casefold() for name in skip_callouts}

    lines: List[str] = []
    code_blocks = tables = 0
    in_fence = in_table = skipping_callout = False
    display_math: Optional[List[str]] = None

    for raw in markdown.splitlines():
        stripped = raw.strip()

        if _FENCE_RE.match(raw):
            if not in_fence:
                code_blocks += 1
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # Multi-line $$ ... $$ display math collapses into one line, so
        # verbalize sees the whole formula at once.
        if display_math is not None:
            display_math.append(stripped)
            if "$$" in stripped:
                lines.append(_finish_sentence(" ".join(display_math)))
                display_math = None
            continue
        if stripped.startswith("$$") and stripped.count("$$") == 1:
            display_math = [stripped]
            continue

        if stripped.startswith(">"):
            content = re.sub(r"^(>\s?)+", "", stripped)
            callout = _CALLOUT_RE.match(content)
            if callout:
                skipping_callout = callout.group(1).casefold() in skip
                content = "" if skipping_callout else callout.group(2)
            elif skipping_callout:
                continue
            stripped = content.strip()
        else:
            skipping_callout = False

        if stripped.startswith("|"):
            if not in_table:
                tables += 1
            in_table = True
            continue
        in_table = False

        if not stripped or _RULE_RE.match(stripped) or _TAG_ONLY_RE.match(stripped):
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            stripped = heading.group(2)
        stripped = _LIST_MARKER_RE.sub("", stripped)

        text = _finish_sentence(clean_inline(stripped))
        if any(ch.isalnum() for ch in text):
            lines.append(text)

    if display_math is not None:  # an unclosed $$ - keep what's there rather than lose it
        lines.append(_finish_sentence(" ".join(display_math)))

    return CleanedNote(lines=lines, skipped_code_blocks=code_blocks, skipped_tables=tables)
