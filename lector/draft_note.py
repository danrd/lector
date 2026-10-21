"""A generated lecture as a draft note in the vault's inbox.

The vault is where content lives and gets voiced (lector-sync); a
generated lecture is new content that hasn't been vetted yet, so it lands
in a top-level `_Inbox/` folder as a draft - read it, fix it, move the
parts worth keeping into a real note. lector-sync skips `_Inbox/`, so
nothing unvetted gets voiced into the audio mirror by accident.

The draft keeps its review state visible instead of hiding it:
unresolved claims stay inline as [NEEDS REVIEW: ...], the verification
outcome of every flagged claim is listed at the end, and the vault notes
the draft was grounded in are linked - so "where did this come from and
what's still in doubt" is answered inside the note itself.
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import List, Optional, Sequence

from lector.knowledge_base import Note
from lector.verification import ClaimVerdict, VerificationResult

INBOX_FOLDER = "_Inbox"

_VERDICT_LABELS = {
    ClaimVerdict.GROUNDED_IN_VAULT: "✅ подтверждено заметками",
    ClaimVerdict.CONFIRMED_BY_MODEL: "☑️ подтверждено моделью (не источником)",
    ClaimVerdict.UNCERTAIN: "❓ требует проверки",
}
_UNSAFE_CHARS_RE = re.compile(r'[<>:"/\\|?*#^\[\]\x00-\x1f]')


def _yaml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _wikilink(note: Note, vault: Optional[Path]) -> str:
    if vault is not None:
        try:
            target = note.path.relative_to(vault).with_suffix("").as_posix()
            return f"[[{target}|{note.title}]]"
        except ValueError:
            pass
    return f"[[{note.path.stem}]]"


def _drop_leading_title(draft: str) -> str:
    """The model usually opens with its own "# Topic" - the note already
    has one, and two H1s would make Obsidian's outline (and lector-sync's
    note title) ambiguous."""
    lines = draft.strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def render_draft_note(topic: str, annotated_draft: str, verification: Sequence[VerificationResult],
                      sources: Sequence[Note] = (), vault: Optional[Path] = None,
                      model: str = "", today: Optional[_dt.date] = None) -> str:
    today = today or _dt.date.today()
    uncertain = sum(r.verdict == ClaimVerdict.UNCERTAIN for r in verification)
    lines = [
        "---",
        "status: draft",
        "source: lector",
        f"topic: {_yaml_string(topic)}",
        f"created: {today.isoformat()}",
        *([f"model: {_yaml_string(model)}"] if model else []),
        "tags: []",
        "---",
        "",
        f"# {topic}",
        "",
        "> [!warning] Черновик lector",
        "> Сгенерирован автоматически"
        + (f", опирается на {len(sources)} заметок хранилища" if sources else ", без опоры на заметки хранилища")
        + f". Проверено утверждений: {len(verification)}, осталось под вопросом: {uncertain}"
        + " (помечены `[NEEDS REVIEW: ...]` в тексте).",
        "> Перед переносом в хранилище: вычитать, снять пометки, оставить нужное.",
        "",
        _drop_leading_title(annotated_draft),
        "",
    ]
    if verification:
        lines += ["---", "", "## Проверка утверждений", ""]
        for result in verification:
            note = f" — {result.note}" if result.note else ""
            lines.append(f"- {_VERDICT_LABELS[result.verdict]}: {result.claim.text}{note}")
        lines.append("")
    if sources:
        lines += ["## Опора на заметки", ""]
        lines += [f"- {_wikilink(note, vault)}" for note in sources]
        lines.append("")
    return "\n".join(lines)


def draft_note_path(inbox: Path, topic: str, today: Optional[_dt.date] = None) -> Path:
    """_Inbox/<topic>.md, or <topic> (<date>).md / (<date> 2) ... if a
    draft on the same topic is already there - never overwrite a draft
    that may have been edited."""
    today = today or _dt.date.today()
    name = re.sub(r"\s+", " ", _UNSAFE_CHARS_RE.sub(" ", topic)).strip(" .")[:100] or "draft"
    candidates = [f"{name}.md", f"{name} ({today.isoformat()}).md"]
    candidates += [f"{name} ({today.isoformat()} {n}).md" for n in range(2, 100)]
    for candidate in candidates:
        if not (inbox / candidate).exists():
            return inbox / candidate
    raise FileExistsError(f"Too many drafts on '{topic}' in {inbox}")


def write_draft_note(inbox: Path, topic: str, annotated_draft: str,
                     verification: Sequence[VerificationResult], sources: List[Note] = (),
                     vault: Optional[Path] = None, model: str = "") -> Path:
    inbox.mkdir(parents=True, exist_ok=True)
    path = draft_note_path(inbox, topic)
    path.write_text(render_draft_note(topic, annotated_draft, verification, sources, vault, model),
                    encoding="utf-8")
    return path
