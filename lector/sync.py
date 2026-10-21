"""Keep an audio mirror of the vault up to date, one track per section.

The vault is the source of truth; this module makes a folder of audio
that follows it:

    <out>/Math/02_Probability — Теория вероятностей/01 Пространство вероятностей.mp3
                                                   02 Аксиоматика Колмогорова, сигма-алгебра.mp3

One note = one folder = one "book" to an audiobook player; one section
(a ## or ### heading with everything under it) = one track, tagged with
album/track/title so players show it as a chapter. Sources sections are
left out, as in lector-read.

Runs are incremental. A manifest in <out> remembers, per note, each
section's hash - of its cleaned text plus everything that changes how it
sounds (voice, formula model) - and which file holds it. A section whose
hash is unchanged is never voiced again: if only its position or heading
moved, its file is renamed and retagged without re-encoding; sections
that disappeared lose their files; new and edited ones are voiced. So
after the first full pass, a night's run costs only what was edited that
day. The manifest is saved after every section, so an interrupted run
picks up where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

from lector.knowledge_base import parse_note
from lector.obsidian import DEFAULT_EXCLUDED_SECTIONS, clean_markdown, list_headings, select_sections

MANIFEST_NAME = ".lector-sync.json"
MANIFEST_VERSION = 1

# Not content: vault plumbing, attachments, Obsidian's own folders, and
# the drafts inbox (voiced once a note is moved into the vault proper).
SKIPPED_TOP_FOLDERS = {"_Meta", "_Inbox", ".obsidian", ".trash", ".git", ".claude", "assets", "Plugins data"}

_UNSAFE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class Section:
    """One voiceable unit of a note."""
    index: int        # 1-based position in the note
    title: str        # heading as spoken/tagged, markup and emoji removed
    lines: List[str]  # cleaned lines (heading first), notation still raw

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class Track:
    """What a section becomes on disk."""
    section: Section
    key: str          # content hash - identity across runs
    path: Path        # relative to the output folder
    metadata: Dict[str, str]


@dataclass
class NotePlan:
    note: str                                                   # vault-relative path
    keep: List[Track] = field(default_factory=list)             # unchanged, in place
    move: List[tuple] = field(default_factory=list)             # (old relative path, Track)
    voice: List[Track] = field(default_factory=list)            # new or edited
    remove: List[str] = field(default_factory=list)             # old relative paths no longer used

    @property
    def is_noop(self) -> bool:
        return not (self.move or self.voice or self.remove)


def safe_name(text: str, limit: int = 80) -> str:
    """A file/folder name that works on Windows and Android."""
    name = _UNSAFE_CHARS_RE.sub(" ", text)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:limit].rstrip(" .") or "untitled"


def note_title(note) -> str:
    """Frontmatter title, else the first # heading, else the file name."""
    if note.frontmatter.get("title"):
        return str(note.frontmatter["title"])
    for heading in list_headings(note.body):
        if heading.level == 1:
            return _spoken_title(heading.text)
    return note.path.stem


def _spoken_title(heading_text: str) -> str:
    cleaned = clean_markdown(heading_text).lines
    return cleaned[0].rstrip(".") if cleaned else heading_text.strip()


def split_sections(body: str, level: int = 3,
                   exclude: Sequence[str] = DEFAULT_EXCLUDED_SECTIONS) -> List[Section]:
    """Cut a note body into sections at every heading of `level` or above
    (deeper headings stay inside their section). The text before the
    first such heading - usually a # title and an intro - is a section of
    its own if it says anything beyond the title. Sections that are only a
    heading (a "## Theory" directly followed by its "###" subsections) are
    dropped: nothing to say."""
    body = select_sections(body, exclude=exclude)
    lines = body.splitlines()
    bounds = [h for h in list_headings(body) if 2 <= h.level <= level]
    starts = [0] + [h.line_index for h in bounds]
    titles = [None] + [h.text for h in bounds]

    sections: List[Section] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        cleaned = clean_markdown("\n".join(lines[start:end])).lines
        heading = titles[position]
        if heading is None:
            # Preamble: a lone "# Title." line is the note's name, not content.
            content = [line for line in cleaned if not _is_h1_only(line, lines[start:end])]
            if not content:
                continue
            title = "Вступление" if _mostly_cyrillic(" ".join(content)) else "Introduction"
        else:
            title = _spoken_title(heading)
            if len(cleaned) <= 1:
                continue
        sections.append(Section(index=len(sections) + 1, title=title, lines=cleaned))
    return sections


def _is_h1_only(cleaned_line: str, raw_lines: List[str]) -> bool:
    h1 = [raw[2:].strip() for raw in raw_lines if raw.startswith("# ")]
    return bool(h1) and cleaned_line.rstrip(".") == _spoken_title(h1[0])


def _mostly_cyrillic(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and sum("Ѐ" <= ch <= "ӿ" for ch in letters) / len(letters) >= 0.3


def section_key(section: Section, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}\0{section.text}".encode("utf-8")).hexdigest()[:16]


def note_folder(vault_relative: Path, title: str) -> Path:
    """<top-level vault folder>/<note file name — title>: sorts like the
    vault does, reads like the note."""
    top = vault_relative.parts[0] if len(vault_relative.parts) > 1 else ""
    name = vault_relative.stem if title in (vault_relative.stem, "") else f"{vault_relative.stem} — {title}"
    return Path(top) / safe_name(name) if top else Path(safe_name(name))


def build_tracks(vault_relative: Path, title: str, sections: List[Section], namespace: str) -> List[Track]:
    folder = note_folder(vault_relative, title)
    total = len(sections)
    return [
        Track(
            section=section,
            key=section_key(section, namespace),
            path=folder / f"{section.index:02d} {safe_name(section.title)}.mp3",
            metadata={"title": section.title, "album": title, "artist": "lector", "album_artist": "lector",
                      "track": f"{section.index}/{total}", "genre": "Speech"},
        )
        for section in sections
    ]


def plan_note(note: str, tracks: List[Track], previous: List[dict], out_dir: Path) -> NotePlan:
    """Compare this run's tracks with what the manifest says is on disk."""
    plan = NotePlan(note=note)
    available = {entry["key"]: entry["path"] for entry in previous if (out_dir / entry["path"]).exists()}
    used = set()
    for track in tracks:
        old_path = available.get(track.key)
        if old_path is None or old_path in used:
            plan.voice.append(track)
            continue
        used.add(old_path)
        entry = next(e for e in previous if e["path"] == old_path)
        if old_path == track.path.as_posix() and entry.get("metadata") == track.metadata:
            plan.keep.append(track)
        else:
            plan.move.append((old_path, track))
    new_paths = {track.path.as_posix() for track in tracks}
    plan.remove = [e["path"] for e in previous if e["path"] not in used and e["path"] not in new_paths]
    return plan


def load_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_NAME
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("version") == MANIFEST_VERSION:
            return manifest
    return {"version": MANIFEST_VERSION, "notes": {}}


def save_manifest(out_dir: Path, manifest: dict) -> None:
    path = out_dir / MANIFEST_NAME
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    partial.replace(path)


def iter_notes(vault: Path, folders: Sequence[str] = (), notes: Sequence[str] = ()) -> Iterable[Path]:
    """Vault-relative paths of the notes in scope, sorted."""
    if notes:
        for note in notes:
            path = Path(note)
            yield path if path.suffix == ".md" else path.with_suffix(".md")
        return
    roots = [vault / folder for folder in folders] if folders else [vault]
    for root in roots:
        for path in sorted(root.rglob("*.md")):
            relative = path.relative_to(vault)
            if relative.parts[0] in SKIPPED_TOP_FOLDERS or any(p.startswith(".") for p in relative.parts):
                continue
            yield relative


@dataclass
class SyncReport:
    notes_seen: int = 0
    kept: int = 0
    moved: int = 0
    voiced: int = 0
    removed: int = 0
    voiced_chars: int = 0
    failed: List[str] = field(default_factory=list)


Voicer = Callable[[Section, Path, Dict[str, str]], None]


def sync(vault: Path, out_dir: Path, namespace: str, voice_section: Voicer, *,
         folders: Sequence[str] = (), notes: Sequence[str] = (), level: int = 3,
         exclude: Sequence[str] = DEFAULT_EXCLUDED_SECTIONS, dry_run: bool = False,
         log: Callable[[str], None] = print) -> SyncReport:
    """Bring `out_dir` in line with the notes in scope. `voice_section`
    writes one section's audio, tagged, to the given absolute path - the
    caller decides how (formula model, TTS backend). Notes that left the
    vault (or the scope, when scope is the whole vault) lose their audio."""
    from lector.ffmpeg import retag_audio

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(out_dir)
    report = SyncReport()
    seen = set()

    for relative in iter_notes(vault, folders, notes):
        key = relative.as_posix()
        source = vault / relative
        if not source.is_file():
            log(f"! {key}: not found")
            continue
        seen.add(key)
        report.notes_seen += 1
        note = parse_note(source)
        title = note_title(note)
        tracks = build_tracks(relative, title, split_sections(note.body, level, exclude), namespace)
        plan = plan_note(key, tracks, manifest["notes"].get(key, []), out_dir)
        report.kept += len(plan.keep)
        if plan.is_noop:
            continue

        chars = sum(len(t.section.text) for t in plan.voice)
        log(f"{key}: {len(plan.voice)} to voice ({chars} chars), {len(plan.move)} to move, "
            f"{len(plan.remove)} to remove, {len(plan.keep)} unchanged")
        if dry_run:
            report.moved += len(plan.move)
            report.voiced += len(plan.voice)
            report.removed += len(plan.remove)
            report.voiced_chars += chars
            continue

        done: Dict[str, dict] = {t.key: _entry(t) for t in plan.keep}
        with tempfile.TemporaryDirectory() as tmp:
            # Moves go through a temp copy first: a reordering can make one
            # section's new path another's old one.
            staged = []
            for old_path, track in plan.move:
                temp = Path(tmp) / f"{track.key}.mp3"
                retag_audio(out_dir / old_path, temp, track.metadata)
                staged.append(track)
                (out_dir / old_path).unlink()
            for path in plan.remove:
                (out_dir / path).unlink(missing_ok=True)
            for track in staged:
                target = out_dir / track.path
                target.parent.mkdir(parents=True, exist_ok=True)
                (Path(tmp) / f"{track.key}.mp3").replace(target)
                done[track.key] = _entry(track)
                report.moved += 1
        report.removed += len(plan.remove)
        _store(manifest, key, tracks, done, out_dir)

        for track in plan.voice:
            target = out_dir / track.path
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                voice_section(track.section, target, track.metadata)
            except Exception as exc:  # noqa: BLE001 - one bad section shouldn't stop the vault
                report.failed.append(f"{key} #{track.section.index} {track.section.title}: {exc}")
                log(f"! {key} #{track.section.index}: {exc}")
                continue
            done[track.key] = _entry(track)
            report.voiced += 1
            report.voiced_chars += len(track.section.text)
            _store(manifest, key, tracks, done, out_dir)
        _prune_empty_dirs(out_dir)

    if not notes and not dry_run:
        for gone in sorted(set(manifest["notes"]) - seen):
            if folders and not any(gone.startswith(f.rstrip("/") + "/") for f in folders):
                continue  # outside this run's scope, not deleted from the vault
            for entry in manifest["notes"].pop(gone):
                (out_dir / entry["path"]).unlink(missing_ok=True)
                report.removed += 1
            log(f"{gone}: note gone, audio removed")
        save_manifest(out_dir, manifest)
        _prune_empty_dirs(out_dir)
    return report


def _entry(track: Track) -> dict:
    return {"key": track.key, "path": track.path.as_posix(), "title": track.section.title,
            "metadata": track.metadata}


def _store(manifest: dict, note_key: str, tracks: List[Track], done: Dict[str, dict], out_dir: Path) -> None:
    """Record what's on disk for this note now, in note order, and save."""
    manifest["notes"][note_key] = [done[t.key] for t in tracks if t.key in done]
    save_manifest(out_dir, manifest)


def _prune_empty_dirs(out_dir: Path) -> None:
    for directory in sorted((p for p in out_dir.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        if not any(directory.iterdir()):
            directory.rmdir()


def estimate_minutes(chars: int, chars_per_second: float = 14.0) -> float:
    """Rough audio length - ~14 characters per second is what the
    2026-10 test readings of Russian notes averaged."""
    return chars / chars_per_second / 60
