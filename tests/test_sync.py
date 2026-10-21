"""Tests for lector/sync.py.

The voicing step is a fake that writes a tiny real mp3 (via ffmpeg) so
the moves - which retag files without re-encoding - run for real. What's
pinned down: how notes split into sections, that an unchanged section is
never voiced twice, that reordering/renaming moves files instead of
paying again, and that deleted sections and notes lose their audio.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from lector.sync import MANIFEST_NAME, safe_name, split_sections, sync

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

NOTE = """---
tags: [math]
---
# Вероятность

> [!map]- Navigation
> [[A]] · [[B]]

## 📚 Теория

### Первый раздел
Текст первого раздела.

### Второй раздел
Текст второго раздела с $x^2$.

#### Подраздел
Остаётся внутри второго.

### 🔗 Источники
- [[Some paper]]
"""


class _FakeVoicer:
    def __init__(self):
        self.voiced = []

    def __call__(self, section, target, metadata):
        self.voiced.append(section.title)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", "anullsrc=r=8000:cl=mono", "-t", "0.1", "-q:a", "9",
                        *sum((["-metadata", f"{k}={v}"] for k, v in metadata.items()), []), str(target)],
                       check=True)


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    (root / "Math").mkdir(parents=True)
    (root / "Math" / "02_Prob.md").write_text(NOTE, encoding="utf-8")
    (root / "_Meta").mkdir()
    (root / "_Meta" / "Index.md").write_text("# Index\nNot content.\n", encoding="utf-8")
    return root


def _run(vault, out, voicer, **kwargs):
    return sync(vault, out, "voice-a", voicer, log=lambda _: None, **kwargs)


def _files(out):
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*.mp3"))


# -- splitting ----------------------------------------------------------------

def test_split_sections_by_level_3_keeping_deeper_headings_inside():
    sections = split_sections(NOTE.split("---\n", 2)[2])

    assert [s.title for s in sections] == ["Первый раздел", "Второй раздел"]
    assert "Остаётся внутри второго." in sections[1].text
    assert all("Some paper" not in s.text for s in sections)


def test_heading_only_and_title_only_parts_are_dropped():
    sections = split_sections("# Title\n\n## Group\n\n### Real\nBody.\n")

    assert [s.title for s in sections] == ["Real"]


def test_preamble_with_content_becomes_its_own_section():
    sections = split_sections("# Заголовок\nВводный абзац.\n\n## Раздел\nТекст.\n")

    assert [s.title for s in sections] == ["Вступление", "Раздел"]


def test_level_2_merges_subsections():
    sections = split_sections("## A\nOne.\n### B\nTwo.\n## C\nThree.\n", level=2)

    assert [s.title for s in sections] == ["A", "C"]
    assert "Two." in sections[0].text


def test_safe_name_strips_characters_windows_forbids():
    assert safe_name('Bayes: P(A|B) / "MAP"?') == "Bayes P(A B) MAP"


# -- syncing ------------------------------------------------------------------

def test_first_run_voices_every_section_into_a_folder_per_note(vault, tmp_path):
    out, voicer = tmp_path / "audio", _FakeVoicer()

    report = _run(vault, out, voicer)

    assert voicer.voiced == ["Первый раздел", "Второй раздел"]
    assert _files(out) == ["Math/02_Prob — Вероятность/01 Первый раздел.mp3",
                           "Math/02_Prob — Вероятность/02 Второй раздел.mp3"]
    assert report.voiced == 2 and report.notes_seen == 1  # _Meta is skipped


def test_second_run_voices_nothing(vault, tmp_path):
    out = tmp_path / "audio"
    _run(vault, out, _FakeVoicer())
    voicer = _FakeVoicer()

    report = _run(vault, out, voicer)

    assert voicer.voiced == []
    assert report.kept == 2


def test_edit_revoices_only_the_edited_section(vault, tmp_path):
    out = tmp_path / "audio"
    _run(vault, out, _FakeVoicer())
    note = vault / "Math" / "02_Prob.md"
    note.write_text(note.read_text(encoding="utf-8").replace("Текст первого", "Новый текст первого"),
                    encoding="utf-8")
    voicer = _FakeVoicer()

    _run(vault, out, voicer)

    assert voicer.voiced == ["Первый раздел"]


def test_inserted_section_shifts_others_without_revoicing_them(vault, tmp_path):
    out = tmp_path / "audio"
    _run(vault, out, _FakeVoicer())
    note = vault / "Math" / "02_Prob.md"
    note.write_text(note.read_text(encoding="utf-8").replace(
        "### Первый раздел", "### Нулевой раздел\nНовый текст.\n\n### Первый раздел"), encoding="utf-8")
    voicer = _FakeVoicer()

    report = _run(vault, out, voicer)

    assert voicer.voiced == ["Нулевой раздел"]
    assert report.moved == 2
    assert _files(out) == ["Math/02_Prob — Вероятность/01 Нулевой раздел.mp3",
                           "Math/02_Prob — Вероятность/02 Первый раздел.mp3",
                           "Math/02_Prob — Вероятность/03 Второй раздел.mp3"]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=track",
                            "-of", "default=nw=1:nk=1", str(out / _files(out)[2])],
                           capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "3/3"


def test_removed_section_and_removed_note_lose_their_audio(vault, tmp_path):
    out = tmp_path / "audio"
    _run(vault, out, _FakeVoicer())
    note = vault / "Math" / "02_Prob.md"
    text = note.read_text(encoding="utf-8")
    note.write_text(text[:text.index("### Второй раздел")], encoding="utf-8")

    _run(vault, out, _FakeVoicer())
    assert _files(out) == ["Math/02_Prob — Вероятность/01 Первый раздел.mp3"]

    note.unlink()
    _run(vault, out, _FakeVoicer())
    assert _files(out) == []
    assert json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))["notes"] == {}


def test_a_different_voice_revoices_everything(vault, tmp_path):
    out = tmp_path / "audio"
    _run(vault, out, _FakeVoicer())
    voicer = _FakeVoicer()

    sync(vault, out, "voice-b", voicer, log=lambda _: None)

    assert len(voicer.voiced) == 2


def test_failed_section_is_reported_and_retried_next_run(vault, tmp_path):
    out = tmp_path / "audio"

    def flaky(section, target, metadata):
        if section.title == "Второй раздел":
            raise RuntimeError("TTS down")
        _FakeVoicer()(section, target, metadata)

    report = _run(vault, out, flaky)
    assert len(report.failed) == 1
    voicer = _FakeVoicer()
    _run(vault, out, voicer)
    assert voicer.voiced == ["Второй раздел"]


def test_dry_run_touches_nothing(vault, tmp_path):
    out = tmp_path / "audio"

    report = _run(vault, out, None, dry_run=True)

    assert report.voiced == 2 and report.voiced_chars > 0
    assert _files(out) == []


def test_scoped_run_keeps_audio_of_notes_outside_its_scope(vault, tmp_path):
    out = tmp_path / "audio"
    (vault / "AI").mkdir()
    (vault / "AI" / "01_X.md").write_text("## S\nText.\n", encoding="utf-8")
    _run(vault, out, _FakeVoicer())

    _run(vault, out, _FakeVoicer(), folders=["Math"])

    assert any(path.startswith("AI/") for path in _files(out))
