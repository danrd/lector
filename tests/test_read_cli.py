"""Tests for lector/read_cli.py.

Like test_cli.py, this pins down wiring rather than re-testing the
pieces: argument validation, where outputs land, that --formulas raw
needs no model and --script-only no synthesis, and that the TTS backend
and cache are composed the way the flags say. The two network-facing
builders (build_llm_runner, build_tts) are replaced with fakes.
"""
from __future__ import annotations

import io
import wave

import pytest

import lector.read_cli as read_cli
from lector.read_cli import build_parser, cache_namespace, default_output, main, resolve_note, validate_args
from lector.tts_runtime import BaseTTSRunner

NOTE = """---
tags: [math]
---
# Вероятность

## Теория
Обычное предложение.
нормировка: $P(\\Omega)=1$;

### 🔗 Источники
- [[Some paper]]

## Исследования
Другой раздел.
"""


def _wav(frames: int = 2400) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


class _FakeTTS(BaseTTSRunner):
    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return _wav()


class _FakeLLM:
    """Verbalizes by a fixed table, answering in verbalize's JSON format."""
    def generate(self, prompt):
        import json
        lines = json.JSONDecoder().raw_decode(prompt[prompt.index("["):])[0]
        return json.dumps([line.replace("$P(\\Omega)=1$", "пэ от омеги равно единице") for line in lines],
                          ensure_ascii=False)


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "Math").mkdir()
    (tmp_path / "Math" / "Prob.md").write_text(NOTE, encoding="utf-8")
    return tmp_path


@pytest.fixture
def fakes(monkeypatch):
    tts = _FakeTTS()
    monkeypatch.setattr(read_cli, "build_tts", lambda args: tts)
    monkeypatch.setattr(read_cli, "build_llm_runner", lambda model: _FakeLLM())
    return tts


def _args(*argv):
    return build_parser().parse_args(list(argv))


# -- validation ---------------------------------------------------------------

def test_needs_exactly_one_of_note_or_script():
    with pytest.raises(SystemExit):
        validate_args(_args())
    with pytest.raises(SystemExit):
        validate_args(_args("note", "--script", "s.txt"))


def test_voice_ref_needs_comet_seed_audio_and_an_existing_file(tmp_path):
    reference = tmp_path / "me.mp3"
    reference.write_bytes(b"x")

    validate_args(_args("note", "--voice-ref", str(reference)))
    with pytest.raises(SystemExit, match="seed_audio"):
        validate_args(_args("note", "--voice-ref", str(reference), "--model", "eleven_v3"))
    with pytest.raises(SystemExit, match="--tts comet"):
        validate_args(_args("note", "--voice-ref", str(reference), "--tts", "openai"))
    with pytest.raises(SystemExit, match="not found"):
        validate_args(_args("note", "--voice-ref", str(tmp_path / "missing.mp3")))


def test_piper_needs_a_voice_model():
    with pytest.raises(SystemExit, match="--voice"):
        validate_args(_args("note", "--tts", "piper"))


# -- paths --------------------------------------------------------------------

def test_resolve_note_tries_vault_then_cwd_and_adds_md(vault):
    assert resolve_note("Math/Prob", str(vault)) == vault / "Math" / "Prob.md"
    assert resolve_note(str(vault / "Math" / "Prob.md"), None) == vault / "Math" / "Prob.md"
    with pytest.raises(SystemExit, match="not found"):
        resolve_note("Math/Nope", str(vault))


def test_default_output_names_the_sections():
    assert str(default_output("02_Probability", [])) == "02_Probability.mp3"
    assert str(default_output("Prob", ["Теорема Байеса", "MLE"])) == "Prob-Теорема_Байеса-MLE.mp3"


def test_cache_namespace_changes_with_voice_and_reference(tmp_path):
    first = tmp_path / "a.mp3"
    first.write_bytes(b"one")
    second = tmp_path / "b.mp3"
    second.write_bytes(b"two")

    namespaces = {
        cache_namespace(_args("n")),
        cache_namespace(_args("n", "--voice", "Maya")),
        cache_namespace(_args("n", "--voice-ref", str(first))),
        cache_namespace(_args("n", "--voice-ref", str(second))),
    }

    assert len(namespaces) == 4


# -- main ---------------------------------------------------------------------

def test_list_sections_prints_headings(vault, fakes, capsys):
    assert main(["Math/Prob", "--vault", str(vault), "--list-sections"]) == 0

    out = capsys.readouterr().out
    assert "  Теория" in out
    assert fakes.calls == []


def test_unknown_section_exits_with_the_heading_list(vault, fakes):
    with pytest.raises(SystemExit, match="Исследования"):
        main(["Math/Prob", "--vault", str(vault), "--section", "Галуа"])


def test_script_only_with_raw_formulas_writes_script_and_no_audio(vault, fakes, tmp_path):
    output = tmp_path / "out" / "prob.mp3"

    code = main(["Math/Prob", "--vault", str(vault), "--section", "Теория", "--formulas", "raw",
                 "--script-only", "--output", str(output)])

    assert code == 0
    script = (tmp_path / "out" / "prob.txt").read_text(encoding="utf-8")
    assert script.splitlines() == ["Теория.", "Обычное предложение.", "нормировка: P(\\Omega)=1;"]
    assert not (tmp_path / "out" / "prob.formulas.md").exists()
    assert not output.exists()
    assert fakes.calls == []


def test_full_run_verbalizes_reviews_and_synthesizes(vault, fakes, tmp_path, capsys):
    output = tmp_path / "prob.wav"

    code = main(["Math/Prob", "--vault", str(vault), "--section", "Теория",
                 "--output", str(output), "--no-cache"])

    assert code == 0
    script = (tmp_path / "prob.txt").read_text(encoding="utf-8")
    assert "нормировка: пэ от омеги равно единице;" in script
    assert "Some paper" not in script  # the sources list is skipped
    review = (tmp_path / "prob.formulas.md").read_text(encoding="utf-8")
    assert "пэ от омеги равно единице" in review
    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnframes() > 0
    assert fakes.calls
    assert "Wrote" in capsys.readouterr().out


def test_script_mode_voices_an_edited_script_through_the_cache(fakes, tmp_path):
    script = tmp_path / "edited.txt"
    script.write_text("Первое предложение. Второе предложение.", encoding="utf-8")
    cache = tmp_path / "cache"

    for _ in range(2):
        assert main(["--script", str(script), "--output", str(tmp_path / "x.wav"),
                     "--cache-dir", str(cache)]) == 0

    assert fakes.calls == ["Первое предложение. Второе предложение."]  # second run came from the cache
