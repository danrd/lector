"""Tests for lector/sync_cli.py - wiring only: the dry run reports and
writes nothing, and a real run goes formulas -> TTS -> tagged mp3, with
the formula review written beside the audio. The LLM and TTS builders
are replaced with fakes; ffmpeg runs for real."""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import wave

import pytest

import lector.sync_cli as sync_cli
from lector.sync_cli import main
from lector.tts_runtime import BaseTTSRunner

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class _FakeTTS(BaseTTSRunner):
    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 1600)
        return buffer.getvalue()


class _FakeLLM:
    def generate(self, prompt):
        lines = json.JSONDecoder().raw_decode(prompt[prompt.index("["):])[0]
        return json.dumps([line.replace("$x^2$", "икс квадрат") for line in lines], ensure_ascii=False)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "Math").mkdir(parents=True)
    (vault / "Math" / "01_Alg.md").write_text("# Алгебра\n\n### Квадрат\nЧисло $x^2$ неотрицательно.\n",
                                              encoding="utf-8")
    tts = _FakeTTS()
    monkeypatch.setattr(sync_cli, "build_tts", lambda args: tts)
    monkeypatch.setattr(sync_cli, "build_llm_runner", lambda model: _FakeLLM())
    return vault, tmp_path / "audio", tts


def test_dry_run_reports_without_writing(setup, capsys):
    vault, out, tts = setup

    assert main(["--vault", str(vault), "--out", str(out), "--dry-run"]) == 0

    assert "1 section(s) to voice" in capsys.readouterr().out
    assert tts.calls == []
    assert not list(out.rglob("*.mp3"))


def test_run_voices_verbalized_text_into_a_tagged_mp3(setup):
    vault, out, tts = setup

    assert main(["--vault", str(vault), "--out", str(out), "--no-cache"]) == 0

    assert tts.calls == ["Квадрат. Число икс квадрат неотрицательно."]
    track = out / "Math" / "01_Alg — Алгебра" / "01 Квадрат.mp3"
    tags = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=album,title,track",
                           "-of", "json", str(track)], capture_output=True, text=True, check=True).stdout
    assert json.loads(tags)["format"]["tags"] == {"album": "Алгебра", "title": "Квадрат", "track": "1/1"}
    review = next(out.glob("review-*.md")).read_text(encoding="utf-8")
    assert "икс квадрат" in review
