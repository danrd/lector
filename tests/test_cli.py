"""Tests for lector/cli.py.

The CLI is composition, not logic, so what's worth pinning down is the
wiring: does each flag reach the thing it configures, is a vault really
optional, does an early-stopped run report the stage it stopped at
instead of pretending it produced audio. The pipeline itself is faked
out here - it has its own tests in test_pipeline.py, and running it for
real would mean an OpenRouter key and a downloaded voice model.
"""
from __future__ import annotations

import pytest

from lector.cli import build_parser, load_notes, main
from lector.pipeline import LectureAudioResult, PipelineStage
from lector.narration import NarrationResult
from lector.audio import AudioResult
from lector.tts_runtime import BaseTTSRunner


class _FakeTTSRunner(BaseTTSRunner):
    def __init__(self):
        self.closed = False

    def synthesize(self, text: str) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True


def _done_result(topic="Some topic", script="A spoken script.", excluded=None,
                  leftover=False, verification=None):
    return LectureAudioResult(
        stage_reached=PipelineStage.DONE, topic=topic, draft="draft",
        verification_results=verification or [],
        annotated_draft="draft",
        narration=NarrationResult(script=script, excluded_claims=excluded or [],
                                   has_leftover_markdown=leftover),
        audio=AudioResult(output_path="out.wav", num_chunks=2, duration_seconds=12.5),
    )


@pytest.fixture
def wired(monkeypatch):
    """Replaces both external dependencies (LLM runner, TTS backend) and
    captures what generate_lecture_audio was called with."""
    import lector.cli as cli

    captured = {}
    tts_runner = _FakeTTSRunner()

    def fake_generate(topic, notes, tokenizer, runner, tts, output_path, config):
        captured.update(topic=topic, notes=notes, tokenizer=tokenizer, runner=runner,
                         tts=tts, output_path=output_path, config=config)
        return _done_result(topic=topic)

    monkeypatch.setattr(cli, "build_llm_runner", lambda model, max_tokens: f"runner:{model}")
    monkeypatch.setattr(cli, "build_tts_runner", lambda backend, voice, speed: tts_runner)
    monkeypatch.setattr(cli, "generate_lecture_audio", fake_generate)

    captured["tts_runner"] = tts_runner
    return captured


# -- argument parsing ------------------------------------------------------

def test_topic_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_defaults_are_the_documented_ones():
    args = build_parser().parse_args(["Galois theory"])

    assert args.topic == "Galois theory"
    assert args.vault is None
    assert args.output == "lecture.wav"
    assert args.tts == "piper"  # local backend by default, not the hosted one


# -- vault loading ---------------------------------------------------------

def test_no_vault_means_no_notes_not_an_error():
    """knowledge_base.py treats an empty retrieval result as a normal
    outcome, so running with no vault at all has to be allowed too."""
    assert load_notes(None) == []


def test_missing_vault_path_is_an_error():
    with pytest.raises(SystemExit, match="Vault not found"):
        load_notes("/definitely/not/a/vault")


def test_vault_is_loaded_when_given(tmp_path):
    (tmp_path / "note.md").write_text("---\ntitle: Galois\n---\nbody", encoding="utf-8")

    notes = load_notes(str(tmp_path))

    assert len(notes) == 1
    assert notes[0].title == "Galois"


# -- wiring ----------------------------------------------------------------

def test_main_passes_the_topic_and_output_through(wired, tmp_path):
    output = str(tmp_path / "out.wav")

    exit_code = main(["Galois theory", "--audio", "--tts", "openai", "--output", output])

    assert exit_code == 0
    assert wired["topic"] == "Galois theory"
    assert wired["output_path"] == output


def test_language_reaches_both_lecture_and_narration(wired, tmp_path):
    """pipeline.py's docstring is explicit that these two are independent
    and not synced for you - so the CLI's single --language flag has to
    set both itself, or half the run comes out in the wrong language."""
    main(["Topic", "--audio", "--tts", "openai", "--language", "Russian", "--output", str(tmp_path / "o.wav")])

    config = wired["config"]
    assert config.lecture.language == "Russian"
    assert config.narration.language == "Russian"


def test_max_chunk_chars_reaches_the_config(wired, tmp_path):
    main(["Topic", "--audio", "--tts", "openai", "--max-chunk-chars", "42", "--output", str(tmp_path / "o.wav")])

    assert wired["config"].max_chunk_chars == 42


def test_model_flag_reaches_the_llm_runner(wired, tmp_path):
    main(["Topic", "--audio", "--tts", "openai", "--model", "some/model", "--output", str(tmp_path / "o.wav")])

    assert wired["runner"] == "runner:some/model"


def test_tts_runner_is_closed_even_though_the_pipeline_returned(wired, tmp_path):
    """The runner holds a loaded model - the CLI uses it as a context
    manager so it's released whether the run succeeded or not."""
    main(["Topic", "--audio", "--tts", "openai", "--output", str(tmp_path / "o.wav")])

    assert wired["tts_runner"].closed is True


def test_running_without_a_vault_passes_an_empty_note_list(wired, tmp_path):
    main(["Topic", "--audio", "--tts", "openai", "--output", str(tmp_path / "o.wav")])

    assert wired["notes"] == []


# -- reporting -------------------------------------------------------------

def test_a_completed_run_writes_the_script_beside_the_wav(wired, tmp_path):
    output = tmp_path / "lecture.wav"

    exit_code = main(["Topic", "--audio", "--tts", "openai", "--output", str(output)])

    assert exit_code == 0
    assert (tmp_path / "lecture.txt").read_text(encoding="utf-8") == "A spoken script."


def test_stopping_at_drafting_exits_nonzero(monkeypatch, tmp_path, capsys):
    import lector.cli as cli

    monkeypatch.setattr(cli, "build_llm_runner", lambda model, max_tokens: object())
    monkeypatch.setattr(cli, "build_tts_runner", lambda backend, voice, speed: _FakeTTSRunner())
    monkeypatch.setattr(cli, "generate_lecture_audio",
                        lambda *a, **k: LectureAudioResult(
                            stage_reached=PipelineStage.DRAFTING, topic="Topic"))

    exit_code = main(["Topic", "--audio", "--tts", "openai", "--output", str(tmp_path / "o.wav")])

    assert exit_code == 1
    assert "Stopped at drafting" in capsys.readouterr().err
    assert not (tmp_path / "o.txt").exists()  # nothing written for a run that produced nothing


def test_stopping_at_narration_exits_nonzero(monkeypatch, tmp_path, capsys):
    import lector.cli as cli

    monkeypatch.setattr(cli, "build_llm_runner", lambda model, max_tokens: object())
    monkeypatch.setattr(cli, "build_tts_runner", lambda backend, voice, speed: _FakeTTSRunner())
    monkeypatch.setattr(cli, "generate_lecture_audio",
                        lambda *a, **k: LectureAudioResult(
                            stage_reached=PipelineStage.NARRATION, topic="Topic", draft="d",
                            narration=NarrationResult(script="")))

    exit_code = main(["Topic", "--audio", "--tts", "openai", "--output", str(tmp_path / "o.wav")])

    assert exit_code == 1
    assert "Stopped at narration" in capsys.readouterr().err


def test_unresolved_claims_are_reported_not_silently_dropped(monkeypatch, tmp_path, capsys):
    """narration.py strips [NEEDS REVIEW: ...] spans before synthesis -
    the point of carrying excluded_claims is that a human still gets told
    what was left out."""
    import lector.cli as cli

    monkeypatch.setattr(cli, "build_llm_runner", lambda model, max_tokens: object())
    monkeypatch.setattr(cli, "build_tts_runner", lambda backend, voice, speed: _FakeTTSRunner())
    monkeypatch.setattr(cli, "generate_lecture_audio",
                        lambda *a, **k: _done_result(excluded=["a shaky fact"]))

    main(["Topic", "--audio", "--tts", "openai", "--output", str(tmp_path / "o.wav")])

    assert "a shaky fact" in capsys.readouterr().err


def test_leftover_markdown_is_warned_about(monkeypatch, tmp_path, capsys):
    import lector.cli as cli

    monkeypatch.setattr(cli, "build_llm_runner", lambda model, max_tokens: object())
    monkeypatch.setattr(cli, "build_tts_runner", lambda backend, voice, speed: _FakeTTSRunner())
    monkeypatch.setattr(cli, "generate_lecture_audio",
                        lambda *a, **k: _done_result(leftover=True))

    main(["Topic", "--audio", "--tts", "openai", "--output", str(tmp_path / "o.wav")])

    assert "still contains markdown" in capsys.readouterr().err


# -- backend selection -----------------------------------------------------

def test_piper_requires_a_voice_path():
    from lector.cli import build_parser, validate_args

    with pytest.raises(SystemExit, match="--voice is required"):
        validate_args(build_parser().parse_args(["Topic", "--audio", "--tts", "piper"]))


def test_orpheus_requires_a_model_path():
    from lector.cli import build_parser, validate_args

    with pytest.raises(SystemExit, match="--voice is required"):
        validate_args(build_parser().parse_args(["Topic", "--audio", "--tts", "orpheus"]))


def test_openai_backend_needs_no_local_model_file():
    from lector.cli import build_parser, validate_args

    validate_args(build_parser().parse_args(["Topic", "--audio", "--tts", "openai"]))  # must not raise


def test_missing_voice_is_reported_before_anything_is_constructed(monkeypatch):
    """Regression test: validation used to live inside build_tts_runner,
    which main() only reaches after building the LLM runner - so a run
    missing --voice reported whatever went wrong in the LLM step first
    (an unrelated missing extra, as a raw traceback) instead of the
    actual problem."""
    import lector.cli as cli

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("argument validation must happen before this")

    monkeypatch.setattr(cli, "build_llm_runner", must_not_be_called)
    monkeypatch.setattr(cli, "build_tts_runner", must_not_be_called)

    with pytest.raises(SystemExit, match="--voice is required"):
        cli.main(["Topic", "--audio", "--tts", "piper"])


def test_a_missing_tts_extra_names_the_extra_not_the_package(monkeypatch):
    """A bare ModuleNotFoundError names a package the user never chose by
    name (piper/snac/openai); the message has to name the extra they'd
    actually type."""
    import lector.cli as cli
    import lector.tts_runtime as tts

    def raise_import_error(*args, **kwargs):
        raise ImportError("No module named 'piper'")

    monkeypatch.setattr(tts, "build_piper_runner", raise_import_error, raising=False)
    monkeypatch.setattr(cli, "_TTS_EXTRAS", {"piper": "tts-local"})

    with pytest.raises(SystemExit, match="tts-local"):
        cli.build_tts_runner("piper", "/some/voice.onnx", 1.0)


def test_missing_openrouter_key_is_a_clear_error(monkeypatch):
    from lector.cli import build_llm_runner

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        build_llm_runner("some/model", 1024)


# -- draft-note mode (the default) -----------------------------------------

def test_draft_mode_needs_a_vault_or_inbox():
    from lector.cli import validate_args

    with pytest.raises(SystemExit, match="_Inbox"):
        validate_args(build_parser().parse_args(["Topic"]))


def test_draft_mode_does_not_need_a_voice():
    from lector.cli import validate_args

    validate_args(build_parser().parse_args(["Topic", "--inbox", "x", "--tts", "piper"]))  # must not raise


def test_draft_mode_writes_into_the_vault_inbox(monkeypatch, tmp_path, capsys):
    import lector.cli as cli
    from lector.pipeline import LectureDraftResult

    (tmp_path / "Math").mkdir()
    (tmp_path / "Math" / "Galois.md").write_text("---\ntitle: Galois theory\n---\nGalois theory groups fields.",
                                                encoding="utf-8")
    monkeypatch.setattr(cli, "build_llm_runner", lambda model, max_tokens: object())
    monkeypatch.setattr(cli, "build_tts_runner", lambda *a: pytest.fail("no TTS in draft mode"))
    monkeypatch.setattr(cli, "generate_lecture_draft", lambda topic, *a, **k: LectureDraftResult(
        stage_reached=PipelineStage.DONE, topic=topic, draft="d", annotated_draft="## Body\nText."))

    assert main(["Galois theory", "--vault", str(tmp_path)]) == 0

    draft = (tmp_path / "_Inbox" / "Galois theory.md").read_text(encoding="utf-8")
    assert "status: draft" in draft and "## Body" in draft
    assert "[[Math/Galois|Galois theory]]" in draft
    assert "Wrote" in capsys.readouterr().out


def test_comet_provider_uses_its_own_default_model(monkeypatch, tmp_path):
    import lector.cli as cli
    from lector.pipeline import LectureDraftResult

    seen = {}
    monkeypatch.setattr(cli, "build_comet_llm_runner", lambda model, max_tokens: seen.setdefault("model", model))
    monkeypatch.setattr(cli, "generate_lecture_draft", lambda topic, *a, **k: LectureDraftResult(
        stage_reached=PipelineStage.DONE, topic=topic, annotated_draft="x"))

    main(["Topic", "--inbox", str(tmp_path), "--provider", "comet"])

    assert seen["model"] == "claude-sonnet-4-6"
