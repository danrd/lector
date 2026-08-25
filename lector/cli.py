"""Command-line entry point: a topic in, a narrated WAV out.

Everything here is composition, not logic - loading the vault, picking a
TTS backend, and reporting how far the run got. The pipeline itself
(lector.pipeline.generate_lecture_audio) is unchanged and stays the only
place that knows the stage order.

Runnable as `lector` (the console script) or `python -m lector`.

A vault is optional on purpose: knowledge_base.py treats "this vault says
nothing about the topic" as a normal outcome rather than a failure, so
running with no vault at all is just the extreme of that - the lecture
comes from the model's own knowledge, and verification flags it as such.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from llm_kit.prompt_builder import ApproxTokenizer

from lector.knowledge_base import Note, load_vault
from lector.lecture import LectureConfig
from lector.narration import NarrationConfig
from lector.pipeline import PipelineConfig, PipelineStage, generate_lecture_audio
from lector.tts_runtime import BaseTTSRunner, SynthesisConfig

PIPER = "piper"
OPENAI = "openai"
ORPHEUS = "orpheus"


def build_llm_runner(model: str, max_tokens: int):
    """The hosted runner, the one backend that needs no local weights -
    which is what a CLI's default should be able to assume. Local
    backends are reachable through llm_kit.llm_setup.build_runner and
    are deliberately not wired to a flag here: choosing one means
    choosing a GGUF/HF checkpoint, quantization and device too, which is
    a config file's job rather than a command line's.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set - the CLI generates through OpenRouter.\n"
            "Set it (see .env.example), or use lector.pipeline.generate_lecture_audio "
            "directly with a local llm_kit runner."
        )
    from llm_kit.llm_runtime import GenerationConfig, OpenRouterRunner

    generation_kwargs = GenerationConfig(max_tokens=max_tokens).to_chat_completions(seed=42)
    try:
        return OpenRouterRunner(models=[model], generation_kwargs=generation_kwargs, api_key=api_key)
    except ImportError as exc:
        # OpenRouterRunner imports `openai` lazily, so a missing extra
        # surfaces here as a bare ModuleNotFoundError naming a package the
        # user never asked for by name.
        raise SystemExit(
            f"The OpenRouter backend needs an optional dependency that isn't installed ({exc}).\n"
            "Install it with: uv sync --extra openrouter"
        ) from exc


# Which extra to install, per backend, for the "you're missing a
# dependency" message - a bare ModuleNotFoundError names a package the
# user never chose by name (piper, snac, openai), not the extra they'd
# actually type.
_TTS_EXTRAS = {PIPER: "tts-local", OPENAI: "tts-cloud", ORPHEUS: "tts-orpheus"}


def validate_args(args) -> None:
    """Check what can be checked from the arguments alone, before any
    backend is constructed. Two backends need a local model file, and
    finding that out only after the LLM runner has already been built
    (and possibly failed on its own missing extra) reports the wrong
    problem entirely."""
    if args.tts in (PIPER, ORPHEUS) and not args.voice:
        what = (".onnx voice model (its matching .onnx.json must sit alongside it)"
                if args.tts == PIPER else "GGUF model")
        raise SystemExit(f"--voice is required for the {args.tts} backend: the path to a {what}.")


def build_tts_runner(backend: str, voice: Optional[str], speed: float) -> BaseTTSRunner:
    """Construct the requested TTS backend, importing only that one -
    all three are optional extras, and asking for Piper must not require
    openai or snac to be installed."""
    config = SynthesisConfig(speed=speed)

    try:
        if backend == PIPER:
            from lector.tts_runtime import build_piper_runner

            return build_piper_runner(voice, config=config)

        if backend == OPENAI:
            from lector.tts_runtime import OpenAITTSRunner

            return OpenAITTSRunner(voice=voice or "alloy", config=config)

        from lector.tts_runtime import build_orpheus_runner

        return build_orpheus_runner(voice, config=config)
    except ImportError as exc:
        raise SystemExit(
            f"The {backend} backend needs an optional dependency that isn't installed ({exc}).\n"
            f"Install it with: uv sync --extra {_TTS_EXTRAS[backend]}"
        ) from exc


def load_notes(vault: Optional[str]) -> List[Note]:
    if vault is None:
        return []
    path = Path(vault)
    if not path.exists():
        raise SystemExit(f"Vault not found: {vault}")
    return load_vault(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lector",
        description="Generate a narrated lecture on a topic, grounded in an Obsidian vault.",
    )
    parser.add_argument("topic", help="What the lecture should be about.")
    parser.add_argument("--vault", default=None,
                        help="Path to an Obsidian vault. Omitted: the lecture draws on the "
                             "model's own knowledge only, and verification flags it as such.")
    parser.add_argument("--output", default="lecture.wav",
                        help="Where to write the WAV (default: lecture.wav). The narration "
                             "script is written alongside it as .txt.")
    parser.add_argument("--tts", choices=[PIPER, OPENAI, ORPHEUS], default=PIPER,
                        help="Synthesis backend (default: piper, local).")
    parser.add_argument("--voice", default=None,
                        help="Piper/Orpheus: path to the model file. OpenAI: voice name "
                             "(default: alloy).")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speaking-rate multiplier (default: 1.0).")
    parser.add_argument("--model", default="openai/gpt-4o-mini",
                        help="OpenRouter model used for drafting/verification/narration.")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Generation cap per LLM call (default: 4096).")
    parser.add_argument("--language", default=None,
                        help="Language for both the draft and the narration.")
    parser.add_argument("--max-chunk-chars", type=int, default=1000,
                        help="Longest text chunk handed to the TTS backend in one call.")
    return parser


def _report(result, output: str) -> int:
    """Print what actually happened and return the process exit code.
    A run that stopped early is reported as the stage it stopped at -
    the pipeline's own soft-failure contract - not as a generic error."""
    if result.stage_reached == PipelineStage.DRAFTING:
        print(f"Stopped at drafting: the prompt for '{result.topic}' didn't fit its token limit.",
              file=sys.stderr)
        return 1

    if result.stage_reached == PipelineStage.NARRATION:
        print("Stopped at narration: no usable script was produced "
              "(the rewrite prompt didn't fit, or the model returned nothing).", file=sys.stderr)
        return 1

    script_path = Path(output).with_suffix(".txt")
    script_path.write_text(result.narration.script, encoding="utf-8")

    print(f"Wrote {output} ({result.audio.duration_seconds:.1f}s, "
          f"{result.audio.num_chunks} chunk(s))")
    print(f"Wrote {script_path}")

    checked = len(result.verification_results)
    if checked:
        print(f"Verified {checked} flagged claim(s).")
    if result.narration.excluded_claims:
        print(f"{len(result.narration.excluded_claims)} claim(s) stayed unresolved and were "
              "left out of the narration:", file=sys.stderr)
        for claim in result.narration.excluded_claims:
            print(f"  - {claim}", file=sys.stderr)
    if result.narration.has_leftover_markdown:
        print("Warning: the script still contains markdown or a review marker - "
              "the rewrite didn't fully clean it up.", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)

    notes = load_notes(args.vault)
    if args.vault is not None:
        print(f"Loaded {len(notes)} note(s) from {args.vault}")

    config = PipelineConfig(
        lecture=LectureConfig(language=args.language),
        narration=NarrationConfig(language=args.language),
        max_chunk_chars=args.max_chunk_chars,
    )

    runner = build_llm_runner(args.model, args.max_tokens)
    with build_tts_runner(args.tts, args.voice, args.speed) as tts_runner:
        result = generate_lecture_audio(
            args.topic, notes, ApproxTokenizer(), runner, tts_runner, args.output, config,
        )

    return _report(result, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
