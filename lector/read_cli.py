"""Command-line entry point for reading an existing vault note aloud:
`lector-read NOTE [--section ...]` -> an audio file of that note, read
verbatim in the language it's written in.

Unlike `lector` (lector/cli.py), nothing here writes new content: the
text is the note's own, cleaned of Obsidian syntax (lector.obsidian),
with only formulas and notation put into words by a model
(lector.verbalize). Every model rewrite is written to a review file
beside the audio, and --script-only stops before synthesis, so the
spoken form of the formulas can be checked - and the script edited, then
voiced with --script - before paying for any audio.

Runnable as `lector-read` (the console script) or `python -m lector.read_cli`.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from lector.audio import synthesize_long_text
from lector.knowledge_base import parse_note
from lector.obsidian import (
    DEFAULT_EXCLUDED_SECTIONS,
    SectionNotFoundError,
    clean_markdown,
    list_headings,
    select_sections,
)
from lector.tts_runtime import (
    COMET_BASE_URL,
    COMET_CLONING_MODEL,
    COMET_MAX_CHUNK_CHARS,
    COMET_OPENAI_MODELS,
    BaseTTSRunner,
    CachedTTSRunner,
    SynthesisConfig,
)
from lector.verbalize import VerbalizationResult, VerbalizeConfig, strip_notation_markup, verbalize_lines

COMET = "comet"
PIPER = "piper"
OPENAI = "openai"

FORMULAS_LLM = "llm"
FORMULAS_RAW = "raw"

DEFAULT_COMET_MODEL = "eleven_v3"
DEFAULT_COMET_VOICE = "Bernard"
DEFAULT_LLM_MODEL = "claude-sonnet-4-6"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lector" / "tts"


def add_voice_arguments(parser: argparse.ArgumentParser) -> None:
    """Formula and synthesis options - shared with lector-sync, which voices
    the same way, just many sections at a time."""
    parser.add_argument("--formulas", choices=[FORMULAS_LLM, FORMULAS_RAW], default=FORMULAS_LLM,
                        help="llm (default): a model reads formulas/notation into words; "
                             "raw: drop the $/` markup only, no model call.")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,
                        help=f"CometAPI chat model for formulas (default: {DEFAULT_LLM_MODEL}).")

    parser.add_argument("--tts", choices=[COMET, PIPER, OPENAI], default=COMET,
                        help="Synthesis backend (default: comet - CometAPI).")
    parser.add_argument("--model", default=None,
                        help=f"CometAPI speech model (default: {DEFAULT_COMET_MODEL}, or "
                             f"{COMET_CLONING_MODEL} with --voice-ref).")
    parser.add_argument("--voice", default=None,
                        help=f"Preset voice name (CometAPI ElevenLabs default: {DEFAULT_COMET_VOICE}; "
                             "OpenAI: alloy etc.), or the .onnx model path for piper.")
    parser.add_argument("--voice-ref", default=None,
                        help="A 15-30 s clean recording of the voice to clone (CometAPI seed_audio).")
    parser.add_argument("--language", default=None,
                        help="Force the speech language code (ru, en, ...). Default: detected per chunk.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking-rate multiplier (default: 1.0).")
    parser.add_argument("--workers", type=int, default=6,
                        help="Chunks synthesized in parallel on hosted backends (default: 6).")
    parser.add_argument("--max-chunk-chars", type=int, default=None,
                        help="Longest text chunk per synthesis call (default: per model).")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                        help="Where synthesized chunks are cached, so re-runs only pay for changed text.")
    parser.add_argument("--no-cache", action="store_true", help="Don't read or write the chunk cache.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lector-read",
        description="Read an Obsidian note (or chosen sections of it) aloud, verbatim.",
    )
    parser.add_argument("note", nargs="?", default=None,
                        help="The note to read: a path, or a path relative to --vault.")
    parser.add_argument("--vault", default=None, help="Vault root that NOTE is relative to.")
    parser.add_argument("--section", action="append", default=[],
                        help="Heading to read (with its subsections); repeatable. Case, emoji and "
                             "punctuation don't matter, and a partial name matches every heading "
                             "containing it. Omitted: the whole note.")
    parser.add_argument("--keep-sources", action="store_true",
                        help="Also read the per-topic link lists (" +
                             ", ".join(DEFAULT_EXCLUDED_SECTIONS) + "), skipped by default.")
    parser.add_argument("--list-sections", action="store_true",
                        help="Print the note's headings and exit.")
    parser.add_argument("--script", default=None,
                        help="Voice this (possibly hand-edited) script .txt instead of a note.")
    parser.add_argument("--output", default=None,
                        help="Audio file to write; format by extension (.mp3, .m4a, .wav). "
                             "Default: <note>[-<section>].mp3 in the current directory.")
    parser.add_argument("--script-only", action="store_true",
                        help="Prepare and write the script (and formula review) without synthesizing.")

    add_voice_arguments(parser)
    return parser


def validate_args(args) -> None:
    if (args.note is None) == (args.script is None):
        raise SystemExit("Give exactly one of NOTE or --script.")
    if args.script and (args.section or args.list_sections or args.script_only):
        raise SystemExit("--section/--list-sections/--script-only apply to a note, not to --script.")
    validate_voice_args(args)


def validate_voice_args(args) -> None:
    if args.voice_ref:
        if args.tts != COMET:
            raise SystemExit("--voice-ref (voice cloning) needs --tts comet.")
        if args.model not in (None, COMET_CLONING_MODEL):
            raise SystemExit(f"--voice-ref needs --model {COMET_CLONING_MODEL}; {args.model} has preset voices only.")
        if not Path(args.voice_ref).is_file():
            raise SystemExit(f"Reference recording not found: {args.voice_ref}")
    if args.tts == PIPER and not args.voice:
        raise SystemExit("--voice is required for piper: the path to a .onnx voice model.")


def resolve_note(note: str, vault: Optional[str]) -> Path:
    candidates = [Path(note)]
    if vault:
        candidates.insert(0, Path(vault) / note)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.suffix != ".md" and candidate.with_suffix(".md").is_file():
            return candidate.with_suffix(".md")
    raise SystemExit(f"Note not found: {note}" + (f" (looked in {vault} too)" if vault else ""))


def default_output(stem: str, sections: Sequence[str]) -> Path:
    if sections:
        slug = "-".join("_".join(section.split()) for section in sections)
        stem = f"{stem}-{slug}"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
    return Path(f"{safe}.mp3")


class CometChatRunner:
    """The one-method llm_kit Runner interface (generate(prompt) -> str)
    over CometAPI's OpenAI-compatible chat endpoint.

    Not llm_kit's OpenRouterRunner, deliberately: that always sends
    temperature/top_p, and through CometAPI's Bedrock route Claude models
    then end with finish_reason "length" and an empty answer (Sonnet 4.6),
    or refuse the request outright (Opus 4.8: "temperature is
    deprecated"). Only max_tokens is sent here."""

    def __init__(self, model: str, client, max_tokens: int = 4096):
        self.model = model
        self.client = client
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


def build_llm_runner(model: str) -> CometChatRunner:
    """A chat model through CometAPI, so the same key that pays for speech
    pays for this too."""
    from lector.tts_runtime import comet_api_key

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Reading formulas with a model needs the openai package: "
                         "uv sync --extra comet (or pass --formulas raw).") from exc
    try:
        api_key = comet_api_key()
    except ValueError as exc:
        raise SystemExit(f"{exc} - see .env.example, or pass --formulas raw.") from exc
    client = OpenAI(base_url=f"{COMET_BASE_URL}/v1", api_key=api_key, timeout=180.0, max_retries=2)
    return CometChatRunner(model, client)


def comet_model(args) -> str:
    return args.model or (COMET_CLONING_MODEL if args.voice_ref else DEFAULT_COMET_MODEL)


def build_tts(args) -> BaseTTSRunner:
    config = SynthesisConfig(speed=args.speed)
    try:
        if args.tts == COMET:
            from lector.tts_runtime import build_comet_runner

            model = comet_model(args)
            voice = args.voice
            if voice is None and model not in COMET_OPENAI_MODELS and model != COMET_CLONING_MODEL:
                voice = DEFAULT_COMET_VOICE
            return build_comet_runner(model=model, voice=voice, reference_audio=args.voice_ref,
                                      language=args.language, config=config)
        if args.tts == OPENAI:
            from lector.tts_runtime import OpenAITTSRunner

            return OpenAITTSRunner(voice=args.voice or "alloy", config=config)
        from lector.tts_runtime import build_piper_runner

        return build_piper_runner(args.voice, config=config)
    except ImportError as exc:
        extra = {COMET: "comet", OPENAI: "tts-cloud", PIPER: "tts-local"}[args.tts]
        raise SystemExit(f"The {args.tts} backend needs an optional dependency that isn't installed "
                         f"({exc}).\nInstall it with: uv sync --extra {extra}") from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def cache_namespace(args) -> str:
    """Everything that changes how a given text sounds - a different
    voice must never be served from another voice's cached chunk."""
    reference = ""
    if args.voice_ref:
        reference = hashlib.sha256(Path(args.voice_ref).read_bytes()).hexdigest()
    model = comet_model(args) if args.tts == COMET else ""
    return "|".join([args.tts, model, args.voice or "", reference, args.language or "", f"{args.speed:g}"])


def chunk_chars(args) -> int:
    if args.max_chunk_chars:
        return args.max_chunk_chars
    if args.tts == COMET:
        return COMET_MAX_CHUNK_CHARS.get(comet_model(args), 1000)
    return 1000


def prepare_script(args, note_path: Path) -> Optional[tuple]:
    """Note -> (cleaned note, verbalization). None when --list-sections
    already did everything that was asked."""
    note = parse_note(note_path)

    if args.list_sections:
        for heading in list_headings(note.body):
            print(f"{'  ' * (heading.level - 1)}{heading.text}")
        return None

    exclude = () if args.keep_sources else DEFAULT_EXCLUDED_SECTIONS
    try:
        body = select_sections(note.body, args.section, exclude=exclude)
    except SectionNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    cleaned = clean_markdown(body)
    if not cleaned.lines:
        raise SystemExit("Nothing left to read in the selected part of the note.")

    if args.formulas == FORMULAS_LLM:
        from llm_kit.prompt_builder import ApproxTokenizer

        verbalization = verbalize_lines(cleaned.lines, ApproxTokenizer(), build_llm_runner(args.llm_model),
                                        VerbalizeConfig())
    else:
        verbalization = VerbalizationResult(lines=[strip_notation_markup(line) for line in cleaned.lines])
    return cleaned, verbalization


def write_review(path: Path, verbalization: VerbalizationResult) -> None:
    """Every line the model changed, original beside spoken form - the
    thing to skim before trusting the formulas, worst cases first."""
    parts: List[str] = ["# Formula readings to review\n"]
    sections = [
        ("Drifted - the model changed ordinary words too, check these first", verbalization.drifted),
        ("Rewritten", [pair for pair in verbalization.rewritten if pair not in verbalization.drifted]),
    ]
    for title, pairs in sections:
        if pairs:
            parts.append(f"\n## {title}\n")
            for original, spoken in pairs:
                parts.append(f"- `{original}`\n  → {spoken}\n")
    if verbalization.failed:
        parts.append("\n## Failed - the model returned nothing usable; read with markup stripped only\n")
        parts.extend(f"- `{line}`\n" for line in verbalization.failed)
    path.write_text("".join(parts), encoding="utf-8")


def _report_preparation(cleaned, verbalization: VerbalizationResult, script_path: Path,
                         review_path: Optional[Path]) -> None:
    print(f"Wrote {script_path} ({len(verbalization.lines)} lines, "
          f"{sum(len(line) for line in verbalization.lines)} characters)")
    if review_path is not None:
        print(f"Wrote {review_path} ({len(verbalization.rewritten)} line(s) with formulas read into words)")
    if cleaned.skipped_code_blocks or cleaned.skipped_tables:
        print(f"Skipped {cleaned.skipped_code_blocks} code block(s) and {cleaned.skipped_tables} table(s) "
              "- not read aloud.")
    if verbalization.drifted:
        print(f"Warning: {len(verbalization.drifted)} line(s) changed more than their formulas - "
              f"see {review_path}.", file=sys.stderr)
    if verbalization.failed:
        reasons = "; ".join(sorted(set(verbalization.errors))) or "unknown"
        print(f"Warning: {len(verbalization.failed)} line(s) couldn't be verbalized and keep raw notation "
              f"({reasons}).", file=sys.stderr)
    if verbalization.has_leftover_notation:
        print("Warning: some notation ($, `, \\command) is still in the script.", file=sys.stderr)


def synthesize(args, script: str, output: Path) -> None:
    tts = build_tts(args)
    if not args.no_cache:
        tts = CachedTTSRunner(tts, args.cache_dir, cache_namespace(args))
    # In-process Piper isn't meant to be called from several threads.
    workers = args.workers if args.tts != PIPER else 1

    with tts, tempfile.TemporaryDirectory() as tmp:
        wav_path = output if output.suffix.lower() == ".wav" else Path(tmp) / "lecture.wav"
        result = synthesize_long_text(script, tts, str(wav_path),
                                      max_chunk_chars=chunk_chars(args), max_workers=workers)
        if wav_path != output:
            from lector.ffmpeg import encode_audio

            encode_audio(wav_path, output)

    cached = f", {tts.hits} from cache" if isinstance(tts, CachedTTSRunner) and tts.hits else ""
    minutes, seconds = divmod(round(result.duration_seconds), 60)
    print(f"Wrote {output} ({minutes}:{seconds:02d}, {result.num_chunks} chunk(s){cached})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)

    if args.script:
        script_path = Path(args.script)
        if not script_path.is_file():
            raise SystemExit(f"Script not found: {args.script}")
        script = script_path.read_text(encoding="utf-8")
        output = Path(args.output) if args.output else script_path.with_suffix(".mp3")
    else:
        note_path = resolve_note(args.note, args.vault)
        prepared = prepare_script(args, note_path)
        if prepared is None:
            return 0
        cleaned, verbalization = prepared
        output = Path(args.output) if args.output else default_output(note_path.stem, args.section)
        output.parent.mkdir(parents=True, exist_ok=True)

        script = "\n".join(verbalization.lines)
        script_path = output.with_suffix(".txt")
        script_path.write_text(script + "\n", encoding="utf-8")
        review_path = None
        if args.formulas == FORMULAS_LLM:
            review_path = output.with_name(f"{output.stem}.formulas.md")
            write_review(review_path, verbalization)
        _report_preparation(cleaned, verbalization, script_path, review_path)

        if args.script_only:
            print(f"Script only - edit it if needed, then voice it with: lector-read --script {script_path}")
            return 0

    if not script.strip():
        raise SystemExit("The script is empty - nothing to synthesize.")
    synthesize(args, script, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
