"""Command-line entry point: keep an audio mirror of the vault up to date.

    lector-sync --vault ~/KB --out ~/KB-audio --folder Math --folder AI
    lector-sync --vault ~/KB --out ~/KB-audio --dry-run      # what would be voiced, and roughly how much

Voices each section the way lector-read voices a note (same --tts,
--voice, --voice-ref, --formulas options), but only sections that are new
or changed since the last run - see lector.sync for how. Every run that
voiced anything also writes <out>/review-<date>.md: the formula readings
the model produced, drifted ones first, to skim at leisure.

Runnable as `lector-sync` or `python -m lector.sync_cli`.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from lector.audio import synthesize_long_text
from lector.obsidian import DEFAULT_EXCLUDED_SECTIONS
from lector.read_cli import (
    FORMULAS_LLM,
    PIPER,
    add_voice_arguments,
    build_llm_runner,
    build_tts,
    cache_namespace,
    chunk_chars,
    validate_voice_args,
)
from lector.sync import Section, estimate_minutes, sync
from lector.tts_runtime import CachedTTSRunner
from lector.verbalize import VerbalizationResult, VerbalizeConfig, strip_notation_markup, verbalize_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lector-sync",
        description="Voice the vault section by section, re-voicing only what changed.",
    )
    parser.add_argument("--vault", required=True, help="Vault root.")
    parser.add_argument("--out", required=True, help="Audio folder (one sub-folder per note).")
    parser.add_argument("--folder", action="append", default=[],
                        help="Top-level vault folder to include; repeatable. Default: every content folder.")
    parser.add_argument("--note", action="append", default=[],
                        help="Vault-relative note to sync instead of whole folders; repeatable.")
    parser.add_argument("--level", type=int, default=3, choices=[2, 3, 4],
                        help="Deepest heading that starts its own track (default: 3, i.e. ## and ###).")
    parser.add_argument("--skip-section", action="append", default=[],
                        help="Also leave out sections with this heading (besides the sources lists).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report what would be voiced, moved and removed.")
    add_voice_arguments(parser)
    return parser


class SectionVoicer:
    """Formulas -> words -> speech -> tagged mp3, for one section at a time;
    collects every formula reading for the run's review file."""

    def __init__(self, args):
        self.args = args
        self.llm = build_llm_runner(args.llm_model) if args.formulas == FORMULAS_LLM else None
        tts = build_tts(args)
        self.tts = tts if args.no_cache else CachedTTSRunner(tts, args.cache_dir, cache_namespace(args))
        self.workers = args.workers if args.tts != PIPER else 1
        self.max_chunk_chars = chunk_chars(args)
        self.reviews: List[tuple] = []
        self._lock = threading.Lock()

    def __call__(self, section: Section, target: Path, metadata: Dict[str, str]) -> None:
        from llm_kit.prompt_builder import ApproxTokenizer

        from lector.ffmpeg import encode_audio

        if self.llm is not None:
            verbalization = verbalize_lines(section.lines, ApproxTokenizer(), self.llm, VerbalizeConfig())
        else:
            verbalization = VerbalizationResult(lines=[strip_notation_markup(line) for line in section.lines])
        if verbalization.rewritten or verbalization.failed:
            with self._lock:
                self.reviews.append((target, verbalization))

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "section.wav"
            synthesize_long_text("\n".join(verbalization.lines), self.tts, str(wav),
                                 max_chunk_chars=self.max_chunk_chars, max_workers=self.workers)
            partial = target.with_name(target.stem + ".partial" + target.suffix)
            encode_audio(wav, partial, metadata)
            partial.replace(target)

    def close(self) -> None:
        self.tts.close()

    def write_review(self, out_dir: Path) -> Optional[Path]:
        if not self.reviews:
            return None
        path = out_dir / f"review-{time.strftime('%Y-%m-%d')}.md"
        parts = [f"# Formula readings - {time.strftime('%Y-%m-%d %H:%M')}\n"]
        drifted = [(t, o, s) for t, v in self.reviews for o, s in v.drifted]
        failed = [(t, line) for t, v in self.reviews for line in v.failed]
        if drifted:
            parts.append("\n## Drifted - ordinary words changed too, check first\n")
            parts += [f"- {t.relative_to(out_dir)}\n  - `{o}`\n  - → {s}\n" for t, o, s in drifted]
        if failed:
            parts.append("\n## Failed - read with markup stripped only\n")
            parts += [f"- {t.relative_to(out_dir)}: `{line}`\n" for t, line in failed]
        parts.append("\n## All readings\n")
        for target, verbalization in self.reviews:
            parts.append(f"\n### {target.relative_to(out_dir)}\n")
            parts += [f"- `{o}`\n  → {s}\n" for o, s in verbalization.rewritten]
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(parts))
        return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_voice_args(args)
    vault, out_dir = Path(args.vault), Path(args.out)
    if not vault.is_dir():
        raise SystemExit(f"Vault not found: {vault}")
    exclude = [*DEFAULT_EXCLUDED_SECTIONS, *args.skip_section]
    namespace = f"{cache_namespace(args)}|{args.formulas}|{args.llm_model if args.formulas == FORMULAS_LLM else ''}"

    if args.dry_run:
        report = sync(vault, out_dir, namespace, voice_section=None, folders=args.folder, notes=args.note,
                      level=args.level, exclude=exclude, dry_run=True)
        print(f"\nDry run over {report.notes_seen} note(s): {report.voiced} section(s) to voice "
              f"({report.voiced_chars} chars, ~{estimate_minutes(report.voiced_chars):.0f} min of audio), "
              f"{report.moved} to move, {report.removed} to remove, {report.kept} unchanged.")
        return 0

    voicer = SectionVoicer(args)
    started = time.time()
    try:
        report = sync(vault, out_dir, namespace, voicer, folders=args.folder, notes=args.note,
                      level=args.level, exclude=exclude)
    finally:
        voicer.close()
        review = voicer.write_review(out_dir)

    print(f"\n{report.notes_seen} note(s): voiced {report.voiced} section(s) "
          f"(~{estimate_minutes(report.voiced_chars):.0f} min of audio) in {(time.time() - started) / 60:.1f} min, "
          f"moved {report.moved}, removed {report.removed}, {report.kept} unchanged.")
    if review:
        print(f"Formula readings: {review}")
    if report.failed:
        print(f"{len(report.failed)} section(s) failed and will be retried next run:", file=sys.stderr)
        for line in report.failed:
            print(f"  - {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
