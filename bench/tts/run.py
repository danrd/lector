"""Benchmark local TTS engines on one Russian fragment, each in its own
virtualenv, then score completeness with a local Whisper.

    python bench/tts/run.py --device cuda --ref me.mp3 --ref-text "Exact words of the clip."

Needs uv and ffmpeg on PATH; works on Windows, Linux and macOS (the C
runtime engine - qwenc - only where that runtime builds: Linux/macOS/WSL2).
First run downloads a few GB of packages and weights per engine.

The reference recording stays outside the repo: pass its path. A clean
10-12 s clip with an exact transcript is what the cloning engines want;
--ref-seconds trims a longer one (cut at a sentence boundary, and give
the transcript of just that part).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TORCH = ["torch", "torchaudio"]
BASE = ["numpy", "soundfile"]

# name -> (virtualenv, packages, script + args)
ENGINES = {
    "silero":       ("silero",  TORCH,                          ["e_silero.py", "v5_6_ru", "eugene"]),
    "qwen06-xvec":  ("qwen",    TORCH + ["qwen-tts"],           ["e_qwen.py", "0.6B", "xvec"]),
    "qwen06-clone": ("qwen",    TORCH + ["qwen-tts"],           ["e_qwen.py", "0.6B", "clone"]),
    "qwen17-xvec":  ("qwen",    TORCH + ["qwen-tts"],           ["e_qwen.py", "1.7B", "xvec"]),
    "qwen17-clone": ("qwen",    TORCH + ["qwen-tts"],           ["e_qwen.py", "1.7B", "clone"]),
    "espeech":      ("espeech", TORCH + ["f5-tts", "ruaccent"], ["e_espeech.py", "RL-V2", "32"]),
}
DEFAULT_ENGINES = ["silero", "qwen06-xvec", "qwen17-clone", "espeech"]


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_venv(name: str, packages: list, torch_backend: str) -> Path:
    """One virtualenv per engine (their torch/transformers pins conflict),
    created and filled on first use, reused after."""
    venv = HERE / ".venvs" / name
    marker = venv / "bench-packages.json"
    wanted = sorted(set(packages + BASE))
    if marker.exists() and json.loads(marker.read_text()) == {"packages": wanted, "torch": torch_backend}:
        return venv_python(venv)
    if not venv.exists():
        subprocess.run(["uv", "venv", "-q", "-p", "3.11", str(venv)], check=True)
    print(f"Installing {name}: {' '.join(wanted)} (torch backend: {torch_backend})", flush=True)
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(venv_python(venv)),
                    f"--torch-backend={torch_backend}", *wanted], check=True)
    marker.write_text(json.dumps({"packages": wanted, "torch": torch_backend}))
    return venv_python(venv)


def prepare_reference(ref: str, seconds: float | None, out_dir: Path) -> Path:
    """Any audio file -> the 24 kHz mono WAV every engine accepts."""
    target = out_dir / "ref.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", ref, "-ac", "1", "-ar", "24000"]
    if seconds:
        cmd += ["-t", str(seconds)]
    subprocess.run([*cmd, str(target)], check=True)
    return target


def summarize(out_dir: Path) -> str:
    results = [json.loads(line) for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    asr = {}
    if (out_dir / "asr.jsonl").exists():
        for line in (out_dir / "asr.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            asr[Path(row["wav"]).stem] = row
    rows = ["| engine | device | RTF | × realtime | load, s | RAM, GB | VRAM, GB | recall |",
            "|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: r["rtf"]):
        recall = asr.get(r["name"], {}).get("recall", "")
        rows.append(f"| {r['name']} | {r['device']} | {r['rtf']:.2f} | {1 / r['rtf']:.1f} | {r['load_s']:.0f} "
                    f"| {r['peak_ram_gb'] or '-'} | {r['peak_vram_gb'] or '-'} | {recall} |")
    table = "\n".join(rows)
    (out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--engines", default=",".join(DEFAULT_ENGINES),
                        help=f"Comma-separated, from: {', '.join(ENGINES)}, qwenc (default: %(default)s)")
    parser.add_argument("--ref", help="Reference recording of the voice to clone (any format ffmpeg reads).")
    parser.add_argument("--ref-text", help="Exact transcript of the (trimmed) reference.")
    parser.add_argument("--ref-seconds", type=float, default=None, help="Trim the reference to this length.")
    parser.add_argument("--text", default=str(HERE / "text_ru.txt"), help="Fragment to synthesize.")
    parser.add_argument("--out", default=None, help="Output folder (default: out/<device>-<time>).")
    parser.add_argument("--qwenc-exe", help="qwen3-tts C runtime binary, for the qwenc engine.")
    parser.add_argument("--qwenc-model", help="Model dir for qwenc (e.g. qwen3-tts-0.6b-base).")
    parser.add_argument("--qwenc-mode", default="int8 xvec",
                        help="'<int8|bf16> <clone|xvec>' for qwenc (default: %(default)s)")
    parser.add_argument("--qwenc-extra", default="",
                        help="Extra runtime flags for qwenc, e.g. '--backend cuda'.")
    parser.add_argument("--asr-device", choices=["cpu", "cuda"], default="cpu",
                        help="Whisper device for scoring (cpu avoids cuDNN DLL trouble on Windows).")
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-engine time limit, seconds.")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else HERE / "out" / f"{args.device}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_backend = "auto" if args.device == "cuda" else "cpu"
    env = {**os.environ, "BENCH_DEVICE": args.device, "BENCH_OUT": str(out_dir), "BENCH_TEXT": args.text,
           "BENCH_CACHE": str(HERE / ".cache"), "PYTHONIOENCODING": "utf-8"}
    if args.ref:
        env["BENCH_REF"] = str(prepare_reference(args.ref, args.ref_seconds, out_dir))
    if args.ref_text:
        env["BENCH_REF_TEXT"] = args.ref_text

    for name in [e.strip() for e in args.engines.split(",") if e.strip()]:
        if name == "qwenc":
            if not (args.qwenc_exe and args.qwenc_model):
                print("qwenc: skipped - needs --qwenc-exe and --qwenc-model", flush=True)
                continue
            python = ensure_venv("base", [], "cpu")
            quant, mode = args.qwenc_mode.split()
            command = [str(python), "e_qwenc.py", args.qwenc_exe, args.qwenc_model, quant, mode,
                       *shlex.split(args.qwenc_extra)]
        elif name in ENGINES:
            venv, packages, script = ENGINES[name]
            python = ensure_venv(venv, packages, torch_backend)
            command = [str(python), *script]
        else:
            print(f"{name}: unknown engine", flush=True)
            continue

        print(f"=== {name}", flush=True)
        log = out_dir / f"log-{name}.txt"
        with open(log, "w", encoding="utf-8") as log_file:
            try:
                code = subprocess.run(command, cwd=HERE, env=env, stdout=log_file, stderr=subprocess.STDOUT,
                                      timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        result = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                  if line.startswith("RESULT ")]
        print(result[-1] if result else f"{name}: failed ({code}) - see {log}", flush=True)

    if not (out_dir / "results.jsonl").exists():
        print("No engine produced audio.")
        return 1
    if not args.no_score:
        python = ensure_venv("asr", ["faster-whisper"], "cpu")
        wavs = sorted(str(p) for p in out_dir.glob("*.wav") if p.name != "ref.wav")
        subprocess.run([str(python), "asr.py", args.text, str(out_dir / "asr.jsonl"), *wavs,
                        f"--device={args.asr_device}"], cwd=HERE, env=env, check=False)
    print(summarize(out_dir))
    print(f"\nAudio, logs and summary.md in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
