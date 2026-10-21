"""Shared plumbing for the engine scripts: config from the environment
(set by run.py), chunking, timing, peak memory, WAV output and one
result line per run.

Engine scripts run inside their own virtualenvs (each engine pins its own
torch/transformers), so this module only uses numpy and soundfile, which
every engine's environment gets.
"""
import json
import os
import time

import numpy as np
import soundfile as sf

DEVICE = os.environ.get("BENCH_DEVICE", "cpu")          # cpu | cuda
OUT_DIR = os.environ["BENCH_OUT"]
TEXT = open(os.environ["BENCH_TEXT"], encoding="utf-8").read().strip()
REF_WAV = os.environ.get("BENCH_REF") or None            # 24 kHz mono wav, prepared by run.py
REF_TEXT = os.environ.get("BENCH_REF_TEXT") or None      # exact transcript of REF_WAV
CACHE_DIR = os.environ.get("BENCH_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"))


def chunks(max_chars=250, text=TEXT):
    """Lines (each ends in . ; :) packed up to max_chars - roughly the unit
    lector hands a backend in one call."""
    out, current = [], ""
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        if current and len(current) + 1 + len(line) > max_chars:
            out.append(current)
            current = line
        else:
            current = f"{current} {line}".strip()
    if current:
        out.append(current)
    return out


def _peak_ram_gb():
    try:
        import resource  # Unix only
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _peak_vram_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 3
    except ImportError:
        pass
    return None


class Bench:
    def __init__(self, name):
        self.name = name
        self.t0 = time.time()
        self.load_s = None

    def loaded(self):
        """Call once the model is ready: load time is reported apart from synthesis."""
        self.load_s = time.time() - self.t0
        self.t1 = time.time()

    def done(self, pieces, sample_rate, **extra):
        synth_s = time.time() - self.t1
        audio = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in pieces])
        os.makedirs(OUT_DIR, exist_ok=True)
        sf.write(os.path.join(OUT_DIR, f"{self.name}.wav"), audio, sample_rate)
        duration = len(audio) / sample_rate
        ram, vram = _peak_ram_gb(), _peak_vram_gb()
        result = {
            "name": self.name, "device": DEVICE, "load_s": round(self.load_s, 1),
            "synth_s": round(synth_s, 1), "audio_s": round(duration, 1),
            "rtf": round(synth_s / duration, 3),
            "peak_ram_gb": round(ram, 2) if ram is not None else None,
            "peak_vram_gb": round(vram, 2) if vram is not None else None,
            **extra,
        }
        print("RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
        with open(os.path.join(OUT_DIR, "results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
