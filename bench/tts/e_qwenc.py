"""Qwen3-TTS through the pure-C runtime (github.com/gabriele-mastrapasqua/qwen3-tts),
one CLI call per chunk. Linux/macOS/WSL2 only - it has no native Windows build.
python e_qwenc.py <path/to/qwen_tts> <model_dir> <int8|bf16> <clone|xvec> [--backend cuda]"""
import os
import subprocess
import sys
import tempfile

import soundfile as sf

from common import REF_TEXT, REF_WAV, Bench, chunks

exe, model_dir, quant, mode = sys.argv[1:5]
extra = sys.argv[5:]
bench = Bench(f"qwenc_{os.path.basename(model_dir.rstrip('/'))}_{quant}_{mode}")
bench.loaded()

pieces, sample_rate = [], None
with tempfile.TemporaryDirectory() as tmp:
    for index, chunk in enumerate(chunks()):
        out = os.path.join(tmp, f"{index}.wav")
        cmd = [exe, "-d", model_dir, "-l", "Russian", "--ref-audio", REF_WAV, "--text", chunk,
               "-o", out, "--seed", "42", "--silent", "-j", str(os.cpu_count() or 4), *extra]
        if quant != "bf16":
            cmd.append(f"--{quant}")
        if mode == "xvec":
            cmd.append("--xvector-only")
        else:
            # Without a transcript the runtime silently falls back to x-vector only.
            cmd += ["--ref-text", REF_TEXT]
        subprocess.run(cmd, check=True)
        data, sample_rate = sf.read(out)
        pieces.append(data)
bench.done(pieces, sample_rate, clone=True, runtime="qwen3-tts C (model reloaded per chunk)")
