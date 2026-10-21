"""Qwen3-TTS through the official qwen-tts package (PyTorch).
python e_qwen.py <0.6B|1.7B> <clone|xvec>
clone = full in-context cloning (needs the reference transcript); xvec = speaker embedding only."""
import sys

import torch

from common import DEVICE, REF_TEXT, REF_WAV, Bench, chunks

size, mode = sys.argv[1], sys.argv[2]
if REF_WAV is None or (mode == "clone" and REF_TEXT is None):
    sys.exit("qwen needs --ref (and --ref-text for clone mode)")
bench = Bench(f"qwen3_{size}_{mode}")

from qwen_tts import Qwen3TTSModel  # noqa: E402 - after argument checks, it's slow to import

cuda = DEVICE == "cuda"
model = Qwen3TTSModel.from_pretrained(
    f"Qwen/Qwen3-TTS-12Hz-{size}-Base",
    device_map="cuda:0" if cuda else "cpu",
    dtype=torch.bfloat16 if cuda else torch.float32,
    attn_implementation="sdpa",  # flash-attn isn't practical on Windows
)
prompt = model.create_voice_clone_prompt(
    ref_audio=REF_WAV, ref_text=REF_TEXT if mode == "clone" else None, x_vector_only_mode=(mode == "xvec"),
)
bench.loaded()

pieces, sample_rate = [], None
for chunk in chunks():
    wavs, sample_rate = model.generate_voice_clone(text=chunk, language="Russian", voice_clone_prompt=prompt)
    pieces.append(wavs[0])
bench.done(pieces, sample_rate, clone=True)
