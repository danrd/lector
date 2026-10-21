"""Silero v5.x Russian (no cloning): python e_silero.py [model_id] [speaker]."""
import os
import sys
import urllib.request

import torch

from common import CACHE_DIR, DEVICE, Bench, chunks

model_id = sys.argv[1] if len(sys.argv) > 1 else "v5_6_ru"
speaker = sys.argv[2] if len(sys.argv) > 2 else "eugene"
bench = Bench(f"silero_{model_id}_{speaker}")

# Loaded as a torch.package directly: torch.hub needs github.com, which
# isn't always reachable (it wasn't from the cloud sandbox).
path = os.path.join(CACHE_DIR, f"silero_{model_id}.pt")
if not os.path.exists(path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    urllib.request.urlretrieve(f"https://models.silero.ai/models/tts/ru/{model_id}.pt", path + ".part")
    os.replace(path + ".part", path)
model = torch.package.PackageImporter(path).load_pickle("tts_models", "model")
model.to(torch.device(DEVICE))
bench.loaded()

sample_rate = 48000
pieces = [model.apply_tts(text=chunk, speaker=speaker, sample_rate=sample_rate, put_accent=True, put_yo=True).cpu()
          for chunk in chunks()]
bench.done(pieces, sample_rate, clone=False)
