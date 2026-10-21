"""ESpeech-TTS-1 (Russian F5-TTS fine-tune) with ruaccent stress marks.
python e_espeech.py [checkpoint] [nfe_steps]   e.g. RL-V2 32"""
import sys

import numpy as np
import onnxruntime as ort

from common import DEVICE, REF_TEXT, REF_WAV, Bench, chunks

checkpoint = sys.argv[1] if len(sys.argv) > 1 else "RL-V2"
nfe = int(sys.argv[2]) if len(sys.argv) > 2 else 32
if REF_WAV is None or REF_TEXT is None:
    sys.exit("espeech needs --ref and --ref-text")
bench = Bench(f"espeech_{checkpoint}_nfe{nfe}")

# ruaccent's ONNX models expect token_type_ids, which current tokenizers
# no longer return - feed zeros rather than pin an old transformers.
_original_run = ort.InferenceSession.run


def _run(self, output_names, feed, *args, **kwargs):
    if "input_ids" in feed and "token_type_ids" not in feed \
            and any(i.name == "token_type_ids" for i in self.get_inputs()):
        feed = dict(feed, token_type_ids=np.zeros_like(feed["input_ids"]))
    return _original_run(self, output_names, feed, *args, **kwargs)


ort.InferenceSession.run = _run

from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text  # noqa: E402
from f5_tts.model import DiT  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from ruaccent import RUAccent  # noqa: E402

repo = f"ESpeech/ESpeech-TTS-1_{checkpoint}"
weights = {"RL-V2": "espeech_tts_rlv2.pt", "RL-V1": "espeech_tts_rlv1.pt", "SFT-256K": "espeech_tts_256k.pt"}[checkpoint]
model = load_model(DiT, dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                   hf_hub_download(repo, weights), vocab_file=hf_hub_download(repo, "vocab.txt"), device=DEVICE)
vocoder = load_vocoder(device=DEVICE)
accent = RUAccent()
accent.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
ref_audio, ref_text = preprocess_ref_audio_text(REF_WAV, accent.process_all(REF_TEXT))
bench.loaded()

pieces, sample_rate = [], None
for chunk in chunks():
    wav, sample_rate, _ = infer_process(ref_audio, ref_text, accent.process_all(chunk), model, vocoder,
                                        nfe_step=nfe, speed=1.0, cross_fade_duration=0.15, device=DEVICE)
    pieces.append(wav)
bench.done(pieces, sample_rate, clone=True)
