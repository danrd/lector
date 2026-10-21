"""Completeness check: transcribe each WAV with a local Whisper and compare
with the text it was synthesized from.
python asr.py <reference_text> <out.jsonl> <wav>... [--device cpu|cuda]

recall = share of the reference's words that came back, in order
(difflib matching blocks). Expect 0.75-0.85 even for a perfect reading:
Whisper writes spelled-out letters its own way ("F-каллиграфическое" for
"эф каллиграфическое"). Look for outliers, not for 1.0."""
import difflib
import json
import re
import sys

from faster_whisper import WhisperModel

args = [a for a in sys.argv[1:] if not a.startswith("--device")]
device = "cuda" if "--device=cuda" in sys.argv else "cpu"
reference_path, out_path, wavs = args[0], args[1], args[2:]
model = WhisperModel("large-v3-turbo", device=device, compute_type="float16" if device == "cuda" else "int8")


def words(text):
    return [w.replace("ё", "е") for w in re.findall(r"\w+", text.lower())]


reference = words(open(reference_path, encoding="utf-8").read())
with open(out_path, "w", encoding="utf-8") as out:
    for wav in wavs:
        segments, info = model.transcribe(wav, language="ru", beam_size=5)
        text = " ".join(s.text.strip() for s in segments)
        hypothesis = words(text)
        matcher = difflib.SequenceMatcher(a=reference, b=hypothesis, autojunk=False)
        matched = sum(block.size for block in matcher.get_matching_blocks())
        row = {"wav": wav, "recall": round(matched / len(reference), 3),
               "precision": round(matched / max(1, len(hypothesis)), 3), "text": text}
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{row['recall']:.2f}  {wav}", flush=True)
