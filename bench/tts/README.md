# Local TTS benchmark

Which open model can voice the vault locally - Russian, ideally in a
cloned voice - and how fast on given hardware. One fixed fragment
(`text_ru.txt`: the "Аксиоматика Колмогорова" lines from
`Math/02_Probability`, already put through lector's formula
verbalization), each engine in its own virtualenv, completeness checked
with a local Whisper.

```bash
# needs uv and ffmpeg on PATH; run.py itself is stdlib-only
python bench/tts/run.py --device cuda \
    --ref path/to/my_voice.mp3 --ref-seconds 10.45 \
    --ref-text "Exact words of the first 10.45 seconds."
```

`--engines` picks from `silero`, `qwen06-xvec`, `qwen06-clone`,
`qwen17-xvec`, `qwen17-clone`, `espeech`, `qwenc` (default: the four
that sounded acceptable - see below). Each run writes the WAVs, per-engine
logs and `summary.md` to `out/<device>-<time>/`. The reference recording
is passed by path and only copied into that (git-ignored) folder.

First run per engine installs its packages (`--torch-backend=auto` picks
the CUDA build of torch for `--device cuda`) and downloads weights - a few
GB each; later runs reuse both.

## Engines

| engine | voice | notes |
|---|---|---|
| `silero` | preset (`eugene`; also `xenia`, `aidar`, `baya`, `kseniya`) | Russian-only, auto stress, very fast on CPU. Non-commercial licence. |
| `qwen06-*`, `qwen17-*` | cloned | Qwen3-TTS Base via the official `qwen-tts` (PyTorch). `xvec` = speaker embedding only, no transcript needed; `clone` = full in-context cloning, needs `--ref-text`. Apache-2.0. |
| `espeech` | cloned | ESpeech-TTS-1 RL-V2, a Russian F5-TTS fine-tune, with `ruaccent` stress marks (patched for current tokenizers in `e_espeech.py`). Diffusion - GPU only in practice. |
| `qwenc` | cloned | Same Qwen3-TTS through the pure-C runtime ([gabriele-mastrapasqua/qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts)), int8. Build it yourself and pass `--qwenc-exe/--qwenc-model`; Linux/macOS/WSL2 only. Its `--serve` mode exposes an OpenAI-compatible `/v1/audio/speech`, so lector can use it through `OpenAITTSRunner(base_url=...)`. Without `--ref-text` it silently falls back to x-vector cloning. |

Dropped after listening: Piper (mispronounces - "Ассиоматика Камагорова"),
XTTS-v2 (substitutes words - "отъезду" for "отрезок"), Chatterbox
Multilingual.

## Results: 4 vCPU (Xeon 2.8 GHz), 15 GB RAM, no GPU - 2026-10-21

RTF = synthesis time / audio length (lower is faster; < 1 is faster
than real time). Recall = share of the fragment's words Whisper hears;
0.75-0.85 is what a complete reading scores, because Whisper spells
letter names its own way ("F-каллиграфическое").

| engine | clone | RTF | peak RAM | recall | heard as |
|---|---|---|---|---|---|
| piper (irina / dmitri) | – | 0.05–0.07 | 0.5 GB | 0.64–0.73 | mispronounces |
| silero v5_6_ru (eugene / xenia / aidar) | – | 0.08–0.13 | 0.7 GB | 0.73–0.82 | **ok** |
| xtts_v2 | yes | 2.0 | 5.0 GB | 0.76 | word errors |
| qwenc 0.6B int8, x-vector | yes | 2.2* | ~3 GB | 0.84 | **ok** |
| qwen3 0.6B (PyTorch fp32), clone | yes | 5.6 | 6.5 GB | 0.76 | |
| qwenc 0.6B int8, full clone | yes | 5.9* | 3.4 GB | 0.81 | |
| chatterbox multilingual | yes | 6.3 | 6.8 GB | 0.81 | not acceptable |
| qwen3 1.7B (PyTorch fp32), clone | yes | 7.2 | 11 GB | 0.78 | **ok** |
| espeech RL-V2, nfe 32 | yes | 58 | 5.5 GB | 0.82 | **ok** |

\* includes reloading the model for each of 4 chunks; `--serve` mode
loads it once. Shared cloud CPU - a rerun after a container restart
measured the same qwenc configuration at 6.5, so treat CPU numbers as ±2×.

What it means for the ~13 h of audio in the vault's AI/CS/Math core: on
a 4-core CPU, Silero takes ~1.5 h; a cloned voice takes a day or more
(qwenc x-vector) to several days (full cloning). A GPU is the practical
path for cloning.

## Running on Windows with an 8 GB RTX 4060 Ti

```powershell
winget install astral-sh.uv Gyan.FFmpeg
git clone https://github.com/danrd/lector; cd lector
uv run --no-project --python 3.11 bench/tts/run.py --device cuda `
    --ref C:\path\to\voice.mp3 --ref-seconds 10.45 --ref-text "..."
```

- 8 GB is enough for every engine here one at a time (run.py runs them
  sequentially): Qwen3 1.7B in bf16 needs ~4-6 GB, ESpeech ~3-4 GB.
- `qwen-tts` is run with `attn_implementation="sdpa"` - flash-attn has no
  practical Windows build.
- Scoring Whisper runs on CPU by default (`--asr-device cuda` needs the
  cuBLAS/cuDNN DLLs that faster-whisper looks for).
- `qwenc` needs WSL2 (`make cuda` there); skip it on native Windows.
