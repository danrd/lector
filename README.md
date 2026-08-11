# lector

Personal lecture generator: give it a topic, it pulls whatever's relevant
from your Obsidian vault (if anything - see below), turns that into a
structured, serious lecture (not a podcast), verifies its own claims,
rewrites the result for speech, and synthesizes audio. Output: a WAV
file (optionally uploaded to YouTube with a static/slide image later -
not built, see "Not built yet").

Status: the topic -> audio content pipeline is complete end to end - see
`lector/pipeline.py`. Built on [`llm_kit`](https://github.com/danrd/toolkit/blob/main/llm_kit/README.md)
(a dependency, from the `toolkit` repo - see that file for what it
provides: prompt composition, backend-agnostic LLM inference, a
resumable task-processing loop).

## Pieces

- **`lector/knowledge_base.py`** — keyword retrieval over a local Obsidian
  vault. Given a topic, scores every note by title/tag/body overlap and
  returns the ones that clear a relevance bar - capped at `top_k`, but
  **empty when nothing does**. This is deliberate, not a fallback: a
  personal vault is sparse by nature, and "no notes on this topic" has to
  be a normal, common result, not something the retrieval papers over by
  returning the "best" of a bunch of irrelevant notes. Wired into
  `llm_kit.prompt_builder.PromptBuilder` as a resolver via
  `make_knowledge_base_resolver(notes)` - when it finds nothing, the
  resolver returns `""` (not `None`), so building the rest of the prompt
  still proceeds normally without vault content.
- **`lector/lecture.py`** — drafts the lecture from the topic + whatever
  the vault retrieval found. The model marks claims drawn from its own
  knowledge (rather than the provided material) with `[UNVERIFIED: ...]`,
  which the next step keys off of.
- **`lector/verification.py`** — checks every `[UNVERIFIED: ...]` claim:
  a targeted vault search first (cheap, no model call), then a model
  self-check in isolation only if that finds nothing. Resolved claims
  lose their marker; unresolved ones become `[NEEDS REVIEW: ...]` -
  visibly flagged rather than silently accepted.
- **`lector/narration.py`** — strips any remaining `[NEEDS REVIEW: ...]`
  spans (never read aloud - stripped *before* the model sees the text,
  not left for it to handle), then rewrites the rest for speech: formulas
  verbalized in words, markdown structure turned into flowing prose.
  Formula verbalization goes through the model, not a rule-based
  converter - a documented, accepted limitation, not a silently ignored
  one.
- **`lector/tts_runtime.py`** — synthesis backends behind one interface
  (`BaseTTSRunner.synthesize(text) -> WAV bytes`): `PiperRunner` (local,
  CPU, no network - the default), `OpenAITTSRunner` (hosted, explicit
  opt-in), and `OrpheusRunner` (a GGUF speech-LLM via `llama_cpp.Llama` -
  the same class `llm_kit`'s own local LLM backend already wraps, so CPU
  vs GPU is the same `n_gpu_layers` knob, not a separate implementation -
  supports voice cloning from a short reference clip). `OrpheusRunner`
  is built from a researched reference implementation but not yet run
  against real downloaded weights - see its docstring.
- **`lector/orpheus_codec.py`** — the token<->SNAC-codes math
  `OrpheusRunner` needs, factored out as pure functions (no torch/
  llama_cpp/snac import) so it's fully unit-tested independent of any
  model actually being loaded.
- **`lector/audio.py`** — splits arbitrary-length narration text into
  sentence-bounded chunks (no backend is asked to synthesize an entire
  lecture in one call), synthesizes each with a given `TTSRunner`, and
  concatenates the WAV frames into one output file.
- **`lector/pipeline.py`** — `generate_lecture_audio(...)` runs the whole
  thing: draft -> verify -> narrate -> synthesize, for a topic and a set
  of vault notes. Soft failures (a prompt that doesn't fit its
  token_limit) are reported, not masked - the result says exactly which
  stage a run got to, with only the fields that stage reached populated.

## Not built yet

- **Actually running on real models.** The code path is complete and
  tested against fakes/stubs throughout, but nobody has downloaded a
  real Piper voice or Orpheus GGUF + SNAC checkpoint and run the
  pipeline against them yet - see `tts_runtime.py`'s docstring.
- **Voice cloning in practice.** `OrpheusRunner` supports it structurally
  (reference audio + transcript), but needs a real reference recording
  to try.
- **YouTube packaging.** Video container + a static/slide image (e.g.
  with formulas) - explicitly a later, lower-priority step; the audio
  pipeline was always the near-term target.
- **External search as a content source.** Drafting is vault + the
  model's own knowledge only; no web/external retrieval - deferred, not
  a small addition.

## Setup

```bash
uv sync --extra test
cp .env.example .env   # fill in OPENROUTER_API_KEY / OPENAI_API_KEY / WANDB_API_KEY as needed
uv run pytest
```

TTS backends are optional extras, install whichever you'll actually use:

```bash
uv sync --extra tts-local     # Piper - local, CPU, default
uv sync --extra tts-cloud     # OpenAI TTS - hosted, explicit opt-in
uv sync --extra tts-orpheus   # Orpheus GGUF + SNAC - local, voice cloning
```

## License

MIT
