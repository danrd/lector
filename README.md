# lector

Personal lecture generator: give it a topic, it pulls whatever's relevant
from your Obsidian vault (if anything - see below), turns that into a
structured, serious lecture (not a podcast), verifies its own claims,
rewrites the result for speech, and synthesizes audio. Output: a WAV
file (optionally uploaded to YouTube with a static/slide image later -
not built, see "Not built yet").

A second, simpler mode - **`lector-read`** - reads an existing vault note
(or chosen sections of it) aloud verbatim, with only formulas put into
words; see "Reading a note aloud" below. That one has been run end to
end against live models.

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
- **`lector/obsidian.py`** — note -> speakable lines for `lector-read`:
  picks sections by heading (case/emoji-insensitive, subsections
  included, the per-topic "🔗 Sources" link lists cut out), then strips
  Obsidian syntax - wikilinks become their visible text, the `map`
  navigation callout, code blocks and tables are skipped (and counted),
  emphasis/tags/emoji go. Formulas and code spans are left untouched for
  the next step.
- **`lector/verbalize.py`** — the only model step of `lector-read`:
  lines containing notation (LaTeX, code spans, Greek letters, arrows)
  are sent, whole, to a model that turns just the notation into words -
  with the grammatical agreement Russian needs ("для набора а один, а
  два"). Lines without notation never reach the model. Every rewrite is
  checked to have kept the line's ordinary words; ones that didn't are
  reported as drifted rather than silently accepted.
- **`lector/read_cli.py`** — the `lector-read` command (see below).
- **`lector/ffmpeg.py`** — decoding hosted TTS output (mp3 at any rate)
  into one WAV format, and encoding the final file.
- **`lector/cli.py`** — the command-line entry point (`lector` or
  `python -m lector`): loads the vault, picks a TTS backend, runs the
  pipeline, and writes the WAV plus the narration script beside it. Pure
  composition - it holds no pipeline logic of its own. A run that stops
  early reports the stage it stopped at and exits non-zero, and anything
  left unresolved by verification is printed rather than quietly dropped.

## Not built yet

- **Actually running on real models.** The code path is complete and
  tested against fakes/stubs throughout, but nobody has downloaded a
  real Piper voice or Orpheus GGUF + SNAC checkpoint and run the
  pipeline against them yet - see `tts_runtime.py`'s docstring.
- **Voice cloning in practice.** `OrpheusRunner` supports it structurally
  (reference audio + transcript), but needs a real reference recording
  to try. (Cloning through CometAPI's `seed_audio` - `lector-read
  --voice-ref` - does work; this item is about the local Orpheus path.)
- **YouTube packaging.** Video container + a static/slide image (e.g.
  with formulas) - explicitly a later, lower-priority step; the audio
  pipeline was always the near-term target.
- **External search as a content source.** Drafting is vault + the
  model's own knowledge only; no web/external retrieval - deferred, not
  a small addition.

## Setup

```bash
uv sync --extra test
cp .env.example .env   # fill in COMET_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY / WANDB_API_KEY as needed
uv run pytest
```

## Reading a note aloud

```bash
uv sync --extra comet          # plus the ffmpeg binary on PATH
export COMET_API_KEY=...       # see .env.example

uv run lector-read Math/02_Probability --vault ~/KB --list-sections
uv run lector-read Math/02_Probability --vault ~/KB --section "Теорема Байеса" --section MLE
```

Writes `02_Probability-Теорема_Байеса-MLE.mp3`, the exact script that
was voiced (`.txt`) and `.formulas.md` - every line whose formulas a
model put into words, original beside spoken form, drifted ones first.
The note is read in its own language (detected per chunk); nothing but
notation is reworded.

To check the formulas before paying for audio: `--script-only`, review
or edit the `.txt`, then `uv run lector-read --script that.txt`. Synthesized
chunks are cached (`~/.cache/lector/tts`), so re-voicing an edited
script only pays for the chunks that changed.

Voices, all through CometAPI (`--tts comet`, the default):

- `--model eleven_v3 --voice Bernard` (default) - ElevenLabs with one of
  Runway's preset voices (`Maya`, `Mark`, `Rachel`, ... - see
  `--voice` in the CometAPI Runway TTS docs). The most natural prosody.
- `--voice-ref me.mp3` - clones the voice in a 15-30 s clean recording
  (switches to `seed_audio`; no transcript needed). Its chunks are kept
  short (~500 characters): seed_audio silently stops at about a minute
  of speech per request.
- `--model gpt-4o-mini-tts --voice onyx` - OpenAI's voices, fastest.

`--formulas raw` skips the model entirely (only the `$`/`` ` `` markup is
dropped); `--llm-model` picks the CometAPI chat model that reads
formulas (default `claude-sonnet-4-6`). `lector-read --help` lists the
rest.

## Generating a lecture on a topic

Generating a lecture needs the `openrouter` extra (the CLI generates
through OpenRouter) and a TTS backend:

```bash
uv sync --extra openrouter --extra tts-local
uv run lector "Galois theory" --vault ~/ObsidianVault --voice /path/to/voice.onnx
```

`--vault` is optional: without it the lecture draws on the model's own
knowledge only, and verification flags those claims as ungrounded.
`uv run lector --help` lists the rest (`--tts`, `--language`, `--speed`,
`--model`).

TTS backends are optional extras, install whichever you'll actually use:

```bash
uv sync --extra tts-local     # Piper - local, CPU, default
uv sync --extra tts-cloud     # OpenAI TTS - hosted, explicit opt-in
uv sync --extra tts-orpheus   # Orpheus GGUF + SNAC - local, voice cloning
```

## License

MIT
