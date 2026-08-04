# lector

Personal lecture generator: give it a topic, it pulls whatever's relevant
from your Obsidian vault (if anything - see below), turns that into a
structured, serious lecture (not a podcast), and narrates it. Output:
audio (optionally uploaded to YouTube with a static/slide image, e.g. for
formulas).

Status: early. Built on [`llm_kit`](llm_kit/README.md) (vendored here, not
a separate dependency yet - see that file for what it provides: prompt
composition, backend-agnostic LLM inference, a resumable task-processing
loop).

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
- **Content generation, verification, narration-script writing** - not
  built yet. Planned as a small pipeline of roles (structure the lecture,
  write it, fact-check it - formulas especially, rewrite for speech) on
  top of `llm_kit`'s prompt/inference/loop pieces.
- **TTS + audio/video assembly** - not built yet; lower priority for now,
  most of the near-term work is on getting the content pipeline right.

## Setup

```bash
uv sync --extra test
cp .env.example .env   # fill in OPENROUTER_API_KEY / WANDB_API_KEY as needed
uv run pytest
```

## License

MIT
