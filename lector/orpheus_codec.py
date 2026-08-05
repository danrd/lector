"""Pure token<->SNAC-codes math for Orpheus GGUF speech generation - no
llama_cpp, torch, or snac import here, so this is fully unit-testable
without any of those installed.

Orpheus (a Llama-3B-backbone speech-LLM) doesn't emit audio directly -
it emits token ids that encode SNAC codec codes, 7 tokens per SNAC
frame, interleaved across SNAC's 3 hierarchical codebooks (coarse,
1 code/frame; mid, 2 codes/frame; fine, 4 codes/frame - a 1:2:4 ratio).
`frame_tokens_to_snac_codes` un-interleaves generated tokens back into
those 3 codebooks for SNAC to decode into audio.
`snac_codes_to_frame_tokens` does the reverse - used to turn a
reference clip's SNAC-encoded codes into the token form Orpheus expects
when it appears as a prior turn in a voice-cloning prompt.

Layout and offsets (128266 base, 4096 = SNAC codebook size) match the
token scheme Orpheus was trained on - see e.g.
https://github.com/Zuellni/Orpheus-GGUF for a reference implementation.
"""
from __future__ import annotations

from typing import List, Tuple

AUDIO_CODE_BASE = 128266
CODEBOOK_SIZE = 4096

START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262


def frame_tokens_to_snac_codes(tokens: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """Un-interleave a flat stream of Orpheus audio tokens (7 per SNAC
    frame) into SNAC's 3 hierarchical codebooks. Inverse of
    snac_codes_to_frame_tokens."""
    if len(tokens) % 7 != 0:
        raise ValueError(f"Expected a multiple of 7 tokens per SNAC frame, got {len(tokens)}")

    layer0: List[int] = []
    layer1: List[int] = []
    layer2: List[int] = []

    for i in range(0, len(tokens), 7):
        t0, t1, t2, t3, t4, t5, t6 = tokens[i:i + 7]
        layer0.append(t0 - AUDIO_CODE_BASE)
        layer1.append(t1 - AUDIO_CODE_BASE - CODEBOOK_SIZE)
        layer2.append(t2 - AUDIO_CODE_BASE - 2 * CODEBOOK_SIZE)
        layer2.append(t3 - AUDIO_CODE_BASE - 3 * CODEBOOK_SIZE)
        layer1.append(t4 - AUDIO_CODE_BASE - 4 * CODEBOOK_SIZE)
        layer2.append(t5 - AUDIO_CODE_BASE - 5 * CODEBOOK_SIZE)
        layer2.append(t6 - AUDIO_CODE_BASE - 6 * CODEBOOK_SIZE)

    return layer0, layer1, layer2


def snac_codes_to_frame_tokens(layer0: List[int], layer1: List[int], layer2: List[int]) -> List[int]:
    """Interleave SNAC's 3 hierarchical codebooks into the flat, offset
    Orpheus token stream. Inverse of frame_tokens_to_snac_codes - used
    to embed a reference clip's codes into a voice-cloning prompt."""
    num_frames = len(layer0)
    if len(layer1) != 2 * num_frames or len(layer2) != 4 * num_frames:
        raise ValueError(
            "SNAC layer lengths must be in a 1:2:4 ratio, got "
            f"{len(layer0)}:{len(layer1)}:{len(layer2)}"
        )

    tokens: List[int] = []
    for i in range(num_frames):
        tokens.append(layer0[i] + AUDIO_CODE_BASE)
        tokens.append(layer1[2 * i] + AUDIO_CODE_BASE + CODEBOOK_SIZE)
        tokens.append(layer2[4 * i] + AUDIO_CODE_BASE + 2 * CODEBOOK_SIZE)
        tokens.append(layer2[4 * i + 1] + AUDIO_CODE_BASE + 3 * CODEBOOK_SIZE)
        tokens.append(layer1[2 * i + 1] + AUDIO_CODE_BASE + 4 * CODEBOOK_SIZE)
        tokens.append(layer2[4 * i + 2] + AUDIO_CODE_BASE + 5 * CODEBOOK_SIZE)
        tokens.append(layer2[4 * i + 3] + AUDIO_CODE_BASE + 6 * CODEBOOK_SIZE)

    return tokens
