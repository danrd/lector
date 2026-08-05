"""Tests for lector/orpheus_codec.py - pure integer math, no torch/
llama_cpp/snac needed, so this can (and should) be tested exhaustively."""
from __future__ import annotations

import pytest

from lector.orpheus_codec import (
    AUDIO_CODE_BASE,
    CODEBOOK_SIZE,
    frame_tokens_to_snac_codes,
    snac_codes_to_frame_tokens,
)


def _make_frame_tokens(layer0_val, layer1_vals, layer2_vals):
    """Build one 7-token frame directly from the documented offset
    layout, independent of the module under test."""
    return [
        layer0_val + AUDIO_CODE_BASE,
        layer1_vals[0] + AUDIO_CODE_BASE + CODEBOOK_SIZE,
        layer2_vals[0] + AUDIO_CODE_BASE + 2 * CODEBOOK_SIZE,
        layer2_vals[1] + AUDIO_CODE_BASE + 3 * CODEBOOK_SIZE,
        layer1_vals[1] + AUDIO_CODE_BASE + 4 * CODEBOOK_SIZE,
        layer2_vals[2] + AUDIO_CODE_BASE + 5 * CODEBOOK_SIZE,
        layer2_vals[3] + AUDIO_CODE_BASE + 6 * CODEBOOK_SIZE,
    ]


# -- frame_tokens_to_snac_codes ----------------------------------------------

def test_frame_tokens_to_snac_codes_decodes_a_single_frame():
    tokens = _make_frame_tokens(layer0_val=10, layer1_vals=[20, 21], layer2_vals=[30, 31, 32, 33])

    layer0, layer1, layer2 = frame_tokens_to_snac_codes(tokens)

    assert layer0 == [10]
    assert layer1 == [20, 21]
    assert layer2 == [30, 31, 32, 33]


def test_frame_tokens_to_snac_codes_decodes_multiple_frames_in_order():
    frame_a = _make_frame_tokens(1, [2, 3], [4, 5, 6, 7])
    frame_b = _make_frame_tokens(11, [12, 13], [14, 15, 16, 17])

    layer0, layer1, layer2 = frame_tokens_to_snac_codes(frame_a + frame_b)

    assert layer0 == [1, 11]
    assert layer1 == [2, 3, 12, 13]
    assert layer2 == [4, 5, 6, 7, 14, 15, 16, 17]


def test_frame_tokens_to_snac_codes_rejects_a_length_not_a_multiple_of_seven():
    with pytest.raises(ValueError):
        frame_tokens_to_snac_codes([AUDIO_CODE_BASE] * 6)


def test_frame_tokens_to_snac_codes_handles_empty_input():
    assert frame_tokens_to_snac_codes([]) == ([], [], [])


# -- snac_codes_to_frame_tokens ----------------------------------------------

def test_snac_codes_to_frame_tokens_matches_the_documented_offset_layout():
    tokens = snac_codes_to_frame_tokens(layer0=[10], layer1=[20, 21], layer2=[30, 31, 32, 33])

    assert tokens == _make_frame_tokens(10, [20, 21], [30, 31, 32, 33])


def test_snac_codes_to_frame_tokens_rejects_a_ratio_mismatch():
    with pytest.raises(ValueError):
        snac_codes_to_frame_tokens(layer0=[1, 2], layer1=[1, 2, 3], layer2=[1, 2, 3, 4, 5, 6, 7, 8])


# -- round trip ---------------------------------------------------------

def test_round_trip_tokens_through_snac_codes_and_back():
    original_tokens = _make_frame_tokens(1, [2, 3], [4, 5, 6, 7]) + \
        _make_frame_tokens(100, [200, 201], [300, 301, 302, 303])

    layer0, layer1, layer2 = frame_tokens_to_snac_codes(original_tokens)
    rebuilt_tokens = snac_codes_to_frame_tokens(layer0, layer1, layer2)

    assert rebuilt_tokens == original_tokens


def test_round_trip_snac_codes_through_tokens_and_back():
    layer0 = [5, 15, 25]
    layer1 = [1, 2, 11, 12, 21, 22]
    layer2 = [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]

    tokens = snac_codes_to_frame_tokens(layer0, layer1, layer2)
    rebuilt_layer0, rebuilt_layer1, rebuilt_layer2 = frame_tokens_to_snac_codes(tokens)

    assert (rebuilt_layer0, rebuilt_layer1, rebuilt_layer2) == (layer0, layer1, layer2)
