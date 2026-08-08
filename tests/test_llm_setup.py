"""Tests for llm_kit/llm_setup.py's LlmConfig."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from llm_kit.llm_setup import LlmConfig, resolve_local_model_path


def test_resolve_local_model_path_downloads_the_quant_file():
    config = SimpleNamespace(base=LlmConfig())
    with patch("huggingface_hub.hf_hub_download") as mock_download:
        mock_download.return_value = "/data/pretrained_models/Qwen3.6-27B-Q4_K_M.gguf"
        path = resolve_local_model_path(config)

    mock_download.assert_called_once_with(
        repo_id="unsloth/Qwen3.6-27B-GGUF",
        filename="Qwen3.6-27B-Q4_K_M.gguf",
        local_dir="/data/pretrained_models",
    )
    assert path == "/data/pretrained_models/Qwen3.6-27B-Q4_K_M.gguf"


def test_resolve_local_model_path_honors_pretrained_models_dir_override():
    config = SimpleNamespace(base=LlmConfig(pretrained_models_dir="/custom/dir"))
    with patch("huggingface_hub.hf_hub_download") as mock_download:
        mock_download.return_value = "/custom/dir/Qwen3.6-27B-Q4_K_M.gguf"
        resolve_local_model_path(config)

    assert mock_download.call_args.kwargs["local_dir"] == "/custom/dir"


def test_resolve_local_model_path_returns_model_as_is_without_quant_file():
    config = SimpleNamespace(base=LlmConfig(quant_file=""))
    assert resolve_local_model_path(config) == "unsloth/Qwen3.6-27B-GGUF"


def test_resolve_local_model_path_passes_through_an_existing_local_file(tmp_path):
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"")
    config = SimpleNamespace(base=LlmConfig(model=str(gguf)))

    with patch("huggingface_hub.hf_hub_download") as mock_download:
        path = resolve_local_model_path(config)

    mock_download.assert_not_called()
    assert path == str(gguf)
