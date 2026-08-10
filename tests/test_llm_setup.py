"""Tests for llm_kit/llm_setup.py's LlmConfig."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_kit.llm_setup import LlmConfig, _start_llama_cpp_server, _start_vllm_server, resolve_local_model_path


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


def _fake_server_config(tmp_path, chat_template_kwargs=None):
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"")
    base = LlmConfig(model=str(gguf), quant_file="")
    generation = SimpleNamespace(chat_template_kwargs=chat_template_kwargs or {}, max_tokens=64)
    return SimpleNamespace(base=base, generation=generation)


def test_start_llama_cpp_server_passes_chat_template_kwargs_when_set(tmp_path, monkeypatch):
    """llama-cpp-python's server doesn't read chat_template_kwargs from the
    request body (unlike vLLM) - it's a model-load-time CLI flag instead,
    so it has to be on the spawn command, not just in generation_kwargs."""
    monkeypatch.chdir(tmp_path)
    config = _fake_server_config(tmp_path, chat_template_kwargs={"enable_thinking": False})

    with patch("subprocess.Popen") as mock_popen:
        _start_llama_cpp_server(config)

    args = mock_popen.call_args[0][0]
    assert "--chat_template_kwargs" in args
    value = args[args.index("--chat_template_kwargs") + 1]
    assert json.loads(value) == {"enable_thinking": False}


def test_start_llama_cpp_server_omits_chat_template_kwargs_when_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _fake_server_config(tmp_path)

    with patch("subprocess.Popen") as mock_popen:
        _start_llama_cpp_server(config)

    args = mock_popen.call_args[0][0]
    assert "--chat_template_kwargs" not in args


def test_start_vllm_server_does_not_stream_pip_install_output(tmp_path, monkeypatch):
    """pip install --upgrade vllm runs on every call (vLLM's wheels are
    CUDA-version-sensitive - see the inline comment) - a routine success
    shouldn't flood the notebook with its output."""
    monkeypatch.chdir(tmp_path)
    config = SimpleNamespace(base=LlmConfig())

    with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
        mock_run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        _start_vllm_server(config)

    assert mock_run.call_args.kwargs.get("capture_output") is True
    mock_popen.assert_called_once()


def test_start_vllm_server_raises_with_full_output_when_pip_install_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = SimpleNamespace(base=LlmConfig())

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = SimpleNamespace(returncode=1, stdout="resolving...", stderr="conflict")

        with pytest.raises(RuntimeError) as exc_info:
            _start_vllm_server(config)

    message = str(exc_info.value)
    assert "resolving..." in message
    assert "conflict" in message
