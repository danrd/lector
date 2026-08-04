"""Tests for llm_kit/prompt_builder.py.

resolvers/filters are looked up in whatever resolver_registry/
filter_registry the caller passes into PromptBuilder's constructor
(default: empty dicts) - the module itself carries no registry and no
sibling-module imports, so it stays usable as a standalone file.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from llm_kit.prompt_builder import PromptBuilder, PromptingConfig


class _FakeTokenizer:
    def tokenize(self, text):
        return text.split()


def _write_block(blocks_dir, name, version, content):
    block_dir = blocks_dir / name
    block_dir.mkdir(parents=True, exist_ok=True)
    (block_dir / f"{version}.j2").write_text(content)


def test_prompt_builder_module_has_no_sibling_module_imports():
    """Regression guard: prompt_builder.py must not import any other
    llm_kit module - it's meant to be usable as a single standalone file,
    with resolvers/filters injected rather than looked up in a shared
    registry module."""
    import llm_kit.prompt_builder as module

    tree = ast.parse(inspect.getsource(module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    sibling_imports = {m for m in imported if m.startswith("llm_kit")}
    assert not sibling_imports, f"prompt_builder.py imports other llm_kit modules: {sibling_imports}"


def test_builds_without_any_registry_when_config_uses_no_resolvers_or_filters(tmp_path):
    _write_block(tmp_path, "greeting", "v1", "Hello, {{ name }}!")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["greeting"], token_limit=100)

    builder = PromptBuilder(config, _FakeTokenizer())
    result = builder.build(task=None, context={"name": "world"})

    assert result == "Hello, world!"


def test_resolver_registry_is_used_when_provided(tmp_path):
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["dynamic"], token_limit=100,
                              resolvers=["dynamic"])

    def my_resolver(task, remaining_tokens, context, builder):
        return f"resolved: {task}"

    builder = PromptBuilder(config, _FakeTokenizer(), resolver_registry={"dynamic": my_resolver})
    result = builder.build(task="my-task", context={})

    assert result == "resolved: my-task"


def test_filter_registry_is_used_when_provided(tmp_path):
    _write_block(tmp_path, "shout", "v1", "{{ text | shout }}")
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["shout"], token_limit=100,
                              filters=["shout"])

    builder = PromptBuilder(config, _FakeTokenizer(), filter_registry={"shout": str.upper})
    result = builder.build(task=None, context={"text": "hi"})

    assert result == "HI"


def test_unregistered_resolver_name_raises_keyerror(tmp_path):
    """No registry passed -> empty default -> a resolver name not found in
    it should fail loudly (KeyError), not silently skip or crash later
    with a confusing error deep inside build()."""
    config = PromptingConfig(blocks_dir=str(tmp_path), blocks=["missing"], token_limit=100,
                              resolvers=["missing"])

    with pytest.raises(KeyError):
        PromptBuilder(config, _FakeTokenizer())
