"""Universal domain-agnostic Jinja2-based prompt builder.
A prompt is composed of ordered "blocks" (BlockSpec). Each block resolves to
a `<name>/<version>.j2` template file under `config.blocks_dir` (default:
`data/prompts/`), rendered with the shared `context` dict. Blocks are
token-budgeted and joined according to `config.join_format`, or via
`tokenizer.apply_chat_template` when `config.chat_template` is set.

`resolvers`/`filters` named in PromptingConfig are looked up in whatever
resolver_registry/filter_registry the caller passes into PromptBuilder's
constructor - this module has no registry of its own and no project-
specific imports, so it stays genuinely reusable as-is: define your own
resolver/filter functions and pass them in as plain {name: callable} dicts.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, ConfigDict, Field


class BlockSpec(BaseModel):
    """Prompt block specification."""
    name: str
    version: str = "v1"
    role: Literal["system", "user"] = "user"  # role for chat template
    tag: Optional[str] = None  # specify tags for wrapping cusomization

    @classmethod
    def parse(cls, spec: "str | tuple | BlockSpec"):
        if isinstance(spec, str):
            return cls(name=spec)
        if isinstance(spec, tuple):
            return cls(name=spec[0], version=spec[1])
        return spec


class PromptingConfig(BaseModel):
    """Config to guide prompt construction."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    blocks_dir: str = "data/prompts"
    blocks: List[BlockSpec | str] = ["general_instruction", "examples", "output_format"]  # list of element types to compose prompt
    block_overrides: Optional[Dict[str, str]] = Field(default_factory=dict)  # specific blocks subsitution while experimenting
    token_limit: int = 9000  # resources management
    min_examples: int = 2    # examples block must fit at least this many
    filters: Optional[List[str]] = Field(default_factory=list)  # project-specific data processing functions
    resolvers: Optional[List[str]] = Field(default_factory=list)  # project-specific complex prompting methods
    join_format: Literal["xml", "md", "plain"] = "xml"  # approach for blocks composing
    chat_template: Optional[str] = None  # optionaly use specific chat template
    assistant_prefix: Optional[str] = None  # string to add before assistant response
    project: Dict[str, Any] = {}  # other project specific prompting settings


class PromptBuilder:
    """Composes a prompt string (or chat message list) from configured blocks."""

    def __init__(self, config: PromptingConfig, tokenizer,
                 resolver_registry: Optional[Dict[str, Callable]] = None,
                 filter_registry: Optional[Dict[str, Callable]] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.resolver_registry = resolver_registry or {}
        self.filter_registry = filter_registry or {}
        self.env = self._make_env()
        self.resolvers: Dict[str, Callable] = {func_name: self.resolver_registry[func_name] for func_name in self.config.resolvers}

    def _make_env(self) -> Environment:
        env = Environment(
            loader=FileSystemLoader(self.config.blocks_dir),
            undefined=StrictUndefined,   # KeyError on unknown variables
            trim_blocks=True,            # strip \n after {% block %} tags
            lstrip_blocks=True,          # strip leading whitespace before {% %}
            auto_reload=True,            # re-read .j2 files whose mtime changed
        )
        for filter_name in self.config.filters:
            env.filters[filter_name] = self.filter_registry[filter_name]
        return env

    def reload_env(self) -> None:
        """Force a brand new Environment (e.g. after changing blocks_dir).
        Not required for ordinary template edits — those are picked up
        automatically via auto_reload."""
        self.env = self._make_env()

    def list_blocks(self) -> List[str]:
        """List '<name>/<version>' pairs found on disk under blocks_dir —
        handy for discovering what's available while experimenting."""
        root = Path(self.config.blocks_dir)
        if not root.exists():
            return []
        return sorted(
            f"{p.parent.name}/{p.stem}"
            for p in root.glob("*/*.j2")
        )

    def render_block(self, name: str, version: str = "v1", **context) -> str:
        """Render a single block template directly, bypassing the block list,
        token budget, and join step. Useful for iterating on one .j2 file
        (e.g. from a notebook) without building the whole prompt."""
        template = self.env.get_template(f"{name}/{version}.j2")
        return template.render(**context)

    def build(self, task, context: Optional[dict] = None,
              overrides: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Render and join all configured blocks.

        `context` feeds Jinja variables to every block template.
        `overrides` fully replaces a block's rendered text by name (keeps the
        old prompts_modifications behaviour, without touching templates).
        Returns None if the prompt can't fit within token_limit (even after
        trimming the examples block down to `min_examples`).
        """
        context = context or {}
        overrides = overrides or {}
        parts: "OrderedDict[str, Tuple[BlockSpec, str]]" = OrderedDict()
        used_tokens = 0

        for spec_raw in self.config.blocks:
            spec = BlockSpec.parse(spec_raw)

            if spec.name in overrides:
                rendered = overrides[spec.name]
            elif spec.name in self.resolvers:
                rendered = self.resolvers[spec.name](
                    task, self.config.token_limit - used_tokens, context, self,
                )
                if rendered is None:
                    return None  # this resolver couldn't fit even its minimum
            else:
                template = self.env.get_template(f"{spec.name}/{spec.version}.j2")
                rendered = template.render(**context)

            cost = self.count_tokens(rendered)
            if used_tokens + cost > self.config.token_limit:
                return None
            parts[spec.name] = (spec, rendered)
            used_tokens += cost

        return self._join(parts)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    def _join(self, parts: "OrderedDict[str, Tuple[BlockSpec, str]]") -> str:
        if self.config.chat_template is None:
            sections = [
                self._wrap(spec.tag or spec.name.upper(), content) if spec.tag else content
                for spec, content in parts.values()
            ]
            return "\n".join(sections)

        role_buckets: Dict[str, list] = {"system": [], "user": []}
        for spec, content in parts.values():
            wrapped = self._wrap(spec.tag or spec.name.upper(), content) if spec.tag else content
            role_buckets[spec.role].append(wrapped)

        messages = []
        if role_buckets["system"]:
            messages.append({"role": "system", "content": "\n".join(role_buckets["system"])})
        messages.append({"role": "user", "content": "\n".join(role_buckets["user"])})

        if self.config.assistant_prefix:
            messages.append({"role": "assistant", "content": self.config.assistant_prefix})
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _wrap(self, tag: str, content: str) -> str:
        if self.config.join_format == "xml":
            return f"<{tag}>\n{content}\n</{tag}>"
        if self.config.join_format == "md":
            return f"## {tag}\n\n{content}\n\n---"
        return content  # "plain"
