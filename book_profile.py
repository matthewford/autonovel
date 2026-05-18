#!/usr/bin/env python3
"""
Book profile loading for autonovel.

The project intentionally keeps this dependency-light. If PyYAML is installed,
it is used. Otherwise a small parser handles the simple nested YAML shape used
by book.yaml.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"List item without list parent: {raw_line}")
            parent.append(_parse_scalar(line[2:]))
            continue

        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"Invalid book.yaml line: {raw_line}")
        key = key.strip()
        value = value.strip()

        if value:
            parent[key] = _parse_scalar(value)
            continue

        container: Any = {}
        parent[key] = container
        stack.append((indent, container))

        # Convert empty mapping to a list when the next meaningful line is a
        # list item at a deeper indent. This keeps the parser tiny but useful.
        lines_after = text.splitlines()[text.splitlines().index(raw_line) + 1 :]
        for next_raw in lines_after:
            if not next_raw.strip() or next_raw.lstrip().startswith("#"):
                continue
            next_indent = len(next_raw) - len(next_raw.lstrip(" "))
            if next_indent > indent and next_raw.strip().startswith("- "):
                parent[key] = []
                stack[-1] = (indent, parent[key])
            break

    return root


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("book.yaml must contain a mapping at the top level")
        return data
    except ModuleNotFoundError:
        return _minimal_yaml_load(text)


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value


def _first(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text.strip() != "":
            return text
    return ""


def _deep_get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


@dataclass(frozen=True)
class RoleConfig:
    backend: str
    role: str
    model: str
    provider: str
    profile: str
    toolsets: str
    use_goals: bool
    temperature: float
    timeout_seconds: int


class BookProfile:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = Path(base_dir or BASE_DIR)
        self.data = _load_yaml(self.base_dir / "book.yaml")

    @property
    def title(self) -> str:
        return str(self.get("title", os.environ.get("AUTONOVEL_TITLE", "the novel")))

    @property
    def chapter_target_words(self) -> int:
        return int(self.get("chapter_target_words", 3200))

    @property
    def chapters_target(self) -> int:
        return int(self.get("chapters_target", 24))

    def get(self, path: str, default: Any = None) -> Any:
        return _deep_get(self.data, path, default)

    def role_config(self, role: str) -> RoleConfig:
        backend = _first(_env("AUTONOVEL_LLM_BACKEND"), self.get("generation.backend", "anthropic"))
        role_key = role if role in {"writer", "judge", "reviewer", "reader"} else "writer"
        default_model_env = {
            "writer": "AUTONOVEL_WRITER_MODEL",
            "judge": "AUTONOVEL_JUDGE_MODEL",
            "reviewer": "AUTONOVEL_REVIEW_MODEL",
            "reader": "AUTONOVEL_JUDGE_MODEL",
        }[role_key]
        default_model = {
            "writer": "claude-sonnet-4-6",
            "judge": "claude-opus-4-6",
            "reviewer": "claude-opus-4-6",
            "reader": "claude-opus-4-6",
        }[role_key]
        prefix = f"AUTONOVEL_HERMES_{role_key.upper()}_"
        hermes_model = _first(
            _env(f"{prefix}MODEL"),
            _env("AUTONOVEL_HERMES_MODEL"),
            self.get(f"generation.{role_key}.model"),
            default_model,
        )
        anthropic_model = _first(
            _env(default_model_env),
            self.get(f"generation.{role_key}.anthropic_model"),
            default_model,
        )
        use_goals = _first(
            _env(f"{prefix}USE_GOALS"),
            _env("AUTONOVEL_HERMES_USE_GOALS"),
            self.get(f"generation.{role_key}.use_goals", self.get("generation.use_goals", False)),
        )
        return RoleConfig(
            backend=backend,
            role=role_key,
            model=hermes_model if backend.lower().strip() == "hermes" else anthropic_model,
            provider=_first(
                _env(f"{prefix}PROVIDER"),
                _env("AUTONOVEL_HERMES_PROVIDER"),
                self.get(f"generation.{role_key}.provider"),
            ),
            profile=_first(
                _env(f"{prefix}PROFILE"),
                _env("AUTONOVEL_HERMES_PROFILE"),
                self.get(f"generation.{role_key}.profile"),
            ),
            toolsets=_first(
                _env(f"{prefix}TOOLSETS"),
                _env("AUTONOVEL_HERMES_TOOLSETS"),
                self.get(f"generation.{role_key}.toolsets", "file,safe"),
            ),
            use_goals=str(use_goals).lower() in {"1", "true", "yes", "on"},
            temperature=float(self.get(f"generation.{role_key}.temperature", 0.7)),
            timeout_seconds=int(self.get(f"generation.{role_key}.timeout_seconds", 900)),
        )


def load_book_profile(base_dir: Path | None = None) -> BookProfile:
    return BookProfile(base_dir)
