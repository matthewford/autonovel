#!/usr/bin/env python3
"""
Central model gateway for autonovel.

Backends:
- anthropic: preserves the repo's original Anthropic Messages API behavior.
- hermes: drives generation through `hermes chat` so subscription-backed
  providers configured in Hermes can be used without changing generator code.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

from book_profile import RoleConfig, load_book_profile
from utils import extract_text_from_response, get_max_tokens_with_thinking

BASE_DIR = Path(__file__).parent
ANTHROPIC_BETA = "context-1m-2025-08-07"
SENTINEL_START = "<<<AUTONOVEL_OUTPUT>>>"
SENTINEL_END = "<<<END_AUTONOVEL_OUTPUT>>>"


def ensure_llm_configured(role: str = "writer") -> None:
    cfg = load_book_profile(BASE_DIR).role_config(role)
    if cfg.backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY in .env first", file=sys.stderr)
        raise SystemExit(1)
    if cfg.backend == "hermes" and not cfg.model:
        print("ERROR: Set AUTONOVEL_HERMES_MODEL or generation.<role>.model", file=sys.stderr)
        raise SystemExit(1)


def generate(
    prompt: str,
    *,
    system: str,
    role: str = "writer",
    max_tokens: int = 4000,
    temperature: float | None = None,
    timeout: int | None = None,
    require_json: bool = False,
) -> str:
    cfg = load_book_profile(BASE_DIR).role_config(role)
    if temperature is None:
        temperature = cfg.temperature
    backend = cfg.backend.lower().strip()
    if backend == "anthropic":
        return _generate_anthropic(prompt, system, cfg, max_tokens, temperature, timeout)
    if backend == "hermes":
        text = _generate_hermes(prompt, system, cfg, timeout)
        if require_json:
            text = _repair_json_if_needed(text, system, cfg, timeout)
        return text
    raise ValueError(f"Unsupported AUTONOVEL_LLM_BACKEND: {cfg.backend}")


def model_fingerprint(role: str = "writer") -> str:
    cfg = load_book_profile(BASE_DIR).role_config(role)
    if cfg.backend == "hermes":
        parts = ["hermes"]
        if cfg.provider:
            parts.append(cfg.provider)
        if cfg.model:
            parts.append(cfg.model)
        if cfg.profile:
            parts.append(f"profile={cfg.profile}")
        if cfg.use_goals:
            parts.append("goals")
        return "/".join(parts)
    return f"anthropic/{cfg.model}"


def _generate_anthropic(
    prompt: str,
    system: str,
    cfg: RoleConfig,
    max_tokens: int,
    temperature: float,
    timeout: int | None,
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_base = os.environ.get("AUTONOVEL_API_BASE_URL", "https://api.anthropic.com")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": ANTHROPIC_BETA,
        "content-type": "application/json",
    }
    total_tokens = get_max_tokens_with_thinking(max_tokens)
    payload = {
        "model": cfg.model,
        "max_tokens": total_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = httpx.post(
        f"{api_base}/v1/messages",
        headers=headers,
        json=payload,
        timeout=timeout or cfg.timeout_seconds,
    )
    resp.raise_for_status()
    return extract_text_from_response(resp.json())


def _generate_hermes(
    prompt: str,
    system: str,
    cfg: RoleConfig,
    timeout: int | None,
) -> str:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".md", prefix="autonovel_prompt_", delete=False
    ) as prompt_file:
        prompt_path = Path(prompt_file.name)
        prompt_file.write(prompt)

    query = f"""You are being used as the model backend for autonovel.

System instructions:
{system}

Read the full task prompt from this file:
{prompt_path}

Return only the requested creative or analytical output. Do not explain your process.
Wrap the output exactly like this:
{SENTINEL_START}
<your output>
{SENTINEL_END}
"""
    if cfg.use_goals:
        goal = (
            "Produce the complete AutoNovel backend response for this call. "
            f"The goal is complete only when the final response contains output wrapped between "
            f"{SENTINEL_START} and {SENTINEL_END}, follows the requested format, and includes no "
            "process commentary outside those markers."
        )
        query = f"/goal {goal}\n\n{query}"

    cmd = ["hermes", "chat", "-Q", "-q", query]
    if cfg.profile:
        cmd.extend(["--profile", cfg.profile])
    if cfg.provider:
        cmd.extend(["--provider", cfg.provider])
    if cfg.model:
        cmd.extend(["-m", cfg.model])
    if cfg.toolsets:
        cmd.extend(["-t", cfg.toolsets])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=timeout or cfg.timeout_seconds or 1800,
        )
    finally:
        try:
            prompt_path.unlink()
        except FileNotFoundError:
            pass

    if result.returncode != 0:
        raise RuntimeError(
            "Hermes generation failed "
            f"(role={cfg.role}, provider={cfg.provider}, model={cfg.model}):\n"
            f"{result.stderr.strip()}"
        )

    return _extract_hermes_output(result.stdout)


def _extract_hermes_output(stdout: str) -> str:
    cleaned = re.sub(r"(?im)^session_id:\s*\S+\s*$", "", stdout).strip()
    match = re.search(
        rf"{re.escape(SENTINEL_START)}\s*(.*?)\s*{re.escape(SENTINEL_END)}",
        cleaned,
        flags=re.DOTALL,
    )
    if match:
        text = match.group(1).strip()
    else:
        text = cleaned.strip()
    if not text:
        raise RuntimeError("Hermes returned empty output")
    return text


def _repair_json_if_needed(
    text: str,
    system: str,
    cfg: RoleConfig,
    timeout: int | None,
) -> str:
    import json

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        repair_prompt = (
            "Repair this into strict JSON only. Preserve the same fields and values. "
            "Return no markdown fences and no commentary.\n\n"
            f"{text}"
        )
        repaired = _generate_hermes(repair_prompt, system, cfg, timeout)
        json.loads(repaired)
        return repaired
