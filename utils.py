#!/usr/bin/env python3
"""
Shared utilities for autonovel scripts.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent


def extract_text_from_response(resp_json):
    """
    Extract text content from an Anthropic Messages API response.

    When extended thinking is enabled, the response 'content' array may contain
    thinking blocks (type="thinking") before the text block (type="text").
    This function correctly finds and returns the text block content.

    Args:
        resp_json: The parsed JSON response from the Anthropic API.

    Returns:
        The text content from the first text block in the response.

    Raises:
        ValueError: If no text block is found in the response.
    """
    content = resp_json.get("content", [])

    # Find the first text block (skip thinking blocks)
    for block in content:
        if block.get("type") == "text":
            return block["text"]

    # Fallback: if content is a simple list with text key (older API format)
    if content and "text" in content[0]:
        return content[0]["text"]

    raise ValueError(
        f"No text block found in response. "
        f"Block types: {[b.get('type', 'unknown') for b in content]}. "
        f"Stop reason: {resp_json.get('stop_reason', 'unknown')}"
    )


def get_thinking_budget(max_tokens):
    """
    Calculate the thinking token budget to reserve alongside max_tokens.

    When extended thinking is enabled, thinking tokens are counted against
    the max_tokens budget. This helper reserves a portion for thinking
    while ensuring enough tokens remain for actual output.

    Args:
        max_tokens: The total max_tokens value.

    Returns:
        A dict with 'thinking' config if budget allows, or empty dict.
    """
    # Reserve up to 50% of max_tokens for thinking, capped at 16k
    # Minimum output reserve: 2000 tokens
    thinking_budget = min(max_tokens // 2, 16000)
    if thinking_budget >= 1000 and (max_tokens - thinking_budget) >= 2000:
        return {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
    return {}


def get_novel_title(base_dir=None):
    """
    Return the configured novel title, falling back to outline headings.
    """
    if base_dir is None:
        base_dir = BASE_DIR
    try:
        from book_profile import load_book_profile

        title = load_book_profile(Path(base_dir)).title
        if title and title != "the novel":
            return title
    except Exception:
        pass

    import re

    outline_path = Path(base_dir) / "outline.md"
    if not outline_path.exists():
        return "the novel"
    text = outline_path.read_text()
    skip = re.compile(
        r"act\s+structure|chapter|foreshadowing|outline|part\s+\d",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title and not skip.search(title):
                return title
    return "the novel"


def get_max_tokens_with_thinking(base_max_tokens):
    """
    Return an appropriate max_tokens value that accounts for thinking overhead.

    When models use extended thinking, a significant portion of max_tokens
    goes to internal reasoning. This function increases the budget to ensure
    adequate room for both thinking and output.

    Args:
        base_max_tokens: The desired output token count.

    Returns:
        The total max_tokens to request (including thinking budget).
    """
    # Multiply by 3x to ensure enough room for thinking + output
    # Minimum: base * 3, capped at 128k
    return min(base_max_tokens * 3, 128000)
