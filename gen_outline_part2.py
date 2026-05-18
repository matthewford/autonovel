#!/usr/bin/env python3
"""Generate remaining chapters + foreshadowing ledger (generic version)."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from book_profile import load_book_profile
from llm import generate
from utils import get_novel_title

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

def call_writer(prompt, max_tokens=16000):
    return generate(
        prompt,
        role="writer",
        max_tokens=max_tokens,
        temperature=0.5,
        system=(
            "You are a novel architect continuing an outline. Write in the same format "
            "as the preceding chapters. Every chapter needs: POV, Location, Save the Cat beat, "
            "% mark, Emotional arc, Try-fail cycle, Beats, Plants, Payoffs, Character movement, "
            "The lie, Word count target."
        ),
    )

# Prefer existing outline.md, fall back to temp file
outline_path = BASE_DIR / "outline.md"
if outline_path.exists():
    part1 = outline_path.read_text()
else:
    try:
        part1 = open('/tmp/outline_output.md').read()
    except FileNotFoundError:
        part1 = ""

mystery = (BASE_DIR / "MYSTERY.md").read_text() if (BASE_DIR / "MYSTERY.md").exists() else ""
profile = load_book_profile(BASE_DIR)
title = get_novel_title(BASE_DIR)

prompt = f"""Continue or complete the chapter outline for "{title}".

THE OUTLINE SO FAR:
{part1}

THE CENTRAL MYSTERY (for reference):
{mystery}

Complete any missing chapters and then write a complete Foreshadowing Ledger table with at least 12-15 threads.
Use the same format as the existing outline (Ch N, POV, Save the Cat beat, % mark, Emotional arc, Try-fail cycle, Beats, Plants, Payoffs, etc.).

Target: {profile.chapters_target} chapters total.
"""

print("Calling writer model...", file=sys.stderr)
result = call_writer(prompt)
print(result)