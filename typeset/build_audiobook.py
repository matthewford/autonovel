#!/usr/bin/env python3
"""Generate audiobook chapters using pocket-tts with Samantha voice."""
import subprocess, os, sys, json, re
from pathlib import Path

BASE = Path("/home/hermes/autonovel")
CHAPTERS = BASE / "chapters"
OUT = BASE / "audiobook" / "chapters"
VOICE = "/home/hermes/.hermes/tts-voices/Samantha/Samantha.safetensors"
POCKET_TTS = "/home/hermes/.hermes/hermes-agent/venv/bin/pocket-tts"

OUT.mkdir(parents=True, exist_ok=True)

def generate_chapter(n):
    chapter_path = CHAPTERS / f"ch_{n:02d}.md"
    out_path = OUT / f"ch_{n:02d}.wav"

    if not chapter_path.exists():
        print(f"  Chapter {n}: source file not found, skipping")
        return False

    if out_path.exists():
        print(f"  Chapter {n}: already exists, skipping")
        return True

    with open(chapter_path) as f:
        text = f.read()

    # Extract title and body
    lines = text.strip().split('\n')
    title = lines[0].lstrip('# ').strip()
    body = '\n'.join(lines[1:]).strip()

    # Clean up markdown formatting for narration
    # Remove backtick code blocks markers
    body = body.replace('`', '')
    # Remove > quote markers
    body = re.sub(r'^> ', '', body, flags=re.MULTILINE)
    # Replace --- scene breaks with a pause marker
    body = body.replace('---', '...')

    # Combine title + body
    narration = f"{title}.\n\n{body}"

    print(f"  Chapter {n}: generating '{title}' ({len(narration)} chars)...")

    cmd = [
        POCKET_TTS, "generate",
        "--text", narration,
        "--voice", VOICE,
        "--language", "english",
        "--output-path", str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  Chapter {n}: FAILED - {result.stderr[:200]}")
        if out_path.exists():
            out_path.unlink()
        return False

    size = out_path.stat().st_size
    print(f"  Chapter {n}: done ({size:,} bytes)")
    return True

# Generate all 8 chapters
print("Generating audiobook chapters...")
success = 0
for n in range(1, 9):
    if generate_chapter(n):
        success += 1

print(f"\nGenerated {success}/8 chapters")

# Assemble full audiobook
print("\nAssembling full audiobook...")
full_out = BASE / "audiobook" / "full_audiobook.wav"

# Build ffmpeg concat list
wav_files = []
for n in range(1, 9):
    p = OUT / f"ch_{n:02d}.wav"
    if p.exists():
        wav_files.append(str(p))

if not wav_files:
    print("No chapter files to assemble!")
    sys.exit(1)

concat_file = BASE / "audiobook" / "concat.txt"
with open(concat_file, 'w') as f:
    for p in wav_files:
        f.write(f"file '{p}'\n")

cmd = [
    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    "-i", str(concat_file),
    "-c:a", "pcm_s16le",
    str(full_out)
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
if result.returncode != 0:
    print(f"Assembly FAILED: {result.stderr[:300]}")
else:
    size = full_out.stat().st_size
    print(f"Full audiobook: {size:,} bytes ({size/1024/1024:.1f} MB)")

# Also convert to OGG/Opus for Telegram
ogg_out = BASE / "audiobook" / "falling-for-her-audiobook.ogg"
print("\nConverting to OGG/Opus for Telegram...")
cmd = [
    "ffmpeg", "-y", "-i", str(full_out),
    "-c:a", "libopus", "-b:a", "48k",
    "-application", "audio",
    str(ogg_out)
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
if result.returncode != 0:
    print(f"OGG conversion FAILED: {result.stderr[:300]}")
else:
    size = ogg_out.stat().st_size
    print(f"OGG audiobook: {size:,} bytes ({size/1024/1024:.1f} MB)")

print("\nDone!")
