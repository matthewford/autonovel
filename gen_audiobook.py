#!/usr/bin/env python3
"""
Generate audiobook from parsed scripts using multiple TTS providers.
Supports ElevenLabs, Kokoro (Local), Piper (Local), MLX-TTS (Local), KittenTTS (Local), and OpenAI.

Usage:
  python gen_audiobook.py                # Generate all chapters (default: elevenlabs)
  python gen_audiobook.py 1              # Single chapter
  python gen_audiobook.py 1 5            # Range
  python gen_audiobook.py --list-voices  # List available voices
  python gen_audiobook.py --test 1       # Generate first segments of ch 1 (test)
  python gen_audiobook.py --provider mlx # Use local MLX-TTS
  python gen_audiobook.py --provider kittentts # Use local KittenTTS
"""
import os
import sys
import json
import io
import re
import time
import argparse
import subprocess
import tempfile
import wave
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import httpx

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env", override=True)

AUDIO_DIR = BASE_DIR / "audiobook"
SCRIPTS_DIR = AUDIO_DIR / "scripts"
OUTPUT_DIR = AUDIO_DIR / "chapters"
VOICES_FILE = BASE_DIR / "audiobook_voices.json"
MODELS_DIR = BASE_DIR / "models"

MAX_CHARS_PER_CALL = 4500  # stay under 5000 limit with overhead
PAUSE_BETWEEN_CALLS = 3.0  # rate limiting — ElevenLabs has per-minute caps


# --- Provider Abstraction ---

class TTSProvider(ABC):
    """Base class for Text-to-Speech providers."""
    @abstractmethod
    def generate(self, segments: List[Dict[str, str]]) -> bytes:
        """Takes a list of {text, voice_id} and returns combined MP3/WAV bytes."""
        pass

    @property
    @abstractmethod
    def max_chars(self) -> int:
        """Maximum characters allowed per generation call."""
        pass

    @property
    def output_format(self) -> str:
        """File extension/format produced by this provider."""
        return "mp3"


class ElevenLabsProvider(TTSProvider):
    """ElevenLabs TTS Provider."""
    def __init__(self, api_key: str):
        from elevenlabs.client import ElevenLabs
        self.client = ElevenLabs(api_key=api_key)
        self._max_chars = MAX_CHARS_PER_CALL

    @property
    def voices(self):
        return self.client.voices

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def generate(self, segments: List[Dict[str, str]]) -> bytes:
        audio = self.client.text_to_dialogue.convert(inputs=segments)
        audio_bytes = b""
        for chunk in audio:
            if isinstance(chunk, bytes):
                audio_bytes += chunk
        return audio_bytes


class KokoroProvider(TTSProvider):
    """Local Kokoro TTS Provider using ONNX."""
    def __init__(self):
        from kokoro_onnx import Kokoro
        model_path = MODELS_DIR / "kokoro-v0_19.onnx"
        voices_path = MODELS_DIR / "voices.bin"

        if not model_path.exists() or not voices_path.exists():
            print(f"ERROR: Kokoro models not found in {MODELS_DIR}")
            print("Please download 'kokoro-v0_19.onnx' and 'voices.bin' to the models/ directory.")
            sys.exit(1)

        self.kokoro = Kokoro(str(model_path), str(voices_path))
        self._max_chars = 1000000

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def generate(self, segments: List[Dict[str, str]]) -> bytes:
        import numpy as np
        import soundfile as sf
        from pydub import AudioSegment

        combined_audio = []
        sample_rate = 24000
        for seg in segments:
            samples, sr = self.kokoro.create(seg['text'], voice=seg['voice_id'], speed=1.0, lang="en-us")
            combined_audio.append(samples)
            sample_rate = sr
            combined_audio.append(np.zeros(int(sr * 0.1)))

        if not combined_audio:
            return b""
        full_audio = np.concatenate(combined_audio)
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, sample_rate, format='WAV')
        buffer.seek(0)
        audio_seg = AudioSegment.from_wav(buffer)
        mp3_buffer = io.BytesIO()
        audio_seg.export(mp3_buffer, format="mp3", bitrate="192k")
        return mp3_buffer.getvalue()


class KittenTTSProvider(TTSProvider):
    """Local KittenTTS Provider."""
    def __init__(self):
        from kittentts import KittenTTS
        self.model = KittenTTS("KittenML/kitten-tts-mini-0.8")
        self._max_chars = 1000000

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def generate(self, segments: List[Dict[str, str]]) -> bytes:
        import numpy as np
        import soundfile as sf
        from pydub import AudioSegment

        combined_audio = []
        sample_rate = 24000
        for seg in segments:
            audio = self.model.generate(seg['text'], voice=seg['voice_id'])
            combined_audio.append(audio)
            combined_audio.append(np.zeros(int(sample_rate * 0.1)))

        if not combined_audio:
            return b""
        full_audio = np.concatenate(combined_audio)
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, sample_rate, format='WAV')
        buffer.seek(0)
        audio_seg = AudioSegment.from_wav(buffer)
        mp3_buffer = io.BytesIO()
        audio_seg.export(mp3_buffer, format="mp3", bitrate="192k")
        return mp3_buffer.getvalue()


class PiperProvider(TTSProvider):
    """Local Piper TTS Provider."""
    def __init__(self):
        from piper import PiperVoice
        self.PiperVoice = PiperVoice
        self._max_chars = 5000
        self.voices_cache = {}

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def _get_voice(self, voice_id: str):
        if voice_id not in self.voices_cache:
            model_path = MODELS_DIR / f"{voice_id}.onnx"
            config_path = MODELS_DIR / f"{voice_id}.onnx.json"
            if not model_path.exists():
                voice_id = "en_US-lessac-medium"
                model_path = MODELS_DIR / f"{voice_id}.onnx"
                config_path = MODELS_DIR / f"{voice_id}.onnx.json"

            if not model_path.exists():
                print(f"ERROR: Piper model {voice_id} not found in {MODELS_DIR}")
                return None
            self.voices_cache[voice_id] = self.PiperVoice.load(str(model_path), str(config_path))
        return self.voices_cache[voice_id]

    def generate(self, segments: List[Dict[str, str]]) -> bytes:
        from pydub import AudioSegment

        combined = AudioSegment.empty()
        for seg in segments:
            voice = self._get_voice(seg['voice_id'])
            if not voice:
                continue

            buffer = io.BytesIO()
            import wave
            with wave.open(buffer, "wb") as wav_file:
                voice.synthesize_wav(seg['text'], wav_file)

            buffer.seek(0)
            part = AudioSegment.from_wav(buffer)
            combined += part
            combined += AudioSegment.silent(duration=100)

        mp3_buffer = io.BytesIO()
        combined.export(mp3_buffer, format="mp3", bitrate="192k")
        return mp3_buffer.getvalue()


class MLXTTSProvider(TTSProvider):
    """Local MLX-TTS Provider for Apple Silicon."""
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self._max_chars = 10000

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def generate(self, segments: List[Dict[str, str]]) -> bytes:
        from pydub import AudioSegment

        combined = AudioSegment.empty()
        with httpx.Client(timeout=300.0) as client:
            for seg in segments:
                response = client.post(
                    f"{self.base_url}/v1/audio/speech",
                    json={
                        "model": "mlx-community/Kokoro-82M",
                        "input": seg['text'],
                        "voice": seg['voice_id'],
                        "speed": 1.0
                    }
                )
                if response.status_code == 200:
                    part = AudioSegment.from_file(io.BytesIO(response.content))
                    combined += part
                    combined += AudioSegment.silent(duration=100)

        mp3_buffer = io.BytesIO()
        combined.export(mp3_buffer, format="mp3", bitrate="192k")
        return mp3_buffer.getvalue()


class OpenAITTSProvider(TTSProvider):
    """OpenAI Cloud TTS Provider."""
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self._max_chars = 4000

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def generate(self, segments: List[Dict[str, str]]) -> bytes:
        from pydub import AudioSegment

        combined = AudioSegment.empty()
        for seg in segments:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=seg['voice_id'],
                input=seg['text']
            )
            part = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
            combined += part
            combined += AudioSegment.silent(duration=200)
        mp3_buffer = io.BytesIO()
        combined.export(mp3_buffer, format="mp3", bitrate="192k")
        return mp3_buffer.getvalue()


class PocketTTSProvider(TTSProvider):
    """Hermes local Pocket TTS command provider."""

    def __init__(self):
        self.script = Path(os.environ.get(
            "AUTONOVEL_POCKET_TTS_SCRIPT",
            "/home/hermes/.hermes/scripts/pocket_tts_provider.py",
        ))
        self.python = os.environ.get(
            "AUTONOVEL_POCKET_TTS_PYTHON",
            "/home/hermes/.hermes/hermes-agent/venv/bin/python",
        )
        self.language = os.environ.get("AUTONOVEL_POCKET_TTS_LANGUAGE", "english")
        self.device = os.environ.get("AUTONOVEL_POCKET_TTS_DEVICE", "cpu")
        self.voice = os.environ.get("AUTONOVEL_POCKET_TTS_VOICE", "Samantha")
        self.hermes_home = os.environ.get("AUTONOVEL_POCKET_TTS_HERMES_HOME", "/home/hermes/.hermes")
        self._max_chars = int(os.environ.get("AUTONOVEL_POCKET_TTS_MAX_CHARS", "5000"))

        if not self.script.exists():
            print(f"ERROR: Pocket TTS provider script not found: {self.script}", file=sys.stderr)
            sys.exit(1)

    @property
    def max_chars(self) -> int:
        return self._max_chars

    @property
    def output_format(self) -> str:
        return "wav"

    def _generate_wav_file(self, text: str, output_path: Path):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
            input_path = Path(f.name)
            f.write(text)

        cmd = [
            self.python,
            str(self.script),
            str(input_path),
            str(output_path),
            self.language,
            self.device,
            self.voice,
        ]
        try:
            env = os.environ.copy()
            env["HERMES_HOME"] = self.hermes_home
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        finally:
            try:
                input_path.unlink()
            except FileNotFoundError:
                pass
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "pocket-tts failed")

    def generate(self, segments: List[Dict[str, str]]) -> bytes:
        wav_paths = []
        with tempfile.TemporaryDirectory(prefix="autonovel_pocket_tts_") as tmp:
            tmp_dir = Path(tmp)
            for idx, seg in enumerate(segments, 1):
                clean_text = re.sub(r"\[.*?\]", "", seg["text"]).strip()
                if not clean_text:
                    continue
                wav_path = tmp_dir / f"segment_{idx:04d}.wav"
                self._generate_wav_file(clean_text, wav_path)
                wav_paths.append(wav_path)

            if not wav_paths:
                return b""

            out_buffer = io.BytesIO()
            writer = None
            try:
                for wav_path in wav_paths:
                    with wave.open(str(wav_path), "rb") as reader:
                        params = reader.getparams()
                        frames = reader.readframes(reader.getnframes())
                        if writer is None:
                            writer = wave.open(out_buffer, "wb")
                            writer.setparams(params)
                        elif reader.getparams()[:3] != writer.getparams()[:3]:
                            raise RuntimeError("Pocket TTS returned inconsistent WAV parameters")
                        writer.writeframes(frames)
                if writer is not None:
                    writer.close()
                    writer = None
                return out_buffer.getvalue()
            finally:
                if writer is not None:
                    writer.close()


def get_client(name: str) -> TTSProvider:
    """Initialize selected TTS provider."""
    name = name.lower()
    if name == "elevenlabs":
        key = os.environ.get("ELEVENLABS_API_KEY", "")
        if not key:
            print("ERROR: ELEVENLABS_API_KEY not set in .env", file=sys.stderr)
            sys.exit(1)
        return ElevenLabsProvider(key)
    elif name == "kokoro":
        return KokoroProvider()
    elif name == "kittentts":
        return KittenTTSProvider()
    elif name == "piper":
        return PiperProvider()
    elif name == "mlx":
        url = os.environ.get("MLX_TTS_SERVER_URL", "http://localhost:8000")
        return MLXTTSProvider(url)
    elif name == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            print("ERROR: OPENAI_API_KEY not set in .env", file=sys.stderr)
            sys.exit(1)
        return OpenAITTSProvider(key)
    elif name in {"pocket-tts", "pocket"}:
        return PocketTTSProvider()
    else:
        print(f"ERROR: Unknown provider {name}")
        sys.exit(1)


def load_voices(provider_name: str):
    """Load voice mapping from audiobook_voices.json."""
    if provider_name in {"pocket-tts", "pocket"}:
        voice = os.environ.get("AUTONOVEL_POCKET_TTS_VOICE", "Samantha")
        return {"NARRATOR": voice, "MINOR": voice}

    if not VOICES_FILE.exists():
        print(f"ERROR: {VOICES_FILE} not found. Create it first.", file=sys.stderr)
        sys.exit(1)
    data = json.loads(VOICES_FILE.read_text())
    voices = {}
    for name, info in data.items():
        if name.startswith("_"):
            continue
        vid = info.get("providers", {}).get(provider_name)
        if vid and vid != "REPLACE_WITH_VOICE_ID":
            voices[name] = vid
    return voices


def load_script(ch_num):
    """Load a chapter's parsed script."""
    path = SCRIPTS_DIR / f"ch{ch_num:02d}_script.json"
    if not path.exists():
        print(f"  Script not found: {path}. Run gen_audiobook_script.py first.", file=sys.stderr)
        return None
    return json.loads(path.read_text())


def chunk_segments(segments, voices, max_chars=MAX_CHARS_PER_CALL):
    """Split segments into chunks that fit within the API character limit.

    Each chunk is a list of {text, voice_id} dicts suitable for text_to_dialogue.
    We try to keep dialogue exchanges together (don't split mid-conversation).
    """
    chunks = []
    current_chunk = []
    current_chars = 0
    fallback_voice = list(voices.values())[0] if voices else None

    for seg in segments:
        speaker = seg["speaker"]
        text = seg["text"]

        # Resolve voice_id
        voice_id = voices.get(speaker)
        if not voice_id:
            voice_id = voices.get("MINOR", voices.get("NARRATOR", fallback_voice))
        if not voice_id:
            continue

        seg_chars = len(text)

        # If this single segment exceeds the limit, split it
        if seg_chars > max_chars:
            # Flush current chunk
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_chars = 0

            # Split long text into sentences
            sentences = text.replace(". ", ".\n").split("\n")
            sub_chunk = []
            sub_chars = 0
            for sent in sentences:
                if sub_chars + len(sent) > max_chars and sub_chunk:
                    chunks.append([{"text": " ".join(sub_chunk), "voice_id": voice_id}])
                    sub_chunk = []
                    sub_chars = 0
                sub_chunk.append(sent)
                sub_chars += len(sent) + 1
            if sub_chunk:
                chunks.append([{"text": " ".join(sub_chunk), "voice_id": voice_id}])
            continue

        # Would adding this segment exceed the limit?
        if current_chars + seg_chars > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0

        # Skip empty/whitespace-only segments
        clean_text = re.sub(r'\[.*?\]', '', text).strip()
        if not clean_text:
            continue

        current_chunk.append({"text": text, "voice_id": voice_id})
        current_chars += seg_chars

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def generate_chapter(ch_num, client, voices, test_mode=False):
    """Generate audio for a single chapter."""
    # Note: client is an instance of TTSProvider (client == provider)
    script = load_script(ch_num)
    if not script:
        return None

    title = script.get("title", f"Chapter {ch_num}")
    segments = script["segments"]

    if test_mode:
        # Only use first 10 segments for testing
        segments = segments[:10]
        print(f"  TEST MODE: using first 10 segments only")

    chunks = chunk_segments(segments, voices, client.max_chars)
    total_chunks = len(chunks)

    print(f"  Ch {ch_num}: '{title}' → {len(segments)} segments → {total_chunks} chunks")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_parts = []

    failed_chunks = []

    for i, chunk in enumerate(chunks, 1):
        chars = sum(len(s['text']) for s in chunk)
        print(f"    [{i}/{total_chunks}] {chars} chars, "
              f"{len(chunk)} segments...", end="", flush=True)

        # Retry logic: up to 3 attempts with backoff
        audio_bytes = None
        last_error = None
        for attempt in range(1, 4):
            try:
                audio_bytes = client.generate(chunk)
                if audio_bytes and len(audio_bytes) > 0:
                    break  # success
            except Exception as e:
                last_error = str(e)
                if attempt < 3:
                    wait = attempt * 10  # 10s, 20s backoff
                    print(f" retry in {wait}s...", end="", flush=True)
                    time.sleep(wait)

        if audio_bytes and len(audio_bytes) > 0:
            audio_parts.append(audio_bytes)
            print(f" ✓ ({len(audio_bytes):,} bytes)")
        else:
            failed_chunks.append(i)
            print(f" ✗ FAILED after 3 attempts: {last_error[:120] if last_error else 'unknown'}")

        if isinstance(client, ElevenLabsProvider) and i < total_chunks:
            time.sleep(PAUSE_BETWEEN_CALLS)

    if failed_chunks:
        print(f"\n  ⚠ WARNING: {len(failed_chunks)} chunks FAILED: {failed_chunks}")
        print(f"  The audio file will have GAPS. Re-run to retry failed chunks.")
        print(f"  Failed chunk numbers: {failed_chunks} of {total_chunks}")

        # Save a manifest of what succeeded and what failed
        manifest = {
            "chapter": ch_num,
            "total_chunks": total_chunks,
            "succeeded": [i for i in range(1, total_chunks+1) if i not in failed_chunks],
            "failed": failed_chunks,
            "complete": len(failed_chunks) == 0,
        }
        manifest_path = OUTPUT_DIR / f"ch_{ch_num:02d}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    if not audio_parts:
        print(f"  No audio generated for Ch {ch_num}")
        return None

    # Concatenate all audio parts
    combined = b"".join(audio_parts)

    suffix = "_test" if test_mode else ""
    out_path = OUTPUT_DIR / f"ch_{ch_num:02d}{suffix}.{client.output_format}"
    out_path.write_bytes(combined)

    size_mb = len(combined) / (1024 * 1024)
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    return str(out_path)


def list_voices(client):
    """List available voices."""
    if isinstance(client, ElevenLabsProvider):
        response = client.voices.get_all()
        print(f"\n{'='*60}")
        print(f"AVAILABLE VOICES ({len(response.voices)})")
        print(f"{'='*60}")
        for voice in response.voices:
            labels = voice.labels or {}
            accent = labels.get("accent", "")
            age = labels.get("age", "")
            gender = labels.get("gender", "")
            desc = labels.get("description", "")
            use_case = labels.get("use_case", "")
            print(f"\n  {voice.name}")
            print(f"    ID: {voice.voice_id}")
            print(f"    {gender} | {age} | {accent}")
            if desc:
                print(f"    {desc}")
            if use_case:
                print(f"    Use case: {use_case}")
    else:
        print("Voice listing is only supported for the ElevenLabs provider.")
        print("For local providers, refer to their documentation for available voice IDs.")


def assemble_full_audiobook():
    """Concatenate all chapter audio files into one."""
    chapter_files = sorted(OUTPUT_DIR.glob("ch_*.mp3"))
    if not chapter_files:
        chapter_files = sorted(OUTPUT_DIR.glob("ch_*.wav"))
    chapter_files = [f for f in chapter_files if "_test" not in f.name]

    if not chapter_files:
        print("No chapter audio files found.")
        return

    print(f"\nAssembling {len(chapter_files)} chapters into full audiobook...")

    if chapter_files[0].suffix == ".wav":
        out = AUDIO_DIR / "full_audiobook.wav"
        with wave.open(str(out), "wb") as writer:
            initialized = False
            for f in chapter_files:
                with wave.open(str(f), "rb") as reader:
                    if not initialized:
                        writer.setparams(reader.getparams())
                        initialized = True
                    elif reader.getparams()[:3] != writer.getparams()[:3]:
                        raise RuntimeError("Chapter WAV files have inconsistent parameters")
                    writer.writeframes(reader.readframes(reader.getnframes()))
        size_mb = out.stat().st_size / (1024 * 1024)
    else:
        combined = b""
        # Add 2 seconds of silence between chapters (simple approach: just concatenate)
        for f in chapter_files:
            combined += f.read_bytes()
        out = AUDIO_DIR / "full_audiobook.mp3"
        out.write_bytes(combined)
        size_mb = len(combined) / (1024 * 1024)
    print(f"  Full audiobook: {out} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Generate audiobook from parsed scripts")
    parser.add_argument("start", nargs="?", type=int, help="Start chapter")
    parser.add_argument("end", nargs="?", type=int, help="End chapter")
    parser.add_argument("--provider", default="pocket-tts", choices=["pocket-tts", "elevenlabs", "kokoro", "piper", "mlx", "kittentts", "openai"], help="TTS provider")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--test", type=int, metavar="CH", help="Test mode: first few segments of chapter")
    parser.add_argument("--assemble", action="store_true", help="Assemble full audiobook from chapters")
    parser.add_argument("--status", action="store_true", help="Show generation status for all chapters")

    args = parser.parse_args()

    client = get_client(args.provider)

    if args.list_voices:
        list_voices(client)
        return

    if args.assemble:
        assemble_full_audiobook()
        return

    if args.status:
        print(f"\n{'='*50}")
        print("AUDIOBOOK GENERATION STATUS")
        print(f"{'='*50}")
        scripts = sorted(SCRIPTS_DIR.glob("ch*_script.json"))
        for script_f in scripts:
            ch_num = int(script_f.stem.replace("_script", "").replace("ch", ""))
            audio_f = OUTPUT_DIR / f"ch_{ch_num:02d}.mp3"
            manifest_f = OUTPUT_DIR / f"ch_{ch_num:02d}_manifest.json"
            if audio_f.exists():
                size_mb = audio_f.stat().st_size / (1024*1024)
                if manifest_f.exists():
                    m = json.loads(manifest_f.read_text())
                    if m.get("failed"):
                        print(f"  Ch {ch_num:2d}: ⚠ PARTIAL ({size_mb:.1f} MB, chunks {m['failed']} failed)")
                    else:
                        print(f"  Ch {ch_num:2d}: ✓ complete ({size_mb:.1f} MB)")
                else:
                    print(f"  Ch {ch_num:2d}: ✓ ({size_mb:.1f} MB)")
            else:
                print(f"  Ch {ch_num:2d}: ✗ not generated")
        return

    voices = load_voices(args.provider)
    if not voices:
        print(f"ERROR: No voices configured for provider '{args.provider}' in audiobook_voices.json")
        print("Run: python gen_audiobook.py --list-voices")
        sys.exit(1)

    if args.test:
        generate_chapter(args.test, client, voices, test_mode=True)
        return

    # Determine chapter range
    scripts = sorted(SCRIPTS_DIR.glob("ch*_script.json"))
    total = len(scripts)

    start = args.start or 1
    end = args.end or total

    print(f"Generating audiobook using {args.provider}: chapters {start}-{end}")
    print(f"  Voices configured: {list(voices.keys())}")
    print()

    for ch_num in range(start, end + 1):
        generate_chapter(ch_num, client, voices)
        print()

    # Assemble
    assemble_full_audiobook()


if __name__ == "__main__":
    main()
