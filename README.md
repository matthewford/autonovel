# autonovel

An autonomous pipeline for writing, revising, typesetting, illustrating,
and narrating a complete novel. From a seed concept to a manuscript and
print-ready PDF, with optional AI-generated art and audiobook assets.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
the same modify-evaluate-keep/discard loop, applied to fiction.

**First novel produced:** *The Second Son of the House of Bells* —
19 chapters, 79,456 words.
See the `autonovel/bells` branch.

---

## Quick Start

```bash
# Clone and setup
git clone <repo-url> && cd autonovel
cp .env.example .env    # Add your API keys

# Install dependencies
uv sync

# Default generation backend is Hermes via book.yaml/.env:
# AUTONOVEL_LLM_BACKEND=hermes
# AUTONOVEL_HERMES_PROVIDER=zai
# AUTONOVEL_HERMES_MODEL=glm-5.1
# AUTONOVEL_HERMES_USE_GOALS=true

# Generate a seed concept (or write your own in seed.txt)
uv run python seed.py

# Create a dedicated branch and update book.yaml for this book
git checkout -b autonovel/<book-slug>

# Run the full pipeline
uv run python run_pipeline.py --from-scratch
```

---

## The Pipeline

### Phase 1: Foundation
Build the world, characters, outline, voice, and canon from a seed concept.
Loop until `foundation_score > 7.5`.

### Phase 2: First Draft
Write chapters sequentially. Evaluate each one. Keep if `score > 6.0`,
retry if not. Forward progress over perfection.

### Phase 3a: Automated Revision
Adversarial editing → apply cuts → reader panel → generate briefs →
rewrite chapters. Plateau detection stops the loop when scores stabilize.

### Phase 3b: Opus Review Loop
Send the full manuscript to Claude Opus for dual-persona review
(literary critic + professor of fiction). Parse actionable items.
Fix the top issues. Repeat until the reviewer runs out of major items.

### Phase 4: Export
Rebuild docs, typeset in LaTeX, and build `manuscript.md`. When optional
providers are configured, export also generates art and audiobook assets.

See [PIPELINE.md](PIPELINE.md) for the full technical specification.

---

## Model Backend

Generation is routed through `llm.py`, configured by `book.yaml` and
environment overrides. The repo supports:

```bash
AUTONOVEL_LLM_BACKEND=hermes     # use `hermes chat`
AUTONOVEL_LLM_BACKEND=anthropic  # direct Anthropic Messages API
```

For Hermes, set provider/model once in `book.yaml` or `.env`. When a
subscription-backed model such as SuperGrok is available in Hermes, swap only
the provider/model/profile settings:

```bash
AUTONOVEL_HERMES_PROVIDER=<provider>
AUTONOVEL_HERMES_MODEL=<model>
AUTONOVEL_HERMES_PROFILE=        # optional; leave blank unless created
AUTONOVEL_HERMES_TOOLSETS=file,safe
AUTONOVEL_HERMES_USE_GOALS=true  # use Hermes /goal auto-continuation
```

Role-specific overrides are also supported:
`AUTONOVEL_HERMES_WRITER_MODEL`, `AUTONOVEL_HERMES_JUDGE_MODEL`,
`AUTONOVEL_HERMES_REVIEWER_MODEL`, and `AUTONOVEL_HERMES_READER_MODEL`.

### Hermes profile

The default `book.yaml` can use a Hermes profile named `autonovel`:

```yaml
generation:
  backend: hermes
  use_goals: true
  writer:
    profile: autonovel
```

Create or inspect the profile with:

```bash
hermes profile create autonovel --clone \
  --description "Specialized fiction-generation backend for /home/hermes/autonovel."

hermes profile show autonovel
```

The profile keeps AutoNovel's Hermes sessions, memory, skills, logs, and goal
state separate from the default assistant profile. Its `SOUL.md` should describe
the role as a dedicated long-form fiction backend that follows `program.md`,
`CRAFT.md`, `ANTI-SLOP.md`, `ANTI-PATTERNS.md`, and the current book files.

Before starting a new book branch, update `book.yaml` with the new slug, title,
target chapter count, and role-specific provider/model settings.

### Hermes goals

When `AUTONOVEL_HERMES_USE_GOALS=true` or `generation.use_goals: true`,
`llm.py` prefixes Hermes backend calls with `/goal`. Hermes then auto-continues
the same session until the goal judge sees the sentinel-wrapped output:

```text
<<<AUTONOVEL_OUTPUT>>>
...
<<<END_AUTONOVEL_OUTPUT>>>
```

This is useful for long creative calls because a single Hermes turn can stop
before a full chapter, review, or planning document is complete. Goal-backed
calls still return only the parsed text between the sentinels to the pipeline.
Disable it with:

```bash
AUTONOVEL_HERMES_USE_GOALS=false
```

or per role:

```bash
AUTONOVEL_HERMES_WRITER_USE_GOALS=false
```

---

## Tools (27 Python scripts)

### Foundation
| Tool | Purpose |
|------|---------|
| `seed.py` | Generate seed concepts |
| `gen_world.py` | Seed → world bible |
| `gen_characters.py` | Seed + world → character registry |
| `gen_outline.py` | Outline with beats and foreshadowing |
| `gen_outline_part2.py` | Foreshadowing ledger |
| `gen_canon.py` | Cross-reference hard facts |
| `voice_fingerprint.py` | Voice analysis and discovery |

### Drafting
| Tool | Purpose |
|------|---------|
| `draft_chapter.py` | Write a single chapter with anti-pattern rules |
| `run_drafts.py` | Batch sequential chapter drafter |

### Evaluation
| Tool | Purpose |
|------|---------|
| `evaluate.py` | Mechanical slop scorer + LLM judge |
| `adversarial_edit.py` | "Cut 500 words" analysis → classified cuts |
| `compare_chapters.py` | Head-to-head Elo tournament |
| `reader_panel.py` | 4-persona novel-level evaluation |
| `review.py` | Opus dual-persona review with stopping conditions |

### Revision
| Tool | Purpose |
|------|---------|
| `gen_brief.py` | Auto-generate revision briefs from feedback |
| `gen_revision.py` | Rewrite a chapter from a revision brief |
| `apply_cuts.py` | Batch adversarial cut applicator |

### Art & Cover
| Tool | Purpose |
|------|---------|
| `gen_art.py` | Art pipeline: style, curate, ornaments, vectorize |
| `gen_art_directions.py` | Generate diverse art directions for curation |
| `gen_cover_composite.py` | Text overlay on cover art |
| `gen_cover_print.py` | Print-ready full-wrap cover (Lulu/KDP specs) |

### Audiobook
| Tool | Purpose |
|------|---------|
| `gen_audiobook_script.py` | Parse chapters into speaker-attributed scripts |
| `gen_audiobook.py` | Generate audio via local Pocket TTS by default, or optional cloud/local providers |

### Orchestration
| Tool | Purpose |
|------|---------|
| `run_pipeline.py` | Full pipeline orchestrator (seed → finished novel) |
| `build_arc_summary.py` | Regenerate arc summary from chapters |
| `build_outline.py` | Regenerate outline from chapters |

---

## File Structure

```
FRAMEWORK (reusable, on master):
  program.md             — Agent instructions per phase
  CRAFT.md               — Craft education (plot, character, world, prose)
  ANTI-SLOP.md           — Word-level AI tell detection
  ANTI-PATTERNS.md       — Structural AI pattern detection
  PIPELINE.md            — Full automation specification
  WORKFLOW.md            — Step-by-step human guide

TEMPLATES (filled per-novel on a branch):
  voice.md               — Part 1: guardrails. Part 2: discovered per novel
  world.md               — World bible template
  characters.md          — Character registry template
  outline.md             — Chapter outline template
  canon.md               — Hard facts database
  MYSTERY.md             — Central mystery (author-only)
  state.json             — Pipeline state tracker

TYPESETTING:
  typeset/novel.tex      — LaTeX template (EB Garamond, trade paperback)
  typeset/build_tex.py   — Chapters → LaTeX with vector ornaments
  typeset/epub_*          — ePub metadata, CSS, and front matter

ART:
  audiobook_voices.json  — Character → ElevenLabs voice mapping
  landing/index.html     — Responsive landing page template

CONFIG:
  book.yaml              — Book profile and role-specific generation settings
  .env.example           — API keys (Anthropic, fal.ai, ElevenLabs)
  book_profile.py        — Profile parser/defaults
  llm.py                 — Anthropic/Hermes model gateway
  pyproject.toml         — Python dependencies
```

---

## How It Works

The novel is five co-evolving layers:

```
  Layer 5:  voice.md          — HOW we write
  Layer 4:  world.md          — WHAT exists
  Layer 3:  characters.md     — WHO acts
  Layer 2:  outline.md        — WHAT HAPPENS
  Layer 1:  chapters/ch_NN.md — THE ACTUAL PROSE
  Cross-cutting: canon.md     — WHAT IS TRUE
```

Changes propagate both down (lore change → outline change → chapter
revision) and up (writing reveals a gap → update lore → check
downstream). The pipeline tracks propagation debts in `state.json`.

### Two Immune Systems

1. **Mechanical** (`evaluate.py`, no LLM): regex scans for banned words,
   fiction clichés, show-don't-tell violations, sentence uniformity.

2. **LLM Judge** (`evaluate.py`, separate model): scores prose quality,
   voice adherence, character distinctiveness, beat coverage.

### The Opus Review Loop

After automated revision cycles, the full manuscript goes to Claude Opus
with this prompt:

> "Read the below novel. Review it first as a literary critic and then
> as a professor of fiction. Give specific, actionable suggestions for
> any defects you find. Be fair but honest. You don't have to find defects."

The dual-persona review catches what automated tools can't: prose-level
repetition, character thinness, ethical gaps, structural monotony. The
loop continues until the reviewer's items are mostly qualified hedges rather than real problems.

---

## API Keys

The pipeline uses three external services:

| Service | Key | Used for |
|---------|-----|----------|
| Anthropic | `ANTHROPIC_API_KEY` | Writing, evaluation, review (Sonnet + Opus) |
| fal.ai | `FAL_KEY` | Cover art and ornament generation (Nano Banana 2) |
| ElevenLabs | `ELEVENLABS_API_KEY` | Multi-voice audiobook generation |

Copy `.env.example` to `.env` and fill in your keys. Only the Anthropic
key is required for the core pipeline. Art and audiobook are optional.

---

## Production History

The first novel, *The Second Son of the House of Bells*, was produced
through this pipeline:

- **Foundation:** World bible, 8 characters, 24-chapter outline, voice discovery
- **Drafting:** 24 chapters, 75,698 words, sequential with evaluation
- **Revision:** 6 automated cycles + 6 Opus review rounds
- **Structural:** 24 → 19 chapters through 4 merges
- **Art:** Linocut cover (Nano Banana 2), 19 woodcut chapter ornaments (vectorized)
- **Audiobook:** 19 chapters parsed into 4,179 speaker-attributed segments
- **Final:** 79,456 words, 6 review rounds, all major items resolved

---

## Inspiration

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the autonomous research loop
- Brandon Sanderson's writing lectures (Laws of Magic, character sliders)
- K.M. Weiland's *Creating Character Arcs*
- Blake Snyder's *Save the Cat*
- Ursula K. Le Guin's "From Elfland to Poughkeepsie"
- [slop-forensics](https://github.com/sam-paech/slop-forensics) and [EQ-Bench Slop Score](https://eqbench.com/slop-score.html)
