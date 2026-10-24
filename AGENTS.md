# claude-video / watch skill

Agent Skills package that gives an agent a video input. Installable across Claude Code (most common host), Codex, Cursor, GitHub Copilot, and 50+ other [Agent Skills](https://agentskills.io) hosts. Pure-stdlib Python that orchestrates `yt-dlp` + `ffmpeg` and an optional Whisper API, local `mlx_whisper` CLI (Apple Silicon), or separately managed local WhisperX process.

## Structure

- `skills/watch/SKILL.md` — canonical skill contract the model reads when `/watch` fires. Source of truth for behavior across every host.
- `skills/watch/scripts/watch.py` — entry point; orchestrates download → frames → transcript.
- `skills/watch/scripts/{download,frames,transcribe,whisper,local_whisper,local_whisperx,verify,setup,config,runtime}.py` — yt-dlp wrapper (+ post caption metadata), ffmpeg frame extraction + auto-fps + tiled dedup, caption parsing, Whisper backend selection + Groq/OpenAI clients, local `mlx_whisper` runner + repairs, WhisperX adapter, transcript verification, preflight/installer, shared config, subprocess helpers.
- `skills/watch/scripts/build-skill.sh` — builds `dist/watch.skill` for claude.ai upload (dev-only).
- `hooks/` — Claude Code SessionStart setup-status hook (Claude Code only).
- `.claude-plugin/` — `plugin.json` + `marketplace.json` (Claude Code plugin + local marketplace).
- `.codex-plugin/plugin.json` — Codex/agents manifest; `"skills": "./skills/"` points the Agent Skills CLI at the self-contained skill folder.
- `.agents/plugins/marketplace.json` — agents marketplace listing pointing at the repo-root plugin.
- `CLAUDE.md` → `@AGENTS.md` — generic-agent entry point.
- `tests/` — pytest suite (ffmpeg-synthesized clips; no network).

## Orientation

- The product is the slash-command-invoked skill (`/watch <url-or-path> [question]`), not a CLI. `scripts/watch.py` is implementation. Features must work across every harness the skill installs into, not just Claude Code.
- **The skill is one self-contained folder: `skills/watch/`.** SKILL.md and `scripts/` are siblings inside it. This is what lets `npx skills add` copy a working skill as a unit — do NOT move SKILL.md or `scripts/` back to the repo root, or non-Claude installers will copy SKILL.md without the scripts.
- **Path resolution is harness-agnostic.** SKILL.md resolves `SKILL_DIR` as the directory of the SKILL.md the model just Read, then runs `${SKILL_DIR}/scripts/...`. Do NOT reintroduce `${CLAUDE_SKILL_DIR}` (Claude-Code-only) — it is unset on Codex/Cursor/agents and breaks every script call there.
- **No `commands/` wrapper.** `/watch` comes from `name: watch` (skills are user-invocable by default). A separate command file creates a duplicate slash command.
- **SKILL.md frontmatter uses only Agent Skills keys:** `name`, `description`, `license`, `allowed-tools`, `metadata`, `compatibility`. Claude's skill upload rejects anything else; the version lives in `metadata.version`.

## Install surfaces

| Surface | Install |
|---------|---------|
| Claude Code | `/plugin marketplace add bradautomates/claude-video` then `/plugin install watch@claude-video` |
| Codex / Cursor / Copilot / +50 | `npx skills add bradautomates/claude-video -g` |
| claude.ai (web) | upload `dist/watch.skill` (built by `skills/watch/scripts/build-skill.sh`) |

## Commands

```bash
# Tests (stdlib + pytest; ffmpeg required for frame tests)
.venv/bin/pytest -q                # or: python3 -m pytest -q

# Build the claude.ai upload bundle (archives skills/watch/ as the bundle root)
bash skills/watch/scripts/build-skill.sh   # → dist/watch.skill

# Dev: mirror the working tree into the installed Claude Code plugin cache
./dev-sync.sh                       # --dry-run to preview
```

## Rules

- Keep the version in sync across `skills/watch/SKILL.md` (`metadata.version`), `.claude-plugin/plugin.json`, and `.codex-plugin/plugin.json` when cutting a release.
- Releasing: tag `vX.Y.Z` and push the tag; `.github/workflows/release.yml` builds `dist/watch.skill` and attaches it to the GitHub release.
- Never commit real API keys or `.env` contents; keys live in `~/.config/watch/.env` (mode `0600`) at runtime.
- Whisper backend order is `auto` = local `mlx` when `mlx_whisper` is on PATH (or `~/.local/bin`) → Groq → OpenAI; WhisperX is explicit-only. Local-backend and verification code lives in `local_whisper.py` / `verify.py` so upstream files carry only thin hooks (keeps rebases onto `bradautomates/claude-video` small). Verification results ride on `segments.quality`; `transcribe_video` keeps its `(segments, backend)` shape. Tests must not depend on the developer's machine having `mlx_whisper`: `tests/conftest.py` replaces `local_whisper.mlx_path` for every test, and tests that need a CLI install a fake one and request the `real_mlx_lookup` fixture (`tests/test_local_whisper.py`) or patch `mlx_path` (`test_whisper.py`, `test_setup.py`, `test_watch.py`).
- Frame dedup is per-tile (`DEDUP_GRID`), not whole-frame. Two slides on one template differ by a whole-frame mean of ~2/255, so a whole-frame threshold deletes decks — `tests/test_dedup.py::test_dedupe_perceptual_keeps_every_slide` guards this.
- Video downloads sort with `-S res:720` (yt-dlp `res` is the smaller dimension), never a `height<=720` filter, which caps 9:16 reels at 360x640.
