---
name: watch
version: "0.2.0"
description: Watch a video (URL or local path). Downloads with yt-dlp, extracts auto-scaled frames with ffmpeg, pulls the transcript from captions (or Whisper fallback — local mlx-whisper on Apple Silicon, else Groq/OpenAI API), and hands the result to Claude so it can answer questions about what's in the video.
argument-hint: "<video-url-or-path> [question]"
allowed-tools: Bash, Read, AskUserQuestion
homepage: https://github.com/bradautomates/claude-video
repository: https://github.com/bradautomates/claude-video
author: bradautomates
license: MIT
user-invocable: true
---

# /watch

You don't have a video input; this skill gives you one. A Python script gets captions first, optionally downloads the video, extracts frames as JPEGs (scene-aware, or fast keyframes at `efficient` detail), gets a timestamped transcript (native captions first, then Whisper as fallback — locally on the Apple Silicon GPU when `mlx_whisper` is installed, otherwise the Groq/OpenAI API), and prints frame paths. You then `Read` each frame path to see the images and combine them with the transcript to answer the user.

## Resolve `SKILL_DIR` (do this before any command)

Every `python3 ...` command below runs a bundled script under `SKILL_DIR/scripts/`. Set `SKILL_DIR` to the **absolute path of the directory containing THIS SKILL.md you just Read** — your harness told you that path in the Read result. The scripts are always a direct sibling of this file (`SKILL_DIR/scripts/watch.py`), in every install layout:

```
Read ~/.claude/plugins/cache/claude-video/watch/<ver>/skills/watch/SKILL.md → SKILL_DIR=…/skills/watch
Read ~/.codex/skills/watch/SKILL.md                                          → SKILL_DIR=~/.codex/skills/watch
Read ~/.agents/skills/watch/SKILL.md                                         → SKILL_DIR=~/.agents/skills/watch
```

Substitute that literal path for `${SKILL_DIR}` in every command. This works on every harness (Claude Code, Codex, Cursor, Gemini CLI, …) without relying on any harness-specific environment variable. Guard once at the start of a run:

```bash
SKILL_DIR="<absolute path of the directory containing the SKILL.md you Read>"
if [ ! -f "$SKILL_DIR/scripts/watch.py" ]; then
  echo "ERROR: scripts/watch.py not found under SKILL_DIR=$SKILL_DIR" >&2
  echo "Re-check the directory of the SKILL.md you Read and substitute it as SKILL_DIR." >&2
  exit 1
fi
```

## Step 0 — Setup preflight (runs every `/watch` invocation, silent on success)

**Python interpreter:** every `python3 ...` command in this skill is for macOS/Linux. On **Windows**, substitute `python` — the `python3` command on Windows is the Microsoft Store stub and will not run the script.

On the first `/watch` invocation in a session, use structured preflight so you can detect first-run setup:

```bash
python3 "${SKILL_DIR}/scripts/setup.py" --json
```

Branch on two fields:

- **`can_proceed: true` and `first_run: false`** → setup is already done (the user may have deliberately skipped Whisper — that's allowed). Proceed to Step 1 without comment — **unless** `whisper_backend` is `"mlx"` and `mlx_model_cached` is `false`: then run `setup.py --warm` first (see "pre-fetch the model" below). This applies regardless of `first_run`; a user who installed `mlx_whisper` for another project has the CLI but not this model.
- **`first_run: true`** → genuine first-time setup. Do these in order:
  1. If `missing_binaries` is non-empty, run the installer first (it auto-installs on macOS / prints commands elsewhere — see below) and confirm the binaries land. **Do not skip this and jump to preferences.**
  2. Run the installer once more if needed so it scaffolds `~/.config/watch/.env` (it only writes the template when the file is absent, so let it create the file *before* you write any values into it).
  3. Encourage a Whisper backend (see below) and ask the watch-preference questions, then write the selected values into `~/.config/watch/.env` and set `SETUP_COMPLETE=true`.
- **`can_proceed: false` and `first_run: false`** → setup was finished before but the environment regressed (e.g. `missing_binaries` after an OS change). Run the installer to remediate, then proceed. Don't re-ask preferences.

A missing Whisper backend is *encouraged to fix, not required*: on a genuine first run `status` will read `needs_key` when binaries are present but neither `mlx_whisper` nor an API key is — that's your cue to encourage one, not a blocker. When `has_mlx_whisper: true`, Whisper is already fully set up with no key (`whisper_backend: "mlx"`); do not ask for a key.

On follow-up `/watch` calls in the same session, use the silent check:

```bash
python3 "${SKILL_DIR}/scripts/setup.py" --check
```

This is a <100ms lookup. Exit 0 means /watch can run — this **includes a user who finished setup without a Whisper key** (keyless is allowed). On exit 0 the script emits **nothing** — proceed to Step 1 without comment. **Do NOT announce "setup is complete" to the user** — they don't need a status message on every turn. The only acceptable user-visible output from Step 0 is when remediation is required.

On non-zero exit, follow the table:

| Exit | Meaning | Action |
|------|---------|--------|
| `2` | Missing binaries (`ffmpeg` / `ffprobe` / `yt-dlp`) | Run installer |
| `3` | Genuine first run with no Whisper backend (no `mlx_whisper`, no API key) | Run installer to scaffold `.env`, then encourage a backend (the user may decline — proceed with `--no-whisper`) |
| `4` | Both missing | Run installer, then encourage a backend |

Exit `3` only fires before the user has completed setup. Once `SETUP_COMPLETE=true` is written, an install with no Whisper backend returns exit 0 and is never nagged again.

The installer is idempotent — safe to re-run:

```bash
python3 "${SKILL_DIR}/scripts/setup.py"
```

On macOS with Homebrew, it auto-installs `ffmpeg` and `yt-dlp`. On Linux/Windows, it prints the exact install commands for the user to run. It scaffolds `~/.config/watch/.env` with commented placeholders and default watch settings at `0600` perms.

**If no Whisper backend is available after install** (`whisper_backend: null` in `--json`), use `AskUserQuestion` to offer the choices — order them by what the machine supports:

- **`apple_silicon: true`** → lead with the **local** option: `brew install pipx && pipx install mlx-whisper` (or `uv tool install mlx-whisper`). Free, no key, the audio never leaves the machine. If they accept, run the install for them (no sudo needed), then re-run `setup.py --json` and confirm `has_mlx_whisper: true` — the script finds the binary in `~/.local/bin` even before `pipx ensurepath` takes effect. Offer the API keys as the alternative.
- **Otherwise** → ask whether they have a Groq API key (preferred — cheaper, faster) or an OpenAI key, and write it into `~/.config/watch/.env` as the matching `GROQ_API_KEY=...` or `OPENAI_API_KEY=...` line.
- If they don't want to set up Whisper at all, proceed with `--no-whisper` and tell them videos without native captions will come back frames-only.

**First use of the local backend — pre-fetch the model.** When `--json` shows `has_mlx_whisper: true` and `mlx_model_cached: false`, the first transcription would download the ~1.5 GB `whisper-large-v3-turbo` weights *inside* the watch run, silently, and a default tool timeout kills it. Instead, run this once, with a long timeout (10 minutes) or in the background, and tell the user it is a one-time download:

```bash
python3 "${SKILL_DIR}/scripts/setup.py" --warm
```

It is safe to re-run: with the model already cached it finishes in a couple of seconds, and it resumes an interrupted download.

The user can pin a backend with `WATCH_WHISPER=mlx|groq|openai` in `~/.config/watch/.env` (default `auto` = `mlx` when installed, else Groq, else OpenAI). Only write this line if they express a preference.

**First-run watch preference:** after the installer has scaffolded `~/.config/watch/.env`, use `AskUserQuestion` to ask one question:

- Default detail (one dial). Present these as `AskUserQuestion` options in this exact order — lightest to heaviest — and keep `(recommended)` on `balanced` even though it is not first (do **not** reorder to put the recommended option first):
  - `transcript` — no frames at all, transcript only (skips video download when captions exist).
  - `efficient` — fast keyframe pass (cap 50).
  - `balanced` (recommended) — scene-aware frames (cap 100, default).
  - `token-burner` — scene-aware, uncapped (maximum fidelity; high token cost).

Write the answer directly into `~/.config/watch/.env` by setting the bare key on its own line — **no trailing inline comment** (a `# note` after the value can break parsing):

```bash
WATCH_DETAIL=balanced
```

Use the user's selected value. If they skip the question, keep the recommended default. Once dependencies, the Whisper-backend choice, and this preference are handled, write or update `SETUP_COMPLETE=true` in the same file. Do not ask this preference question again when `SETUP_COMPLETE=true`.

**Structured mode (optional):** `python3 "${SKILL_DIR}/scripts/setup.py" --json` emits `{status, can_proceed, first_run, setup_complete, missing_binaries, whisper_backend, has_api_key, has_mlx_whisper, mlx_model_cached, apple_silicon, config_file, watch_detail, platform}` where `status` is one of `ready | needs_install | needs_key | needs_install_and_key` and `whisper_backend` is `mlx | groq | openai | null` (what a run would use right now). `status` describes the *ideal* state (a backend is encouraged, so a first run with none reads `needs_key`); `can_proceed` is the operational gate (binaries present AND a backend is available OR setup was already completed). Branch on `can_proceed`/`first_run` to decide whether to run; use `status` to decide what to encourage.

Within a single session, you can skip Step 0 on follow-up `/watch` calls — once `--check` returned 0, nothing about the environment changes between turns.

## When to use

- User pastes a video URL (YouTube, Vimeo, X, TikTok, Twitch clip, most yt-dlp-supported sites) and asks about it.
- User points at a local video file (`.mp4`, `.mov`, `.mkv`, `.webm`, etc.) and asks about it.
- User types `/watch <url-or-path> [question]`.

## Recommended limits

- **Best accuracy: videos under 10 minutes.** Frame coverage scales inversely with duration.
- **Universal rate cap: 2 fps.** The script never samples faster than 2 fps, even when a budget or `--fps` would imply more.
- **The frame ceiling is set by the detail mode** (`WATCH_DETAIL` in `~/.config/watch/.env`, or `--detail`), not a single global cap:
  - `transcript` → no frames
  - `efficient` → up to **50** (keyframes)
  - `balanced` (default) → up to **100** (scene-aware)
  - `token-burner` → **uncapped** (scene-aware; a soft warning prints past 250 frames)
  - `--max-frames N` overrides whichever cap the mode would otherwise use.
- **Full-video frame budget by duration.** Token cost grows with frame count, so the script targets a budget by duration. This budget sets the fps and the uniform-sampling fallback; scene-aware selection can fill up to the detail cap above, whichever is lower:
  - ≤30s → ~12-30 frames
  - 30s-1min → ~40 frames
  - 1-3min → ~60 frames
  - 3-10min → ~80 frames
  - \>10min → up to the detail cap, sparsely spaced (warning printed)
- If the user hands you a long video, consider asking whether they want a specific section before burning tokens on a sparse scan.

## How to invoke

**Step 1 — parse the user input.** Separate the video source (URL or path) from any question the user asked. Example: `/watch https://youtu.be/abc what language is this in?` → source = `https://youtu.be/abc`, question = `what language is this in?`.

**Step 2 — run the watch script.** Pass the source verbatim. Do not shell-escape it yourself beyond normal quoting. **Set a long tool timeout when Whisper will run** (no captions: any local file, and many short-form URLs): the local `mlx` backend transcribes at roughly 10-20× real time, so a 30-minute recording needs 2-3 minutes — beyond the default 2-minute Bash timeout. Use a 10-minute timeout, or run in the background for anything over ~45 minutes.

```bash
python3 "${SKILL_DIR}/scripts/watch.py" "<source>"
```

Optional flags:
- `--detail transcript|efficient|balanced|token-burner` — fidelity/speed dial. `transcript` = no frames (transcript only, skips video download when captions exist); `efficient` = fast keyframes (cap 50); `balanced` = scene-aware frames (cap 100); `token-burner` = scene-aware, uncapped.
- `--start T` / `--end T` — focus on a section. Accepts `SS`, `MM:SS`, or `HH:MM:SS`. When either is set, fps auto-scales denser (see "Focusing on a section" below).
- `--timestamps T1,T2,…` — grab a frame at each of these absolute timestamps (`SS`, `MM:SS`, or `HH:MM:SS`). Use this after reading the transcript to capture deictic moments the presenter flags ("look here", "as you can see", "notice this") that visual selection alone may miss. See "Transcript-cue frames" below.
- `--max-frames N` — override the preset cap for tighter token budget (e.g. `--max-frames 40`)
- `--resolution W` — change frame width in px (default 512; bump to 1024 only if the user needs to read on-screen text)
- `--fps F` — override auto-fps (clamped to 2 fps max)
- `--out-dir DIR` — keep working files somewhere specific (default: an auto-generated tmp dir)
- `--whisper mlx|groq|openai` — force a specific Whisper backend (default: `WATCH_WHISPER` from `.env`, else `auto` = local `mlx` when installed, then Groq, then OpenAI)
- `--no-whisper` — disable the Whisper fallback entirely (frames-only if no captions)
- `--no-dedup` — keep near-duplicate frames. By default a tiled frame-delta pass drops frames where *no region* changed versus the previous kept one (held slides, static screen recordings, paused video) so the frame budget goes to distinct content; the report's **Frames** line notes how many were dropped. A change confined to one region — a new bullet, a code diff, a dialog — is kept. Pass this only if the user needs every sampled frame (e.g. judging subtle frame-to-frame motion).

### Focusing on a section (higher frame rate)

When the user asks about a specific moment — "what happens at the 2 minute mark?", "zoom into 0:45 to 1:00", "the first 10 seconds" — pass `--start` and/or `--end`. The script switches to focused-mode budgets, which are denser than full-video budgets (still capped at 2 fps, and still bounded by the detail-mode cap — the counts below assume the default `balanced` cap of 100; `efficient` tops out at 50):

- ≤5s → 2 fps (up to 10 frames)
- 5-15s → 2 fps (up to 30 frames)
- 15-30s → ~2 fps (up to 60 frames)
- 30-60s → ~1.3 fps (up to 80 frames)
- 60-180s → ~0.6 fps (100 frames, capped)

Focused mode is the right call for:
- Any moment/range the user names explicitly ("around 2:30", "the intro", "the last 30 seconds").
- Any video longer than ~10 minutes where the user's question is about a specific part — running focused on the relevant section is far more useful than a sparse scan of the whole thing.
- Re-runs after a full scan didn't have enough detail in some region.

Transcript is auto-filtered to the same range. Frame timestamps are absolute (real video timeline, not offset-from-start).

Examples:
```bash
# Last 10 seconds of a 1 minute video
python3 "${SKILL_DIR}/scripts/watch.py" video.mp4 --start 50 --end 60

# Zoom into 2:15 → 2:45 at 2 fps (60 frames)
python3 "${SKILL_DIR}/scripts/watch.py" "$URL" --start 2:15 --end 2:45 --fps 2

# From 1h12m to the end of the video
python3 "${SKILL_DIR}/scripts/watch.py" "$URL" --start 1:12:00
```

**Step 3 — Read every frame path the script lists.** The Read tool renders JPEGs directly as images for you. Read all frames in a single message (parallel tool calls) so you see them together. The frames are in chronological order with a `t=MM:SS` timestamp so you can align them to the transcript.

**Step 4 — answer the user.** You now have two streams of evidence:
- **Frames** — what's on screen at each timestamp
- **Transcript** — what's said at each timestamp. The report's header shows the source (`captions` = yt-dlp pulled native subs; `whisper (mlx)` = transcribed locally on this machine; `whisper (groq)` or `whisper (openai)` = transcribed by API).

If the report contains a **"transcript verification failed"** warning block, repeat it to the user in plain words: Whisper produced a repetition loop in the listed windows, and the speech there was lost. Do not answer questions about those windows from the transcript alone — use the frames, and offer a re-run (`--whisper` with a different backend, or `--start/--end` around the window). A **"Note: no speech was recognised in the last Ns"** line is advisory — look at the frames for that span; only mention it if someone is visibly speaking there.

If the user asked a specific question, answer it directly citing timestamps. If they didn't ask anything, summarize what happens in the video — structure, key moments, notable visuals, spoken content.

This holds for `transcript` detail too: even with no frames, produce a **summary** like the other modes — do not paste the full transcript into chat. Synthesize structure, key moments, and spoken content with timestamps; quote only the lines that matter. Offer the raw transcript only if the user explicitly asks for it.

**Step 5 — clean up.** The script prints a working directory at the end. If the user isn't going to ask follow-ups about this video, delete it with `rm -rf <dir>`. If they might, leave it in place.

## Detail and frames

Default behavior comes from `~/.config/watch/.env`:

- `WATCH_DETAIL=transcript|efficient|balanced|token-burner` (default: `balanced`)

At `transcript` detail, captions are enough to return a report without downloading video. If captions are missing, the script downloads audio only and tries Whisper. If no transcript can be produced, it reports the limitation clearly; re-run with `--detail balanced` for frames.

At `efficient` detail, the script downloads the video and extracts **keyframes only** (`ffmpeg -skip_frame nokey`) — a near-instant pass that lands frames on scene cuts. If a clip has fewer than 4 keyframes it falls back to uniform sampling.

At `balanced` / `token-burner` detail, the script extracts **scene-aware** frames: ffmpeg scene-change selection first, falling back to uniform sampling only when the video is effectively static. `balanced` caps at 100 frames; `token-burner` is uncapped. Frame report lines include both timestamp and selection reason. Extracted images are clamped to a maximum 1998px height for Claude Read compatibility.

## Transcript-cue frames

Visual frame selection (scene/keyframe) can miss the moments a presenter explicitly flags — "look here", "as you can see", "notice this", "watch what happens" — because pointing at a slide is often a *low* visual change. `--timestamps` lets you force a frame at those exact moments. **You** decide which moments matter, by reading the transcript:

1. Run once at `--detail transcript` (or any detail) to get the timestamped transcript.
2. Scan it for deictic cues — phrases where the speaker directs attention to something on screen. This is a judgment call (ignore rhetorical "look, the point is…"); that's why it's done by you, not a regex.
3. Re-run with `--timestamps 4:32,7:10,9:55` (absolute source times). For a URL, point the second run at the **downloaded local file** in the work dir so it doesn't re-download.

Behavior:
- **Additive by default.** Cue frames (`reason=transcript-cue`) are merged into whatever `--detail` already selected, in chronological order.
- **Pinned and counted first.** Cue frames are reserved against the frame cap before the detail engine runs, so they're never evicted by even-sampling.
- **Honors focus mode.** With `--start/--end`, any cue timestamp outside the window is dropped (reported in the summary). Coordinates are always absolute source time.
- **Cue-only frames.** `--detail transcript --timestamps …` skips scene/keyframe sampling and returns *only* the cue frames (it will download the video to do so, since frames need pixels).

## Transcription

The script gets a timestamped transcript in one of two ways:

1. **Native captions (free, preferred).** yt-dlp pulls manual or auto-generated subtitles from the source platform if available.
2. **Whisper fallback.** If no captions came back (or the source is a local file), the script extracts mono 16 kHz audio with ffmpeg and transcribes it with the first available backend:
   - **mlx** (local) — `mlx-community/whisper-large-v3-turbo` via the `mlx_whisper` CLI on the Apple Silicon GPU. No key, no upload, no per-minute cost; roughly real-time ÷ 10 on an M-series Mac. Install: `pipx install mlx-whisper`. Runs with `--condition-on-previous-text False --temperature 0.2`, the flags that stop Whisper's repetition-loop failure.
   - **Groq** — `whisper-large-v3`. Preferred API: cheaper, faster. Audio (~0.5 MB/min mp3) is uploaded. Get a key at console.groq.com/keys.
   - **OpenAI** — `whisper-1`. API fallback. Get a key at platform.openai.com/api-keys.

Keys and the `WATCH_WHISPER` preference live in `~/.config/watch/.env`. Default order is mlx → Groq → OpenAI; override with `--whisper <backend>` for one run or `WATCH_WHISPER=<backend>` permanently. Use `--no-whisper` to skip the fallback entirely.

**Every Whisper transcript is verified** before it is trusted. Whisper's failures are silent — it can emit the same phrase for minutes while discarding the real speech (a decoder loop), and the result reads fluently. The script scans for runs of ≥3 identical consecutive segments; on the `mlx` backend, looped windows are automatically re-transcribed in isolation at a higher temperature and spliced back in. Anything still looping is printed as a `> **Warning: transcript verification failed**` block in the report listing the affected windows — surface it to the user (see Step 4). Separately, a transcript that ends >30 s before the audio does gets a `> **Note:**` — that is usually outro music or silence, not lost speech; check the frames for that span rather than warning the user.

When `auto` selected `mlx` and it fails at runtime (broken install, interrupted model download), the script falls back to a configured API key on its own and says so on stderr; the report header then shows `whisper (groq)` / `whisper (openai)`.

## Failure modes and handling

- **Setup preflight failed** → run `python3 "${SKILL_DIR}/scripts/setup.py"` (auto-installs ffmpeg/yt-dlp via brew on macOS, scaffolds the `.env`). For the Whisper backend, ask the user via `AskUserQuestion` (local `mlx-whisper` on Apple Silicon, or an API key written to `~/.config/watch/.env`).
- **No transcript available** → captions missing AND (no Whisper backend OR Whisper failed). Script prints a hint pointing to setup. Proceed frames-only and tell the user.
- **Transcript verification warning in the report** → Whisper looped in the listed windows. Tell the user; don't rely on the transcript for those windows. Re-run with a different `--whisper` backend, or focus on the window with `--start/--end`.
- **`mlx_whisper` fails** → its stderr is printed (common: first-run model download interrupted, or not Apple Silicon). Re-run, or fall back with `--whisper groq|openai` if a key is set.
- **Long video warning printed** → acknowledge it in your answer. Offer to re-run focused on a specific section via `--start`/`--end` rather than a sparse full-video scan.
- **Download fails** → yt-dlp's error goes to stderr. If it's a login-required or region-locked video, tell the user plainly; do not keep retrying.
- **Whisper request fails** → the error is printed to stderr (likely: invalid key or rate limit). Audio over the API's 25 MB upload cap is split into chunks and transcribed automatically, so length alone won't fail it; if some chunks fail the transcript is partial and the dropped chunks are noted on stderr. The report will say "none available" only if every chunk fails. You can retry with `--whisper openai` if Groq failed (or vice versa).

## Token efficiency

This skill burns tokens primarily on frames. Order of magnitude:
- 80 frames at 512px wide is roughly 50-80k image tokens depending on aspect ratio.
- The transcript is cheap (a few thousand tokens at most for a 10-minute video).
- Bumping `--resolution` to 1024 roughly quadruples the image tokens per frame. Only do it when necessary.

If you already watched a video this session and the user asks a follow-up, do **not** re-run the script — you already have the frames and transcript in context. Just answer from what you have.

## Security & Permissions

**What this skill does:**
- Runs `yt-dlp` locally to download the video and pull native captions when the source supports them (public data; the request goes directly to whatever host the URL points at)
- Runs `ffmpeg` / `ffprobe` locally to extract frames as JPEGs and, when Whisper is needed, a mono 16 kHz audio clip
- Runs `mlx_whisper` locally (Apple Silicon GPU) when it is installed and selected — the audio never leaves the machine. The first run downloads the ~1.5 GB `whisper-large-v3-turbo` weights from Hugging Face into `~/.cache/huggingface/hub/`
- Sends the extracted audio clip to Groq's Whisper API (`api.groq.com/openai/v1/audio/transcriptions`) when Groq is the selected backend and `GROQ_API_KEY` is set
- Sends the extracted audio clip to OpenAI's audio transcription API (`api.openai.com/v1/audio/transcriptions`) when OpenAI is the selected backend and `OPENAI_API_KEY` is set
- Writes the downloaded video, frames, audio, and an intermediate transcript to a working directory under the system temp dir (or `--out-dir` if specified) so Claude can `Read` them
- Reads / creates `~/.config/watch/.env` (mode `0600`) to store the Whisper API key(s), the `WATCH_WHISPER` / `WATCH_DETAIL` preferences, and a `SETUP_COMPLETE` marker. As a fallback, also reads `.env` in the current working directory

**What this skill does NOT do:**
- Does not upload the video itself to any API — only the extracted audio goes out, and only when native captions are missing AND an API backend (not `mlx`) is selected AND Whisper is not disabled with `--no-whisper`
- Does not access any platform account (no login, no session cookies, no posting) — yt-dlp only ever requests public data
- Does not share API keys between providers (Groq key only goes to `api.groq.com`, OpenAI key only goes to `api.openai.com`; the `mlx` backend uses no key)
- Does not log, cache, or write API keys to stdout, stderr, or output files
- Does not persist anything outside the working directory, `~/.config/watch/.env`, and (mlx only) the Hugging Face model cache — clean up the working directory when you're done (Step 5)

**Bundled scripts:** `scripts/watch.py` (entry point), `scripts/download.py` (yt-dlp wrapper), `scripts/frames.py` (ffmpeg frame extraction + tiled dedup), `scripts/transcribe.py` (caption parsing), `scripts/whisper.py` (backend selection, Groq / OpenAI clients), `scripts/local_whisper.py` (mlx_whisper runner + loop repair), `scripts/verify.py` (transcript verification), `scripts/setup.py` (preflight + installer)

Review scripts before first use to verify behavior.
