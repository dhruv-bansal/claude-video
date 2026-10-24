#!/usr/bin/env bash
# Advisory only: config parsing and permissions belong to the shared Python code.
set -u
PLUGIN_DIR="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SETUP_SCRIPT="$PLUGIN_DIR/skills/watch/scripts/setup.py"
# Local Whisper backend: mlx_whisper on PATH, or where pipx puts it before `pipx ensurepath`.
HAS_MLX=""
{ command -v mlx_whisper >/dev/null 2>&1 || [[ -x "$HOME/.local/bin/mlx_whisper" ]]; } && HAS_MLX="yes"
APPLE_SILICON=""
[[ "$(uname -s 2>/dev/null)" == "Darwin" && "$(uname -m 2>/dev/null)" == "arm64" ]] && APPLE_SILICON="yes"
for interpreter in python3 python; do
  if command -v "$interpreter" >/dev/null 2>&1 && "$interpreter" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
    if "$interpreter" "$SETUP_SCRIPT" --check >/dev/null 2>&1; then
      exit 0
    fi
    echo '/watch: setup needs attention. Invoke the watch skill to check dependencies and configuration.'
    if [[ -n "$APPLE_SILICON" && -z "$HAS_MLX" ]]; then
      echo '/watch: optional free local transcription on this Mac: `pipx install mlx-whisper` (no API key, nothing uploaded).'
    fi
    exit 0
  fi
done
echo '/watch: Python 3.10+ is needed. Install a working Python interpreter and invoke the watch skill.'
exit 0
