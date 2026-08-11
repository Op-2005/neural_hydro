#!/usr/bin/env bash
# Set up the cross-vendor review leg (poldrack/ai-peer-review) for this project.
#
#   ./setup_crossvendor.sh                   install the tool
#   ./setup_crossvendor.sh --install-prompts merge this project's domain prompts into the config
#   ./setup_crossvendor.sh --check           report readiness, then exit
#
# Idempotent: re-running is safe. Never overwrites API keys already in the config.

set -euo pipefail

TOOL_DIR="${AI_PEER_REVIEW_DIR:-$HOME/tools/ai-peer-review}"
CONFIG="$HOME/.ai-peer-review/config.json"
PROMPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/crossvendor_prompts.json"

check() {
  echo "=== cross-vendor review readiness ==="
  if [ -d "$TOOL_DIR" ]; then echo "tool:     installed at $TOOL_DIR"
  else echo "tool:     NOT installed  (run this script with no args)"; fi

  if [ -f "$CONFIG" ]; then
    python3 - "$CONFIG" <<'PY'
import json, sys, pathlib
cfg = json.loads(pathlib.Path(sys.argv[1]).read_text())
keys = sorted(k for k, v in cfg.get("api_keys", {}).items() if v)
print("keys:    ", ", ".join(keys) if keys else "NONE — cross-vendor pass will be skipped")
sysp = cfg.get("prompts", {}).get("system", "")
print("prompts: ", "project (scientific ML + hydrology)"
      if "hydrology" in sysp else "DEFAULT neuroscience — run --install-prompts")
PY
  else
    echo "keys:     no config — cross-vendor pass will be skipped"
    echo "prompts:  not installed"
  fi
  echo
  echo "The 'review' skill runs with or without this leg; without keys it reports"
  echo "corroboration as in-family only."
}

install_prompts() {
  [ -f "$PROMPTS" ] || { echo "error: $PROMPTS not found" >&2; exit 1; }
  mkdir -p "$(dirname "$CONFIG")"
  [ -f "$CONFIG" ] || echo '{"api_keys": {}, "prompts": {}}' > "$CONFIG"

  # Merge prompts in, preserving api_keys. Back up first.
  cp "$CONFIG" "$CONFIG.bak"
  python3 - "$CONFIG" "$PROMPTS" <<'PY'
import json, pathlib, sys
cfg_p, new_p = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
cfg = json.loads(cfg_p.read_text())
new = json.loads(new_p.read_text())
cfg.setdefault("api_keys", {})
cfg.setdefault("prompts", {}).update(new["prompts"])   # keys untouched
cfg_p.write_text(json.dumps(cfg, indent=2))
print(f"merged {len(new['prompts'])} prompts into {cfg_p} (api_keys preserved; backup at {cfg_p}.bak)")
PY
}

case "${1:-install}" in
  --check)           check; exit 0 ;;
  --install-prompts) install_prompts; echo; check; exit 0 ;;
esac

# Install the tool.
if [ ! -d "$TOOL_DIR" ]; then
  mkdir -p "$(dirname "$TOOL_DIR")"
  git clone https://github.com/poldrack/ai-peer-review.git "$TOOL_DIR"
else
  echo "tool already present at $TOOL_DIR — skipping clone"
fi

cd "$TOOL_DIR"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -e .
echo "installed ai-peer-review into $TOOL_DIR/.venv"

install_prompts

cat <<'EOF'

Next: add at least one non-Anthropic key (two vendors makes it a real cross-vendor signal):

  cd "$TOOL_DIR" && source .venv/bin/activate
  ai-peer-review set-key openai   "$OPENAI_API_KEY"
  ai-peer-review set-key google   "$GOOGLE_API_KEY"
  ai-peer-review set-key together "$TOGETHER_API_KEY"

Then just say "review" — the leg activates automatically once keys are present.
EOF
