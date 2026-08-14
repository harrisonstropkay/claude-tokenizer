#!/usr/bin/env bash
# Build the agent image, stage the hidden test split, and launch with Harbor.
#
#   ./run.sh              # real run: Claude Code on GLM-5.2 (RITS) via the proxy
#   ./run.sh oracle       # harness smoke test: run solution/solve.sh (reference baseline)
#   ./run.sh nop          # harness smoke test: score the untouched ratio seed
#
# GLM secrets (proxy master key, LiteLLM key) are read from the installed
# `glaude` at runtime and handed to the agent container as ${VAR} templates, so
# nothing sensitive is written into this repo or into Harbor's job config.
# Results, token counts, and timestamps land under algo_discovery/jobs/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="$REPO_ROOT/algo_discovery/task"
GLAUDE="${GLAUDE_BIN:-$HOME/.local/bin/glaude}"
JOBS_DIR="${JOBS_DIR:-$REPO_ROOT/algo_discovery/jobs}"
HARBOR="${HARBOR_BIN:-harbor}"
MODEL="${MODEL:-rits/zai-org/glm-5-2-fp8}"
AGENT="${1:-claude-code}"

echo "==> Building agent image (count-tokens-agent:latest)"
docker build -t count-tokens-agent:latest -f "$TASK_DIR/environment/Dockerfile" "$REPO_ROOT"

# Stage the hidden test split into tests/ (gitignored). This copies it onto the
# HOST only; it does not reach the running agent. The agent image (built above from
# task/environment/) never COPYs it, and Harbor bind-mounts only the run's output
# dirs into the agent container — not task/tests/. The single path that puts this
# dir into the container is the verifier's upload to /tests, which runs *after* the
# agent has stopped and with the network cut. So the labels are unreachable during
# the agent's turns by construction, not just by convention.
echo "==> Staging held-out test split into task/tests/"
cp "$REPO_ROOT/data/splits_1000_1200/test.jsonl" "$TASK_DIR/tests/test.jsonl"

# oracle / nop are harness smoke tests: no model, no secrets. `oracle` is Harbor's
# built-in agent that runs solution/solve.sh; `nop` just scores the seed as-is.
if [[ "$AGENT" == "oracle" || "$AGENT" == "nop" ]]; then
  echo "==> Running smoke-test agent: $AGENT"
  exec "$HARBOR" run -p "$TASK_DIR" -a "$AGENT" -o "$JOBS_DIR" -y
fi

# --- GLM routing: read only the env exports we need from glaude, never run it ---
if [[ ! -r "$GLAUDE" ]]; then
  echo "run.sh: cannot read $GLAUDE (set GLAUDE_BIN to your glaude path)" >&2
  exit 1
fi
eval "$(grep -E '^export (HTTPS_PROXY|HTTP_PROXY|NO_PROXY|ANTHROPIC_BASE_URL|ANTHROPIC_API_KEY|CLAUDE_CODE_MAX_CONTEXT_TOKENS)=' "$GLAUDE")"

echo "==> Launching count-tokens on $MODEL via RITS"
# ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY are read from this shell's env by the
# claude-code adapter; proxy + context-window vars are not auto-forwarded, so we
# template them into the container (values resolve from the env set just above).
exec "$HARBOR" run \
  -p "$TASK_DIR" \
  -a claude-code \
  -m "$MODEL" \
  -o "$JOBS_DIR" \
  -y \
  --ae HTTPS_PROXY='${HTTPS_PROXY}' \
  --ae HTTP_PROXY='${HTTP_PROXY}' \
  --ae NO_PROXY='${NO_PROXY}' \
  --ae CLAUDE_CODE_MAX_CONTEXT_TOKENS='${CLAUDE_CODE_MAX_CONTEXT_TOKENS}'
