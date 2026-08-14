#!/usr/bin/env bash
# Harbor's default (shared) verifier: this whole tests/ dir is uploaded to /tests
# in the agent's own container and run here AFTER the agent has finished. The
# verify phase is airgapped (task.toml `[verifier] network_mode="no-network"`), so
# score.py reads the answer key (/tests/test.jsonl) and the agent's final
# /app/approx_count_tokens.py, then writes /logs/verifier/reward.json. We
# deliberately do NOT set -e: score.py already turns every candidate failure into a
# finite worst score, so it exits 0 and writes a reward on its own; letting it run
# to completion is what guarantees the reward file exists.
set -uo pipefail
mkdir -p /logs/verifier
python3 /tests/score.py
