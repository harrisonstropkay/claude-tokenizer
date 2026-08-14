#!/usr/bin/env bash
# Harness smoke test, NOT a ground-truth solution. There is no known closed form
# for the Claude 4.7+ tokenizer — reconstructing it is the whole research problem
# — so no script here can be a true "oracle". Harbor's built-in `oracle` agent
# just runs this file; we use it to exercise the full pipeline (build -> agent
# phase -> airgapped in-container scoring -> reward.json) deterministically, for
# free, with no model and no secrets.
#
# It installs a reasonable *reference baseline* so that smoke test also reports a
# meaningful score: cl100k over-counts relative to Claude's tokenizer, so we scale
# it by the aggregate ratio (true tokens / cl100k tokens) measured on train. This
# doubles as a hint of the intended workflow — fit on train, embed the constant.
#
# It also calls `submit` twice (starting seed, then calibrated version) so the
# smoke test exercises the snapshot -> submissions.json path end to end, the same
# way a real agent would iterate.
set -euo pipefail

# Snapshot the starting seed first, so submissions.json has a "before" point.
submit
sleep 1

python3 - <<'PY'
import json, tiktoken

enc = tiktoken.get_encoding("cl100k_base")
true_total = cl100k_total = 0
with open("/app/task_data/train.jsonl", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cl100k_total += len(enc.encode(row["text"], disallowed_special=()))
        true_total += int(row["n_tokens"])
scale = true_total / cl100k_total if cl100k_total else 1.0

with open("/app/approx_count_tokens.py", "w", encoding="utf-8") as out:
    out.write(
        "import tiktoken\n\n"
        "_ENC = tiktoken.get_encoding('cl100k_base')\n"
        f"_SCALE = {scale!r}\n\n\n"
        "def approx_count_tokens(text: str) -> int:\n"
        "    n = len(_ENC.encode(text, disallowed_special=()))\n"
        "    return max(1, round(n * _SCALE))\n"
    )
print(f"oracle calibrated cl100k scale = {scale}")
PY

# Snapshot the improved version.
submit
