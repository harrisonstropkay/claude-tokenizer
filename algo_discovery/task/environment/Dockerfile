# Agent workspace image — also the scoring environment: in Harbor's default shared
# verifier mode the held-out set is graded right here after the agent finishes. The
# agent iterates with full internet (research is allowed); it gets the training
# split, the starting function, and a preinstalled python stack (it may pip-install
# more while online — scoring is offline, so nothing new can be fetched then).
# Claude Code is baked in (Harbor skips its own install when `claude` is present),
# which pins the agent version and avoids a per-trial download.
# Build context is the repo root (so it can COPY the training split).
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUTF8=1 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates git bash procps \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code@2.1.232 && claude --version

# Preinstalled scoring stack (pinned for reproducibility). The agent may add more.
RUN pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu \
 && pip install "tiktoken==0.8.0" "numpy==2.1.3"

RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base').encode('warm up the cache')"

# `submit` snapshots the current function to /app/submissions/ with a timestamp,
# so the verifier can score each iteration and we can plot accuracy vs. spend.
COPY algo_discovery/task/environment/submit /usr/local/bin/submit
RUN chmod +x /usr/local/bin/submit

WORKDIR /app
COPY algo_discovery/seeds/ratio_seed.py /app/approx_count_tokens.py
COPY data/splits_1000_1200/train.jsonl /app/task_data/train.jsonl
