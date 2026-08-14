# algo-discovery: reconstruct `count_tokens`

Can an agent reverse-engineer Anthropic's tokenizer? The agent is dropped into a
sandbox with a starting `approx_count_tokens(text) -> int` and told to make it
match Claude 4.7+'s real token counts. It researches, iterates, and leaves its
best function behind. When it finishes, Harbor scores that function against a
hidden test split (by MSE) **inside the agent's own container, with the network
cut** — so the function can lean on anything the agent installed, but the labels
never reach it. Each time it improves the function it runs `submit`, which
snapshots the function with a timestamp so every iteration is scored too — giving
an accuracy-vs-spend curve over the run (see "The accuracy-vs-cost curve" below).

[Harbor](https://github.com/harbor-framework/harbor) owns the whole loop —
environment, agent, verifier, token/timestamp accounting — so this repo is just a
task definition plus a launcher.

## Layout

```
algo_discovery/
  run.sh              # build agent image + stage test split + launch (Claude Code on GLM, or a smoke test)
  task/
    task.toml         # Harbor task: shared verifier (airgapped verify phase), timeouts
    instruction.md    # the prompt the agent gets
    environment/      # agent image: node + claude-code + pinned py stack + train split + seed + `submit`
    tests/            # scorer (score.py, test.sh); run.sh stages the HIDDEN test.jsonl here
    solution/solve.sh # harness smoke test: reference baseline (see below), run via `-a oracle`
  seeds/              # ratio_seed.py — the starting function baked into the agent image
  jobs/               # Harbor run outputs — gitignored
```

## Security model

- **The agent never sees the answer key.** `test.jsonl` is gitignored; `run.sh`
  stages it into `task/tests/` only at launch. Harbor uploads that dir to `/tests`
  and scores **after the agent has finished and been stopped** — it will not run
  again in that box — with `[verifier] network_mode="no-network"`, so labels can't
  leak and scoring can't fetch anything. During the agent's own turns it has only
  `train.jsonl`. (`score.py` also reads the labels into memory and unlinks the
  file before importing the candidate, as defense in depth.)
- **Secrets stay out of the repo.** The GLM proxy master key and LiteLLM key are
  read from your installed `glaude` at runtime and passed to the container as
  `${VAR}` templates — they never land in this repo or in Harbor's job config.

## Prerequisites

- **Docker** (builds and runs both images).
- **Harbor** on `PATH` — install once, durably: `uv tool install harbor`
  (pins to 0.21.x; provides the `harbor` CLI).
- For a real run only: an installed `glaude` (`~/.local/bin/glaude`) holding the
  GLM proxy/key/base-url. `run.sh` reads these at runtime; override its location
  with `GLAUDE_BIN=...`.

## Run it

```bash
./run.sh          # real run: Claude Code on GLM-5.2 (RITS) via the proxy
./run.sh oracle   # harness smoke test: run solution/solve.sh (reference baseline)
./run.sh nop      # harness smoke test: score the untouched ratio seed
```

`oracle` and `nop` need no model or secrets — they run the whole pipeline (build
→ agent phase → airgapped in-container scoring → `reward.json`) deterministically
and for free, so you can confirm the harness works before spending on a real run.

There is deliberately no true "oracle" here: reconstructing Claude's tokenizer
exactly is the open problem, so no reference solution can score a perfect MSE.
`solution/solve.sh` is only a *smoke test* — Harbor's built-in `oracle` agent runs
it. It installs a reference baseline (cl100k scaled by a single ratio fit on train,
RMSE ≈ 29) so the smoke test also reports a meaningful, non-trivial score.

## What you get

Each run writes to `jobs/<job>/<trial>/`. The key files:

- **`result.json`** — the headline. `agent_result.{n_input_tokens, n_cache_tokens,
  n_output_tokens, cost_usd}` (total tokens/cost), `verifier_result.rewards` =
  `{reward: -mse, mse, rmse, mae, exact_match, n, ...}` for the *final* function
  (`reward` is `-mse`; higher is better), and `started_at`/`finished_at` plus
  per-phase timing (environment setup, agent setup, agent execution, verifier).
- **`agent/trajectory.json`** — the per-LLM-call series: every step has a
  `timestamp` and `metrics.{prompt_tokens, completion_tokens, cached_tokens,
  cost_usd}`. This is the **spend axis**. Keep the three token counts *separate* —
  input, output, and cache price differently, so don't pre-sum them into one
  number here; carry the raw counts and apply your own per-token rates to get USD
  (Harbor's `cost_usd` won't apply to GLM).
- **`verifier/submissions.json`** — the per-submission series: one entry per
  `submit`, each with `ts`, `epoch`, and its hidden-test `{reward, mse, rmse, mae,
  exact_match, n_ok, n_err}`. This is the **performance axis**.

## The accuracy-vs-cost curve

Harbor's reward is a single scalar, so it can't hold a time-series; there's also
no native "grade one run repeatedly" mode (multi-step trials fragment the run into
separate agent invocations). Instead we score each `submit` snapshot in the
verifier and emit `submissions.json`, then join it to `trajectory.json` by
timestamp:

- **x (spend)** — the running totals of `prompt_tokens`, `completion_tokens`, and
  `cached_tokens` from `trajectory.json` steps up to each submission's `ts`,
  **kept as three separate numbers**. Convert to USD later with your own rates;
  don't collapse them here.
- **y (performance)** — that submission's `rmse` (or `-mse`) from
  `submissions.json`.

Both series share the same UTC wall clock (the container's), so a merge-by-time
gives you the input/output/cache tokens accrued so far vs. accuracy at every
checkpoint the agent recorded.

## Notes

- Scoring is non-gameable: a prediction that raises, hangs (per-call timeout), or
  returns a non-integer counts as **0** for that text. Every `submit` snapshot is
  scored under the same rules; snapshots beyond a wall-clock budget are marked
  `skipped_over_budget` so the verifier always finishes and writes its results.
- Scoring runs in the agent's own container, so what the agent tests locally is
  exactly what the verifier runs — same interpreter, same libraries (including any
  the agent pip-installed). The preinstalled stack (`tiktoken`, `numpy`, `torch`)
  is pinned for reproducibility.
