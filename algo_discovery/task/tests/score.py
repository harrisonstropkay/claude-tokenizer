"""Score `approx_count_tokens` against held-out labels, offline and in-process.

Runs inside the agent's OWN container, in Harbor's default shared verifier mode:
the agent has already finished and will not run again, and the container is torn
down right after scoring. The verify phase is airgapped (`network_mode =
"no-network"`), so labels can't leak and scoring can't fetch anything. Scoring in
the agent's box means the candidate can rely on anything the agent installed while
it was online — no need for a fixed dependency list. Two things are scored:

- The final `/app/approx_count_tokens.py` -> the headline reward, written to
  `/logs/verifier/reward.json`. Harbor's reward contract is a FLAT dict of scalar
  name -> number, so that is all this file contains (primary key `reward` = -MSE,
  higher is better). It cannot hold a time-series.
- Every timestamped snapshot in `/app/submissions/` (written by the agent's
  `submit`) -> a per-submission time-series in `/logs/verifier/submissions.json`.
  This is the auxiliary analysis artifact Harbor's scalar reward can't express.
  Join it with the agent's trajectory.json (per-call tokens + timestamps) to plot
  accuracy vs. spend over the run.

Design choices (apply to every function scored):

- In-process, index-aligned scoring. Texts and labels are zipped; a prediction is
  matched to its label by position, never by parsing a shared stdout stream. This
  structurally removes prediction/label-misalignment bugs.
- Non-gameable, always-defined reward. Any error (bad import, exception, timeout,
  non-int return) counts the item as prediction 0 and is penalized by the squared
  label. There is no "skip on error", so the score is defined for every candidate
  and cannot be improved by failing.
- Untrusted-code hygiene. Candidates are LLM-written. We read the labels fully
  into memory and unlink the test file *before* importing any candidate, so code
  that probes the filesystem cannot read the answer key. Candidate stdout/stderr
  is suppressed. A per-call wall-clock timeout bounds pathological inputs, and a
  total wall-clock budget bounds scoring the (agent-controlled number of)
  snapshots so the verifier always finishes and writes its results.
"""

import contextlib
import glob
import importlib.util
import io
import json
import os
import signal
import time

CANDIDATE = os.environ.get("CANDIDATE_PATH", "/app/approx_count_tokens.py")
TESTSET = os.environ.get("TESTSET_PATH", "/tests/test.jsonl")
REWARD_JSON = os.environ.get("REWARD_JSON_PATH", "/logs/verifier/reward.json")
SUBMISSIONS_DIR = os.environ.get("SUBMISSIONS_DIR", "/app/submissions")
SUBMISSIONS_JSON = os.environ.get("SUBMISSIONS_JSON_PATH", "/logs/verifier/submissions.json")
PER_CALL_TIMEOUT_SEC = int(os.environ.get("PER_CALL_TIMEOUT_SEC", "10"))
SUBMISSIONS_BUDGET_SEC = float(os.environ.get("SUBMISSIONS_BUDGET_SEC", "1200"))

_WORST_REWARD = -1.0e18  # finite sentinel so a broken candidate still ranks last


class _CallTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _CallTimeout()


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _load_labels(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(int(row["n_tokens"]))
    return texts, labels


def _load_fn(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module.approx_count_tokens


def _score_path(path, texts, labels):
    """Score one candidate file. Returns a flat metrics dict; on import failure
    the candidate ranks last with mse/rmse/mae = None (dropped from reward.json)."""
    n = len(labels)
    try:
        fn = _load_fn(path)
    except Exception:
        return {"reward": _WORST_REWARD, "mse": None, "rmse": None, "mae": None,
                "exact_match": 0.0, "n": n, "n_ok": 0, "n_err": n, "load_error": 1}

    se = ae = exact = n_ok = n_err = 0
    for text, y in zip(texts, labels):
        signal.alarm(PER_CALL_TIMEOUT_SEC)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                pred = int(fn(text))  # int() also coerces numpy ints; floats truncate
        except BaseException:  # exception, timeout, or non-int return
            pred = 0
            n_err += 1
        else:
            n_ok += 1
        finally:
            signal.alarm(0)
        d = pred - y
        se += d * d
        ae += abs(d)
        exact += int(d == 0)

    mse = se / n
    return {"reward": -float(mse), "mse": float(mse), "rmse": float(mse ** 0.5),
            "mae": float(ae / n), "exact_match": float(exact / n),
            "n": n, "n_ok": n_ok, "n_err": n_err, "load_error": 0}


def _write_reward(metrics):
    """Reward.json must be a flat dict of scalars; drop any None values."""
    clean = {k: v for k, v in metrics.items() if v is not None}
    _write_json(REWARD_JSON, clean)
    print(json.dumps(clean, sort_keys=True))


def _submission_meta():
    """Map snapshot filename -> {ts, epoch} from the agent-written manifest."""
    meta = {}
    manifest = os.path.join(SUBMISSIONS_DIR, "manifest.jsonl")
    if not os.path.isfile(manifest):
        return meta
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            with contextlib.suppress(Exception):
                row = json.loads(line)
                meta[row["file"]] = {"ts": row.get("ts"), "epoch": row.get("epoch")}
    return meta


def _score_submissions(texts, labels):
    """Score every snapshot in SUBMISSIONS_DIR within a wall-clock budget.
    Returns a list sorted by submission time, or None if there are none."""
    if not os.path.isdir(SUBMISSIONS_DIR):
        return None
    files = sorted(glob.glob(os.path.join(SUBMISSIONS_DIR, "submission_*.py")))
    if not files:
        return None

    meta = _submission_meta()
    started = time.monotonic()
    out = []
    for path in files:
        name = os.path.basename(path)
        info = meta.get(name, {})
        if time.monotonic() - started > SUBMISSIONS_BUDGET_SEC:
            # Headline reward.json is already written; record the rest as skipped
            # rather than risk the verifier container timing out mid-scoring.
            out.append({"file": name, "ts": info.get("ts"),
                        "epoch": info.get("epoch"), "skipped_over_budget": 1})
            continue
        metrics = _score_path(path, texts, labels)
        out.append({"file": name, "ts": info.get("ts"),
                    "epoch": info.get("epoch"), **metrics})

    out.sort(key=lambda r: (r.get("epoch") is None, r.get("epoch") or 0, r["file"]))
    return out


def main():
    texts, labels = _load_labels(TESTSET)
    if len(labels) == 0:
        _write_reward({"reward": _WORST_REWARD, "error_empty_testset": 1})
        return

    # Read the answer key fully, then remove it before running untrusted code so
    # candidates cannot read labels off the filesystem (best-effort).
    with contextlib.suppress(OSError):
        os.remove(TESTSET)

    signal.signal(signal.SIGALRM, _on_alarm)

    # Headline reward: the final in-place function. Written first so it survives
    # even if snapshot scoring later hits its budget.
    _write_reward(_score_path(CANDIDATE, texts, labels))

    # Auxiliary time-series: every timestamped snapshot, for accuracy-vs-spend.
    submissions = _score_submissions(texts, labels)
    if submissions is not None:
        _write_json(SUBMISSIONS_JSON, submissions)
        print(f"scored {len(submissions)} submissions -> {SUBMISSIONS_JSON}")


if __name__ == "__main__":
    main()
