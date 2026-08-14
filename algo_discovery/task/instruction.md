# Approximate the Claude tokenizer

Make this function return the **exact** number of tokens Anthropic's tokenizer
(Claude 4.7+, via the `count_tokens` API) assigns to `text`:

```python
# /app/approx_count_tokens.py
def approx_count_tokens(text: str) -> int:
    ...
```

A rough starting version is already there. Make it exact.

## The goal

There is a real, deterministic algorithm behind these counts — recover it. Aim for
a perfect score: the right integer on **every** text. Treat each remaining error as
a clue, find the root cause, fix it, repeat. Don't stop while any text is wrong.

You have full internet access — research prior work on this tokenizer and use
anything that helps.

## How you're scored

- **Metric:** mean squared error (MSE) vs. the true counts. **Lower is better; 0 is
  perfect.**
- **Write your solution into `/app/approx_count_tokens.py`** (keep the same function
  name and signature). Whatever is in that file when you finish is what gets graded.
- **Run `submit` after every single change to the function** — no matter how small,
  and even if you're not sure it helped. It snapshots the current file with a
  timestamp, and every snapshot is scored. Make it a reflex: change the function →
  `submit`. Never batch multiple edits between submits.
- **Failure = a 0-token prediction, which is a penalty, not a pass.** If the
  function raises, hangs, or returns a non-integer on some text, that text scores as
  `0` tokens — almost always far off, so it *inflates* your MSE. Return a real
  integer for every possible input.

## Environment

- **Training data:** `/app/task_data/train.jsonl`, one object per line:
  `{"text": ..., "n_tokens": ...}` (`n_tokens` is the true count). Measure your own
  MSE on it as you go.
- **Scoring runs right here, after you finish** — same interpreter and libraries you
  test with. `tiktoken`, `numpy`, and `torch` are preinstalled; `pip install`
  anything else you need.
- **Scoring is offline.** Install packages and download any data (weights, vocab,
  tables) *now*, while you have network. At runtime the function must rely only on
  what's already on disk (or embedded in the file).
