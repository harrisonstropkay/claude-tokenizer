"""Seed: constant chars->tokens ratio.

The simplest possible baseline: multiply character length by the aggregate
tokens-per-character ratio measured on the training set
(sum(n_tokens) / sum(len(text)) = 0.358906, i.e. ~2.786 chars/token).
This is the starting point the agent will improve.
"""
TOKENS_PER_CHAR = 0.358906


def approx_count_tokens(text: str) -> int:
    return max(1, round(len(text) * TOKENS_PER_CHAR))
