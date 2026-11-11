"""Yang (2013) expected D×N overlap under a fully productive grammar.

Nouns follow a Zipfian distribution over ``N`` types, and each of the ``S`` determiner
tokens independently picks the favored determiner with probability ``b`` (the bias).
The expected overlap is the expected share of noun types that occur with both
determiners. ``overlap_stats`` gives the count-based version for a list of
``"det noun"`` pairs; ``determiner.analytical_metrics`` is the probability-weighted
version used for models.

Usage as a script::

    python -m determiner.overlap pairs.txt     # one "det noun" pair per line
"""
import math
import sys
from collections import Counter
from typing import Dict, Iterable


def harmonic(n: int, a: float = 1) -> float:
    """Generalized harmonic number H(n, a) = sum_{i=1..n} 1 / i^a."""
    s = 0
    for i in range(1, n + 1):
        s += 1.0 / math.pow(i, a)
    return s


def expected_overlap(N: int, S: int, r: int, a: float = 1, b: float = 0.82, hN: float = None) -> float:
    """Probability that the noun of Zipf rank ``r`` appears with both determiners."""
    if hN is None:
        hN = harmonic(N, a)
    p = 1.0 / (math.pow(r, a) * hN)
    eo = 1 - sum([math.pow((p * di + 1.0 - p), S) for di in [b, 1.0 - b]]) + math.pow(1 - p, S)
    assert eo > 0, 'r=%d, predicted=%.6f' % (r, eo)
    return eo


def average_expected_overlap(N: int, S: int, a: float = 1, b: float = 0.82) -> float:
    """Expected overlap of a sample with ``N`` noun types and ``S`` tokens (Yang 2013)."""
    hN = harmonic(N, a)
    sumo = 0
    for r in range(1, N + 1):
        sumo += expected_overlap(N, S, r, a, b, hN=hN)
    return sumo / N


def overlap_stats(pairs: Iterable[str]) -> Dict[str, float]:
    """
    Count-based D×N statistics for ``"det noun"`` strings (``an`` is merged into ``a``).

    Returns N (noun types), S (tokens), bias (share of tokens taking each noun's favored
    determiner), r = S / N, the empirical overlap (share of noun types seen with both
    determiners) and the predicted overlap under a productive grammar.
    """
    counts = Counter()
    for pair in pairs:
        tokens = pair.strip().split()
        if len(tokens) != 2:
            continue
        det, noun = tokens
        counts[("a" if det.lower() in ("a", "an") else det.lower(), noun)] += 1

    per_noun = {}
    for (det, noun), c in counts.items():
        per_noun.setdefault(noun, Counter())[det] += c
    N = len(per_noun)
    S = sum(counts.values())
    bias = sum(max(c.values()) for c in per_noun.values()) / S
    empirical = sum(1 for c in per_noun.values() if c["a"] > 0 and c["the"] > 0) / N
    return {
        "N": N, "S": S, "bias": bias, "r": S / N,
        "empirical": empirical,
        "predicted": average_expected_overlap(N, S, b=bias),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m determiner.overlap PAIRS_FILE")
    with open(sys.argv[1]) as f:
        stats = overlap_stats(f)
    print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
