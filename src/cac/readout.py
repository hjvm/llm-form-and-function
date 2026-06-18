"""Generic, N-category readouts of a CAC distribution.

These are study-agnostic surprisal-style readouts that work for any candidate set.
Construction-specific metrics (e.g. the determiner study's overlap / TPR / bias) are
NOT here — they live in the client that owns that linguistics.
"""
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


def mass_on(probs: Dict[str, float], subset: Iterable[str]) -> float:
    """Total probability mass on a subset of categories.

    Generalizes 'P(the)' (= ``mass_on(d, ["the"])``) and the RI rate
    (= ``mass_on(d, non_finite_categories)``).
    """
    return float(sum(probs.get(c, 0.0) for c in subset))


def surprisal(probs: Dict[str, float], category: str, base: Optional[float] = None) -> float:
    """Surprisal -log p(category). Natural log by default; pass ``base=2`` for bits."""
    p = float(probs.get(category, 0.0))
    if p <= 0.0:
        return float("inf")
    s = -np.log(p)
    return float(s if base is None else s / np.log(base))


def choose_prediction(probs: Dict[str, float], candidates: Optional[Sequence[str]] = None):
    """Return ``(argmax_category, sampled_category)`` over the distribution.

    ``argmax`` uses first-maximum tie-breaking (``np.argmax``). The determiner study
    keeps its own ``choose_determiner_predictions`` (which ties toward ``the``), so
    this generic helper does not change any committed determiner predictions.
    """
    cats = list(candidates) if candidates is not None else list(probs.keys())
    vals = np.array([float(probs.get(c, 0.0)) for c in cats], dtype=float)
    argmax = cats[int(np.argmax(vals))]
    total = float(vals.sum())
    p = vals / total if (total > 0 and np.isfinite(total)) else np.full(len(cats), 1.0 / len(cats))
    sample = cats[int(np.random.choice(len(cats), p=p))]
    return argmax, sample
