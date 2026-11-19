"""Coin-flip baselines: P(the) at every attested site without looking at the context.

A constant coin is the independence model itself: at fixed sites its analytical empirical
overlap equals conditional_overlap(child's noun counts, q) exactly, so its D×N outcome is fixed
by the child's nouns and q (docs/BASELINES.md). The noun-level coin carries each noun's corpus
P(the | noun) and is the informative one.
"""
import numpy as np
import pandas as pd

from baselines.common import Scorer

COINS = {
    "fair (P(the)=0.5)": lambda s: np.full(len(s), 0.5),
    "0.82 toward 'the'": lambda s: np.full(len(s), 0.82),
    "0.82 toward 'a'": lambda s: np.full(len(s), 0.18),
    "0.82 toward noun's favored det": lambda s: np.where(s.noun_pthe >= 0.5, 0.82, 0.18),
    "noun-level bias (corpus P(the|noun))": lambda s: s.noun_pthe.values,
    "noun-level bias, leave-one-out": lambda s: s.noun_pthe_loo.values,
}


def run(scorer: Scorer, n_draws: int = 1000, seed: int = 20261118) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = [scorer.evaluate_with_draws(name, f(scorer.sites), n_draws, rng) for name, f in COINS.items()]
    return pd.concat(out, ignore_index=True)
