"""
Analytical metrics for the D×N overlap and TPR benchmarks.

All functions operate on probability distributions stored in the prediction CSVs
produced by run_experiments.py (discourse experiment). No model inference is
required; these are post-hoc, deterministic computations over stored P_the / P_a
values.

Inputs
------
sites_df : DataFrame with one row per D×N site.
    Required columns: 'noun', 'P_the', 'P_a', 'original_det'

tpr_df : DataFrame with one row per eligible cross-speaker reference pair.
    Required columns: 'd_A' (int, 1 = prior human said 'the', 0 = 'a'),
                      'P_the_B' (float, model probability of 'the' at target site)
"""

import numpy as np
import pandas as pd
from determiner.overlap_list_v2 import average_expected_overlap


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _p_the(sites_df: pd.DataFrame) -> pd.Series:
    """Return P(the) normalized within each row, guarding against rounding."""
    total = sites_df["P_the"] + sites_df["P_a"]
    return sites_df["P_the"] / total.replace(0, np.nan)


# ---------------------------------------------------------------------------
# D×N overlap metrics
# ---------------------------------------------------------------------------

def analytical_bias(sites_df: pd.DataFrame) -> float:
    """
    Noun-type-level determiner bias (analytical analog of find_bias).

    For each noun type n, the expected count of 'the' tokens is sum(P_the_i)
    and of 'a' tokens is sum(1-P_the_i) across its k_n sites. The dominant
    count is max of the two. Bias = sum of dominant counts / S.
    """
    S = len(sites_df)
    if S == 0:
        return np.nan
    p = _p_the(sites_df)
    df = sites_df[["noun"]].copy()
    df["_p"] = p
    df["_q"] = 1.0 - p
    g = df.groupby("noun")[["_p", "_q"]].sum()
    dominant = g.max(axis=1)
    return float(dominant.sum() / S)


def analytical_empirical_overlap(sites_df: pd.DataFrame) -> float:
    """
    Expected fraction of noun types attested with both determiners.

    For each noun n with sites i = 1…k_n:
        P(overlap for n) = 1 - prod(P_the_i) - prod(1-P_the_i)

    This is exact inclusion-exclusion: the events 'no the ever' and 'no a ever'
    are mutually exclusive (a site cannot produce both), so their intersection
    has probability 0 and the standard IE remainder vanishes.

    Sanity checks:
        k_n=1, p=0.7  →  1 - 0.7 - 0.3 = 0    (impossible with one site)
        k_n=2, p=[1,0] →  1 - 0   - 0   = 1    (certain overlap)
    """
    p = _p_the(sites_df)
    df = sites_df[["noun"]].copy()
    df["_p"] = p

    def _p_overlap(p_series: pd.Series) -> float:
        p_vals = p_series.dropna().values
        if len(p_vals) == 0:
            return 0.0
        prod_the = float(np.prod(p_vals))
        prod_a   = float(np.prod(1.0 - p_vals))
        return max(0.0, 1.0 - prod_the - prod_a)

    return float(df.groupby("noun")["_p"].apply(_p_overlap).mean())


def analytical_predicted_overlap(sites_df: pd.DataFrame) -> float:
    """
    Yang (2013) expected overlap using analytical N, S, and bias.

    Returns NaN when bias == 1.0 (all tokens predicted as the same
    determiner, formula undefined) or when the Yang assertion fails.
    """
    N = sites_df["noun"].nunique()
    S = len(sites_df)
    if N == 0 or S == 0:
        return np.nan
    b = analytical_bias(sites_df)
    if np.isnan(b) or b >= 1.0 or b <= 0.0:
        return np.nan
    try:
        return float(average_expected_overlap(N, S, b=b))
    except (AssertionError, ZeroDivisionError, ValueError):
        return np.nan


def analytical_accuracy(sites_df: pd.DataFrame) -> float:
    """
    Argmax accuracy: fraction of sites where argmax(P_the, P_a) == original_det.

    Consistent with BLiMP-style forced-choice evaluation. Fully determined
    by stored probabilities; no sampling required.
    """
    p = _p_the(sites_df)
    predicted = p.map(lambda x: "the" if (not np.isnan(x) and x >= 0.5) else "a")
    return float((predicted == sites_df["original_det"]).mean())


# ---------------------------------------------------------------------------
# TPR metrics
# ---------------------------------------------------------------------------

def analytical_tpr(tpr_df: pd.DataFrame) -> float:
    """
    Overall TPR: expected probability of switching from the prior human determiner.

        TPR = mean(|d_A - P_the_B|)

    When d_A = 0 (prior said 'a')  → contribution = P_the_B (prob of switching to 'the')
    When d_A = 1 (prior said 'the') → contribution = 1 - P_the_B (prob of switching to 'a')
    """
    return float((tpr_df["d_A"] - tpr_df["P_the_B"]).abs().mean())


def analytical_tpr_the(tpr_df: pd.DataFrame) -> float:
    """P(model says 'the' | prior human said 'a')."""
    sub = tpr_df[tpr_df["d_A"] == 0]
    return float(sub["P_the_B"].mean()) if len(sub) > 0 else np.nan


def analytical_tpr_a(tpr_df: pd.DataFrame) -> float:
    """P(model says 'a' | prior human said 'the')."""
    sub = tpr_df[tpr_df["d_A"] == 1]
    return float((1.0 - sub["P_the_B"]).mean()) if len(sub) > 0 else np.nan
