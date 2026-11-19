"""Shared inputs and scoring for the baselines.

Sites
    One row per attested D×N site of the committed Manchester inventory. ``sentence_id`` is the
    row number of the site in that file, the same key Step 3 uses to join model predictions and
    TPR cases.
TPR cases
    ``output/manchester_tpr_childes/tpr_cached_inputs.csv`` from Step 2.
Scoring
    ``evaluate(p_the)`` takes P(the) at every site (P(a) = 1 - P(the)) and returns, for child
    and caretaker sites, the D×N test (paired t across the 12 dyads, JZS BF01) and the TPR test
    (one-sample t of the 12 per-dyad TPRs against the pooled adult mean), as in the paper.

Closed forms (docs/BASELINES.md, "One formula, four special cases")
    conditional_overlap      nouns fixed:  mean over nouns of 1 - prod P(the) - prod P(a)
    unconditional_overlap    nouns drawn S times from pi:  E_n = 1 - (1 - pi a)^S - (1 - pi t)^S + (1 - pi)^S
    yang_predicted_overlap   the paper's prediction: Zipf nouns over N ranks, one bias b
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel

from determiner.analytical_metrics import jzs_bf01
from determiner.paths import MANCHESTER_TPR_DIR, RESULTS_DIR, SITE_INVENTORY_DIR

ALPHA = 0.05
KEY = ["child_name", "speaker", "sentence_id"]
SITE_INVENTORY = SITE_INVENTORY_DIR / "manchester_det_noun_locations.csv"
TPR_CASES = MANCHESTER_TPR_DIR / "tpr_cached_inputs.csv"
BASELINE_RESULTS = RESULTS_DIR / "baselines"


# --------------------------------------------------------------------------- inputs
def load_inventory(path: Path = SITE_INVENTORY) -> pd.DataFrame:
    """The site inventory with ``sentence_id`` (row number), ``speaker`` and ``the`` (0/1)."""
    inv = pd.read_csv(path)
    inv = inv.reset_index().rename(columns={"index": "sentence_id"})
    inv["speaker"] = inv.speaker_type
    inv["the"] = (inv.det.str.lower() == "the").astype(int)
    return inv


def load_sites(path: Path = SITE_INVENTORY) -> pd.DataFrame:
    """Sites plus each noun's corpus-wide P(the | noun) (all 75,133 tokens, children and mothers)."""
    inv = load_inventory(path)
    sites = inv[["sentence_id", "child_name", "speaker", "noun", "det"]].rename(columns={"det": "original_det"})
    sites["orig_the"] = inv["the"].values
    g = inv.dropna(subset=["noun"]).groupby("noun")["the"].agg(n="size", n_the="sum")
    sites["noun_n"] = sites.noun.map(g.n)
    sites["noun_nthe"] = sites.noun.map(g.n_the)
    sites["noun_pthe"] = sites.noun_nthe / sites.noun_n
    p_corpus = float(g.n_the.sum() / g.n.sum())
    loo = sites.noun_n - 1          # leave-one-out: the noun's bias without the site itself
    sites["noun_pthe_loo"] = np.where(loo > 0, (sites.noun_nthe - sites.orig_the) / loo.replace(0, np.nan), p_corpus)
    return sites


def load_tpr_cases(path: Path = TPR_CASES) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} is written by Step 2 (pipeline/02_human_baseline.ipynb); run it first.")
    meta = pd.read_csv(path, usecols=["sentence_id", "child_name", "speaker", "noun", "original_det", "previous_det"])
    meta["d_A"] = meta.previous_det.str.lower().map({"a": 0, "an": 0, "the": 1})
    meta = meta.dropna(subset=["d_A"])
    meta["d_A"] = meta.d_A.astype(int)
    meta["human_switch"] = ((meta.original_det.str.lower() == "the").astype(int) != meta.d_A).astype(int)
    return meta


def load_tpr_references() -> Tuple[pd.Series, pd.Series, float]:
    refs = pd.read_csv(RESULTS_DIR / "tpr" / "tpr_human_perdyad_references.csv")
    child_ref = refs[refs.speaker == "child"].set_index("child_name_norm").tpr_unrestricted
    mother_ref = refs[refs.speaker == "mother"].set_index("child_name_norm").tpr_unrestricted
    adult = float(pd.read_csv(RESULTS_DIR / "tpr" / "tpr_human_summary.csv").iloc[0].adult_tpr_population_mean)
    return child_ref, mother_ref, adult


# --------------------------------------------------------------------------- closed forms
def yang_predicted_overlap(N: int, S: int, b: float) -> float:
    """Vectorised determiner.overlap.average_expected_overlap(N, S, b=b) (identical values)."""
    r = np.arange(1, N + 1, dtype=float)
    p = 1.0 / (r * np.sum(1.0 / r))
    eo = 1 - (p * b + 1 - p) ** S - (p * (1 - b) + 1 - p) ** S + (1 - p) ** S
    return float(eo.mean())


def conditional_overlap(f: np.ndarray, t: np.ndarray) -> float:
    """Nouns fixed: noun n used f_n times, P(the) = t_n at each of its sites.
    Mean over nouns of 1 - t^f - (1 - t)^f (Charles Yang's '1 - the^s - a^s'; with one t for every
    noun this is simple_expected_overlap() of the original overlap code)."""
    f, t = np.asarray(f, float), np.asarray(t, float)
    return float(np.mean(1 - t ** f - (1 - t) ** f))


def unconditional_overlap(pi: np.ndarray, t: np.ndarray, S: int) -> float:
    """Nouns drawn S times with replacement from pi, P(the | noun) = t: expected number of nouns
    with both determiners over expected number of distinct nouns drawn."""
    pi, t = np.asarray(pi, float), np.asarray(t, float)
    both = 1 - (1 - pi * (1 - t)) ** S - (1 - pi * t) ** S + (1 - pi) ** S
    seen = 1 - (1 - pi) ** S
    return float(both.sum() / seen.sum())


def bias_hat(f: np.ndarray, n_the: np.ndarray) -> float:
    """The paper's bias: sum over nouns of the favoured determiner's count, over S."""
    f, n_the = np.asarray(f, float), np.asarray(n_the, float)
    return float(np.maximum(n_the, f - n_the).sum() / f.sum())


def counts_overlap(nouns, the) -> Tuple[float, float, float, int]:
    """Realised sample: (empirical overlap, Yang-predicted overlap, bias, N) as in the paper."""
    k = pd.factorize(pd.Series(nouns))[0]
    f = np.bincount(k)
    t = np.bincount(k, weights=np.asarray(the, float))
    N, S = len(f), int(f.sum())
    b = bias_hat(f, t)
    emp = float(((t > 0) & (t < f)).mean())
    return emp, (yang_predicted_overlap(N, S, b) if b < 1 else np.nan), b, N


# --------------------------------------------------------------------------- scoring
class Scorer:
    """Scores any assignment of P(the) to the sites exactly as the paper scores a model."""

    def __init__(self, sites: pd.DataFrame = None, tpr_cases: pd.DataFrame = None):
        self.sites = load_sites() if sites is None else sites
        self.meta = load_tpr_cases() if tpr_cases is None else tpr_cases
        self.child_ref, self.mother_ref, self.adult_pop = load_tpr_references()

    @staticmethod
    def overlap_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Per speaker sample: analytical bias, empirical and predicted overlap (vectorised
        equivalents of analytical_bias / analytical_empirical_overlap / analytical_predicted_overlap)."""
        d = df.assign(q=1 - df.P_the)
        gg = d.groupby(["child_name", "speaker", "noun"])
        per = pd.DataFrame({"sp": gg.P_the.sum(), "sq": gg.q.sum(), "pp": gg.P_the.prod(), "pq": gg.q.prod()})
        per["dom"] = per[["sp", "sq"]].max(axis=1)
        per["ov"] = (1 - per.pp - per.pq).clip(lower=0)
        out = []
        for (c, spk), grp in per.groupby(level=[0, 1]):
            S = grp.sp.sum() + grp.sq.sum()
            b = grp.dom.sum() / S
            out.append(dict(child_name=c, speaker=spk, bias=b, empirical=grp.ov.mean(), N=len(grp), S=int(round(S)),
                            predicted=yang_predicted_overlap(len(grp), int(round(S)), b) if 0 < b < 1 else np.nan))
        return pd.DataFrame(out)

    @staticmethod
    def dxn_test(ov: pd.DataFrame, speaker: str) -> Dict[str, float]:
        o = ov[ov.speaker == speaker].dropna()
        t, p = ttest_rel(o.empirical, o.predicted)
        return dict(bias=o.bias.mean(), emp=o.empirical.mean(), pred=o.predicted.mean(),
                    dxn_t=t, dxn_p=p, dxn_bf01=jzs_bf01(t, len(o)))

    def tpr_tests(self, tp: pd.DataFrame, speaker: str) -> Dict[str, float]:
        """Main test (paper): one-sample t of the 12 per-dyad TPRs against the pooled adult mean.
        Also: paired against the same speakers' per-dyad human TPR."""
        c = tp[tp.speaker == speaker]
        per = c.groupby("child_name").agg(mt=("contrib", "mean"), ht=("human_switch", "mean"))
        ref = self.child_ref if speaker == "child" else self.mother_ref
        # same name normalisation as the model-TPR notebook ('warr' -> 'warren' gives 'warrenen')
        per["ref"] = [ref[n.lower().replace("warr", "warren")] for n in per.index]
        t1, p1 = ttest_1samp(per.mt, self.adult_pop)
        _, pp = ttest_rel(per.mt, per.ref)
        return dict(tpr=per.mt.mean(), tpr_sd=per.mt.std(ddof=1), tpr_t=t1, tpr_p=p1,
                    tpr_bf01=jzs_bf01(t1, len(per)), tpr_p_paired_human=pp)

    def evaluate(self, p_the: np.ndarray):
        df = self.sites[["child_name", "speaker", "sentence_id", "noun"]].copy()
        df["P_the"] = np.clip(p_the, 0, 1)
        ov = self.overlap_rows(df)
        tp = self.meta.merge(df[KEY + ["P_the"]], on=KEY, how="inner")
        tp["contrib"] = (tp.d_A - tp.P_the).abs()
        res = {}
        for spk, lab in [("child", "child sites"), ("mother", "mother sites")]:
            r = self.dxn_test(ov, spk)
            r.update(self.tpr_tests(tp, spk))
            res[lab] = r
        return res, ov, tp

    def evaluate_with_draws(self, name: str, p_the: np.ndarray, n_draws: int, rng: np.random.Generator,
                            extra: Dict = None) -> pd.DataFrame:
        """Analytical scoring (as the models) plus n_draws realised transcripts (a determiner
        actually drawn at every site); medians and pass rates (p > 0.05) of the draws."""
        extra = extra or {}
        rows = []
        res, _, _ = self.evaluate(p_the)
        for lab, r in res.items():
            rows.append(dict(model=name, sites=lab, mode="analytical (as models)", **extra.get(lab, {}), **r))
        if n_draws:
            mc = []
            for _ in range(n_draws):
                rd, _, _ = self.evaluate((rng.random(len(p_the)) < p_the).astype(float))
                for lab, r in rd.items():
                    mc.append(dict(sites=lab, **r))
            mc = pd.DataFrame(mc)
            for lab, m in mc.groupby("sites"):
                rows.append(dict(model=name, sites=lab, mode=f"sampled (median of {n_draws})",
                                 **{k: m[k].median() for k in m.columns if k != "sites"},
                                 dxn_pass_rate=(m.dxn_p > ALPHA).mean(), tpr_pass_rate=(m.tpr_p > ALPHA).mean()))
        return pd.DataFrame(rows)
