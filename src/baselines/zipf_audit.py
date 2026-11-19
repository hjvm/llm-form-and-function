"""What the D×N test measures when nouns are fixed to the transcripts (docs/BASELINES.md).

decomposition()        per speaker sample: empirical overlap; the paper's prediction Y (Zipf, b-hat);
                       the conditional expectation C with the actual noun counts (b-hat and 0.82);
                       zipfian_fit() slope; frequency spectrum against an exact Zipf(1) sample with
                       the same S and the same expected number of noun types.
zipf_noun_coins()      constant coins on nouns drawn from Zipf(1) vs on the transcripts' nouns.
formula_self_check()   data generated exactly as the formula assumes (Zipf nouns, one bias b),
                       scored as in the paper (b-hat) and with the true b.
session_consistency()  P(same determiner) for two uses of a noun within vs across sessions.
overlap_by_frequency() overlap at child sites by how often the child used the noun, per model.
"""
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom, linregress, ttest_rel

from baselines.common import (ALPHA, bias_hat, conditional_overlap, counts_overlap, load_inventory,
                              yang_predicted_overlap)

BINS = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 10 ** 9)]
BIN_LABELS = ["1", "2", "3-4", "5-9", "10+"]


def zipf_p(N: int) -> np.ndarray:
    r = 1.0 / np.arange(1, N + 1)
    return r / r.sum()


def expected_types(N: int, S: int) -> float:
    return float((1 - (1 - zipf_p(N)) ** S).sum())


def calibrated_vocabulary(N_obs: int, S: int) -> int:
    """Zipf(1) vocabulary size whose S-token samples show N_obs noun types on average."""
    return int(round(brentq(lambda x: expected_types(int(x), S) - N_obs, N_obs, N_obs * 60, xtol=1)))


def zipfian_fit_slope(f: np.ndarray) -> float:
    """Same computation as zipfian_fit() in the original overlap code (overlap_list_v2.py):
    OLS slope of log frequency on log rank."""
    x = np.sort(np.log(np.arange(len(f), 0, -1)))
    y = -np.sort(-np.log(f))
    return linregress(x, y).slope


def _samples():
    inv = load_inventory().dropna(subset=["noun"])
    for (c, spk), g in inv.groupby(["child_name", "speaker"]):
        cnt = g.groupby("noun")["the"].agg(["sum", "size"])
        yield c, spk, g, cnt["size"].values, cnt["sum"].values


def decomposition() -> pd.DataFrame:
    rows = []
    for c, spk, g, f, n_the in _samples():
        N, S = len(f), int(f.sum())
        b = bias_hat(f, n_the)
        Nt = calibrated_vocabulary(N, S)
        p = zipf_p(Nt)
        seen = (1 - (1 - p) ** S).sum()
        d = dict(child_name=c, speaker=spk, N=N, S=S, bias=b,
                 empirical=float(((n_the > 0) & (n_the < f)).mean()),
                 Y_paper=yang_predicted_overlap(N, S, b),
                 C_own_bias=conditional_overlap(f, np.full(N, b)),
                 C_082=conditional_overlap(f, np.full(N, 0.82)),
                 zipfian_fit_slope=zipfian_fit_slope(f),
                 loglog_r2=linregress(np.log(np.arange(1, N + 1)), np.log(np.sort(f)[::-1])).rvalue ** 2,
                 zipf_vocabulary=Nt)
        for (lo, hi), lab in zip(BINS, BIN_LABELS):
            d[f"share_f{lab}"] = float(((f >= lo) & (f <= hi)).mean())
            d[f"zipf_share_f{lab}"] = float((binom.cdf(min(hi, S), S, p) - binom.cdf(lo - 1, S, p)).sum() / seen)
        rows.append(d)
    return pd.DataFrame(rows)


def _pass_rate(s: pd.DataFrame, pred_col: str = "pred") -> float:
    pv = s.groupby("sim").apply(lambda x: ttest_rel(x.emp, x[pred_col])[1], include_groups=False)
    return float((pv > ALPHA).mean()), float(pv.median())


def zipf_noun_coins(dec: pd.DataFrame, n_draws: int = 1000, seed: int = 1119) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    coins = {"fair": lambda b: 0.5, "0.82 toward the": lambda b: 0.82, "speaker's own bias": lambda b: b}
    real = {(c, spk): pd.factorize(pd.Series(g.noun.values))[0] for c, spk, g, _, _ in _samples()}
    rows, points = [], []
    for nouns in ["Zipf(1), calibrated vocabulary", "transcripts"]:
        for cname, qf in coins.items():
            sims = []
            for r in dec.itertuples():
                q = qf(r.bias)
                p = zipf_p(r.zipf_vocabulary)
                for k in range(n_draws):
                    ranks = rng.choice(r.zipf_vocabulary, r.S, p=p) if nouns.startswith("Zipf") else real[(r.child_name, r.speaker)]
                    e, pr, b, n = counts_overlap(ranks, rng.random(r.S) < q)
                    sims.append((k, r.child_name, r.speaker, e, pr, b, n))
            s = pd.DataFrame(sims, columns=["sim", "child_name", "speaker", "emp", "pred", "bias", "N"])
            for spk, ss in s.groupby("speaker"):
                rate, med = _pass_rate(ss)
                rows.append(dict(nouns=nouns, coin=cname, sites=spk, emp=ss.emp.mean(), pred=ss.pred.mean(),
                                 bias=ss.bias.mean(), N=ss.N.mean(), dxn_p_median=med, dxn_pass_rate=rate))
            points.append(s.groupby(["child_name", "speaker"])[["emp", "pred"]].mean().reset_index()
                          .assign(nouns=nouns, coin=cname))
    return pd.DataFrame(rows), pd.concat(points, ignore_index=True)


def formula_self_check(dec: pd.DataFrame, n_draws: int = 300, seed: int = 7) -> pd.DataFrame:
    """Zipf(1) nouns (calibrated vocabulary); each noun's favoured determiner (random direction)
    chosen with probability b = the speaker's own bias; scored with b-hat (paper) and with b."""
    rng = np.random.default_rng(seed)
    sims = []
    for r in dec.itertuples():
        Nt = r.zipf_vocabulary
        p, fav = zipf_p(Nt), rng.random(Nt) < 0.5
        for k in range(n_draws):
            ranks = rng.choice(Nt, r.S, p=p)
            hit = rng.random(r.S) < r.bias
            the = np.where(fav[ranks], hit, ~hit)
            e, pr, bh, n = counts_overlap(ranks, the)
            sims.append((k, r.child_name, r.speaker, e, pr, yang_predicted_overlap(n, r.S, r.bias), bh, r.bias))
    s = pd.DataFrame(sims, columns=["sim", "child_name", "speaker", "emp", "pred", "pred_true_b", "bias_hat", "bias_true"])
    rows = []
    for spk, ss in s.groupby("speaker"):
        r1, m1 = _pass_rate(ss, "pred")
        r2, m2 = _pass_rate(ss, "pred_true_b")
        rows.append(dict(sample_size=f"{spk}-sized", emp=ss.emp.mean(), pred_bias_hat=ss.pred.mean(),
                         pred_true_bias=ss.pred_true_b.mean(), bias_hat=ss.bias_hat.mean(), bias_true=ss.bias_true.mean(),
                         pass_rate_bias_hat=r1, pass_rate_true_bias=r2))
    points = s.groupby(["child_name", "speaker"])[["emp", "pred", "pred_true_b"]].mean().reset_index()
    return pd.DataFrame(rows), points


def session_consistency() -> pd.DataFrame:
    inv = load_inventory().dropna(subset=["noun"])
    rows = []
    for (c, spk, n), g in inv.groupby(["child_name", "speaker", "noun"]):
        if len(g) < 2:
            continue
        w = [(len(x), x["the"].sum()) for _, x in g.groupby("filename")]
        same = lambda k, th: th * (th - 1) / 2 + (k - th) * (k - th - 1) / 2
        wp, ws = sum(k * (k - 1) / 2 for k, _ in w), sum(same(k, th) for k, th in w)
        k, th = len(g), g["the"].sum()
        rows.append((c, spk, wp, ws, k * (k - 1) / 2 - wp, same(k, th) - ws))
    R = pd.DataFrame(rows, columns=["child_name", "speaker", "w_pairs", "w_same", "a_pairs", "a_same"])
    S = R.groupby(["speaker", "child_name"]).sum()
    S["p_same_within_session"] = S.w_same / S.w_pairs
    S["p_same_across_sessions"] = S.a_same / S.a_pairs
    return S.reset_index()


def overlap_by_frequency(sites: pd.DataFrame, models: Dict[str, np.ndarray], speaker: str = "child") -> pd.DataFrame:
    out = []
    for name, p in models.items():
        d = sites[["child_name", "speaker", "noun"]].assign(p=p, q=1 - p)
        g = d.groupby(["child_name", "speaker", "noun"]).agg(f=("p", "size"), pp=("p", "prod"), pq=("q", "prod")).reset_index()
        g = g[g.speaker == speaker]
        g["overlap"] = 1 - g.pp - g.pq
        g["f_bin"] = pd.cut(g.f, [0, 1, 2, 4, 9, 10 ** 6], labels=BIN_LABELS)
        r = g.groupby("f_bin", observed=True).overlap.mean()
        out += [dict(model=name, f=k, overlap=v) for k, v in r.items()]
        out.append(dict(model=name, f="mean |P(the) - 0.5| per site", overlap=float(np.abs(p[sites.speaker.values == speaker] - 0.5).mean())))
    return pd.DataFrame(out).pivot(index="f", columns="model", values="overlap")
