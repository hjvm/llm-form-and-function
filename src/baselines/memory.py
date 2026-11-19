"""Memory model after Yang (2013, PNAS 110:6324, 'Role of memory in language learning').

For a speaker who produced S D×N tokens, draw S (determiner, noun) pairs with replacement, in
proportion to frequency, from the adult input; compare the sample's overlap with the speaker's
own empirical overlap (Yang's comparison) and score it with the D×N test.

Input pool: every Manchester adult's D×N tokens (the 12 mothers from the site inventory plus
investigators, fathers, a grandmother and one other adult; siblings and friends of unknown age
excluded), 58,599 tokens in all.

Closed form (Charles Yang, p.c.): the expected overlap is the unconditional overlap with pi = the
pool's noun frequencies and t = the pool's P(the | noun); the simulation is a Monte Carlo
estimate of it. With the speaker's own nouns and counts held fixed instead, the same
determiner statistics give conditional_overlap(f, t).
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from baselines.common import (ALPHA, BASELINE_RESULTS, conditional_overlap, counts_overlap, load_inventory,
                              unconditional_overlap)

OTHER_ADULTS = BASELINE_RESULTS / "inputs" / "manchester_other_adult_det_noun.csv"


def adult_pool() -> pd.DataFrame:
    inv = load_inventory()
    mot = inv[inv.speaker == "mother"][["child_name", "noun", "the"]]
    oth = pd.read_csv(OTHER_ADULTS)
    oth["the"] = (oth.det == "the").astype(int)
    return pd.concat([mot, oth[["child_name", "noun", "the"]]], ignore_index=True).dropna(subset=["noun"])


def run(n_draws: int = 1000, seed: int = 13) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inv = load_inventory().dropna(subset=["noun"])
    pool = adult_pool()
    pn, pt = pool.noun.values, pool.the.values
    P = pool.groupby("noun").the.agg(["sum", "size"])
    pi, t = (P["size"] / P["size"].sum()).values, (P["sum"] / P["size"]).values
    p_the_noun = pd.Series(t, P.index)
    rng = np.random.default_rng(seed)
    per, sims = [], []
    for (c, spk), g in inv.groupby(["child_name", "speaker"]):
        S = len(g)
        real_emp, real_pred, real_b, N = counts_overlap(g.noun.values, g.the.values)
        cnt = g.groupby("noun").size()
        tt = p_the_noun.reindex(cnt.index)
        per.append(dict(child_name=c, speaker=spk, S=S, N=N, real_emp=real_emp,
                        closed_form=unconditional_overlap(pi, t, S),
                        expected_types=float((1 - (1 - pi) ** S).sum()),
                        conditional_on_speaker_nouns=conditional_overlap(cnt.values, tt.fillna(pool.the.mean()).values),
                        speaker_nouns_unseen_in_pool=float(tt.isna().mean())))
        for k in range(n_draws):
            idx = rng.integers(0, len(pn), S)
            e, p, b, n = counts_overlap(pn[idx], pt[idx])
            sims.append((k, c, spk, e, p, b, n))
    per = pd.DataFrame(per)
    sims = pd.DataFrame(sims, columns=["sim", "child_name", "speaker", "empirical", "predicted", "bias", "N"])
    mean = sims.groupby(["child_name", "speaker"])[["empirical", "predicted", "bias", "N"]].mean().add_prefix("draws_")
    per = per.merge(mean.reset_index(), on=["child_name", "speaker"])
    rows = []
    for spk, s in sims.groupby("speaker"):
        q = per[per.speaker == spk]
        pv = s.groupby("sim").apply(lambda x: ttest_rel(x.empirical, x.predicted)[1], include_groups=False)
        rows.append(dict(model="memory model (all Manchester adults)", sites=f"{spk} sites",
                         emp_draws=q.draws_empirical.mean(), emp_closed_form=q.closed_form.mean(),
                         max_abs_diff_closed_form_vs_draws=float((q.closed_form - q.draws_empirical).abs().max()),
                         pred=q.draws_predicted.mean(), bias=q.draws_bias.mean(), N_sample=q.draws_N.mean(),
                         N_speaker=q.N.mean(), real_emp=q.real_emp.mean(),
                         yang_wilcoxon_p_less=wilcoxon(q.closed_form, q.real_emp, alternative="less").pvalue,
                         conditional_on_speaker_nouns=q.conditional_on_speaker_nouns.mean(),
                         p_real_vs_conditional=ttest_rel(q.real_emp, q.conditional_on_speaker_nouns)[1],
                         dxn_p_median=float(pv.median()), dxn_pass_rate=float((pv > ALPHA).mean())))
    return pd.DataFrame(rows), per
