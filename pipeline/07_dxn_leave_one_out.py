#!/usr/bin/env python3
"""D×N test with Carl left out, and leave-one-dyad-out, for humans, every paper model (child sites) and the baselines.
Run from the repo root after 06_baselines.py. Writes results/baselines/dxn_leave_one_out.csv."""
import sys, glob; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import ttest_rel
from determiner.analytical_metrics import jzs_bf01
from baselines.common import Scorer
from baselines import coins, ngram
rows = []
def add(name, group, df):  # df: child_name, emp, pred (12 rows)
    df = df.set_index("child_name")
    def test(d):
        t, p = ttest_rel(d.emp, d.pred); return t, p, jzs_bf01(t, len(d))
    t, p, bf = test(df)
    tc, pc, bfc = test(df.drop("Carl"))
    loo = [test(df.drop(c)) for c in df.index]
    lp = [x[1] for x in loo]
    rows.append(dict(source=name, group=group, carl_emp=df.emp["Carl"], carl_pred=df.pred["Carl"],
        carl_resid=df.emp["Carl"] - df.pred["Carl"], biggest_resid_child=(df.emp - df.pred).abs().idxmax(),
        p_all=p, bf_all=bf, p_noCarl=pc, bf_noCarl=bfc, loo_p_min=min(lp), loo_p_max=max(lp),
        loo_drop_minp=df.index[int(np.argmin(lp))], loo_drop_maxp=df.index[int(np.argmax(lp))]))
h = pd.read_csv("results/overlap/overlap_human_baseline.csv")
h["child_name"] = h.speaker.str.replace("_mot", ""); h["spk"] = np.where(h.speaker.str.endswith("_mot"), "mother", "child")
for spk, nm in [("child", "Children"), ("mother", "Caretakers")]:
    add(nm, "human", h[h.spk == spk].rename(columns={"empirical": "emp", "naive_zipf_predicted": "pred"}))
for f in sorted(glob.glob("results/overlap/*/*/analytical_overlap_summary.csv")):
    d = pd.read_csv(f); d = d[d.speaker == "child"]
    add(d.model_name.iloc[0], "paper LM (" + d.model_type.iloc[0] + ")", d.rename(columns={"empirical": "emp", "naive_predicted": "pred"}))
sc = Scorer()
bl = {"Fair coin": coins.COINS["fair (P(the)=0.5)"](sc.sites), "0.82 coin": coins.COINS["0.82 toward 'the'"](sc.sites),
      "Noun-level coin": coins.COINS["noun-level bias (corpus P(the|noun))"](sc.sites)}

for m, lab in ngram.NGRAM_MODELS.items():
    bl[lab] = ngram.p_the(pd.read_csv(f"results/baselines/ngram/{m}_cac_predictions.csv.gz"), sc.sites.sentence_id)
for lab, p in bl.items():
    _, ov, _ = sc.evaluate(p); ov = ov[ov.speaker == "child"]
    add(lab, "baseline", ov.rename(columns={"empirical": "emp", "predicted": "pred"}))
m = pd.read_csv("results/baselines/memory_per_dyad.csv"); m = m[m.speaker == "child"]
add("Memory model (draw means)", "baseline", m.rename(columns={"draws_empirical": "emp", "draws_predicted": "pred"}))
R = pd.DataFrame(rows)
R["pass_all"] = R.p_all > 0.05; R["pass_noCarl"] = R.p_noCarl > 0.05
R.to_csv("results/baselines/dxn_leave_one_out.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_rows", 100)
print(R.round(3).to_string())
