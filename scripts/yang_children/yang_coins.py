import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import ttest_rel
from determiner.analytical_metrics import jzs_bf01
from baselines.common import Scorer, load_inventory
O = "results/baselines/yang_children"
man = load_inventory().dropna(subset=["noun"])[["child_name", "speaker", "noun", "the"]]
y = pd.read_csv("output/yang_children/Yang/det_noun_locations.csv").dropna(subset=["noun"])
lo = dict(Naomi=13, Eve=18, Sarah=27, Adam=27, Peter=16, Nina=23); hi = dict(Naomi=61, Eve=27, Sarah=61, Adam=58, Peter=34, Nina=47)
y = y[[lo[c] <= a < hi[c] + 1 for c, a in zip(y.child_name, y.age_months)]]
y = y.assign(speaker=y.speaker_type, the=(y.det == "the").astype(int))[["child_name", "speaker", "noun", "the"]]
allp = pd.concat([man, y]); g = allp.groupby("noun")["the"].mean()
sites = allp[allp.speaker == "child"].copy()
def test(d):
    t, p = ttest_rel(d.emp, d.pred); return dict(n=len(d), emp=d.emp.mean(), pred=d.pred.mean(), p=p, bf01=jzs_bf01(t, len(d)))
rows = []
for name, pt in [("0.82 coin", np.full(len(sites), .82)), ("Noun-level coin", sites.noun.map(g).values)]:
    ov = Scorer.overlap_rows(sites.assign(P_the=pt, sentence_id=0)).set_index("child_name")[["empirical", "predicted"]].set_axis(["emp", "pred"], axis=1)
    M = ov.drop(list(lo)); Y = ov.loc[list(lo)]
    rows.append(dict(model=name, sample="Manchester 12", **test(M)))
    rows.append(dict(model=name, sample="Manchester 11 (no Carl)", **test(M.drop("Carl"))))
    for c in lo: rows.append(dict(model=name, sample=f"Carl -> {c}", **test(pd.concat([M.drop("Carl"), Y.loc[[c]]]))))
    rows.append(dict(model=name, sample="Manchester 11 + Yang 6", **test(pd.concat([M.drop("Carl"), Y]))))
    rows.append(dict(model=name, sample="Manchester 12 + Yang 6", **test(pd.concat([M, Y]))))
R = pd.DataFrame(rows); print(R.round(3).to_string()); R.to_csv(f"{O}/coins_dxn_with_yang.csv", index=False)
