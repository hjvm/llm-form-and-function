import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import ttest_rel
from determiner.analytical_metrics import jzs_bf01
from baselines.common import counts_overlap
O = "results/baselines/yang_children"
def ov(g):
    e, p, b, n = counts_overlap(g.noun.values, (g.det == "the").values); return pd.Series(dict(S=len(g), N=n, bias=b, emp=e, pred=p))
# validation: spaCy 3.8 re-extraction vs committed 3.7.1 inventory
m38 = pd.read_csv("output/yang_children/Manchester/det_noun_locations.csv").dropna(subset=["noun"])
print("spaCy 3.8 re-extraction of Manchester children:"); print(m38[m38.speaker_type == "child"].groupby("child_name").apply(ov, include_groups=False).round(3))
yang_tab = pd.DataFrame({"S": [884, 831, 2453, 3729, 2873, 4542], "N": [349, 283, 640, 780, 480, 660], "yang_exp": [.218, .254, .288, .337, .422, .451], "yang_emp": [.198, .216, .292, .323, .404, .467],
                         "lo": [13, 18, 27, 27, 16, 23], "hi": [61, 27, 61, 58, 34, 47]}, index=["Naomi", "Eve", "Sarah", "Adam", "Peter", "Nina"])
y = pd.read_csv("output/yang_children/Yang/det_noun_locations.csv").dropna(subset=["noun"]); y = y[y.speaker_type == "child"]
full = y.groupby("child_name").apply(ov, include_groups=False)
trim = y[[yang_tab.lo[c] <= a < yang_tab.hi[c] + 1 for c, a in zip(y.child_name, y.age_months)]].groupby("child_name").apply(ov, include_groups=False)
T = yang_tab.join(full.add_suffix("_full")).join(trim.add_suffix("_trim"))
print(T.round(3).to_string()); T.to_csv(f"{O}/yang_children_overlap.csv")
h = pd.read_csv("results/overlap/overlap_human_baseline.csv"); h = h[~h.speaker.str.endswith("_mot")].set_index("speaker")[["empirical", "naive_zipf_predicted"]]
h.columns = ["emp", "pred"]
def test(d):
    t, p = ttest_rel(d.emp, d.pred); return dict(n=len(d), mean_emp=d.emp.mean(), mean_pred=d.pred.mean(), t=t, p=p, bf01=jzs_bf01(t, len(d)))
rows = []
for ver in ["trim", "full"]:
    Y = T[[f"emp_{ver}", f"pred_{ver}"]].set_axis(["emp", "pred"], axis=1)
    if ver == "trim": rows.append(dict(sample="Manchester 12 (paper)", **test(h)))
    if ver == "trim": rows.append(dict(sample="Manchester without Carl (11)", **test(h.drop("Carl"))))
    rows.append(dict(sample=f"Yang's 6 alone ({ver})", **test(Y)))
    for c in Y.index:
        rows.append(dict(sample=f"Carl replaced by {c} ({ver})", **test(pd.concat([h.drop("Carl"), Y.loc[[c]]]))))
    rows.append(dict(sample=f"Manchester 11 + Yang 6 ({ver})", **test(pd.concat([h.drop("Carl"), Y]))))
    rows.append(dict(sample=f"Manchester 12 + Yang 6 ({ver})", **test(pd.concat([h, Y]))))
R = pd.DataFrame(rows); print(R.round(3).to_string()); R.to_csv(f"{O}/children_dxn_with_yang.csv", index=False)
