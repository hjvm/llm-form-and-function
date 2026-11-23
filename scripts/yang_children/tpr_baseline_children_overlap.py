"""D×N overlap for the six children in the other-adult TPR baseline but not in the overlap tests
(Lara, Naima, Lily, Alex, William, Ethan), from the committed tpr-data site inventory (spaCy 3.7.1),
alone and added to the paper's 12 and to the 18 (12 + Yang's 6)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.stats import ttest_rel
from determiner.analytical_metrics import jzs_bf01
from baselines.common import counts_overlap
from baselines.zipf_audit import zipfian_fit_slope
SIX = ["lara", "naima", "lily", "alex", "william", "ethan"]
inv = pd.read_csv("results/site_inventory/tpr_data_det_noun_locations.csv").dropna(subset=["noun"])
inv = inv[inv.child_name.isin(SIX) & inv.speaker_type.isin(["child", "mother"])]
rows = []
for (c, s), g in inv.groupby(["child_name", "speaker_type"]):
    e, p, b, n = counts_overlap(g.noun.values, (g.det == "the").values)
    rows.append(dict(child_name=c.capitalize(), speaker=s, S=len(g), N=n, bias=b, emp=e, pred=p,
                     zipf_slope=zipfian_fit_slope(g.noun.value_counts().values.astype(float))))
six = pd.DataFrame(rows)
h = pd.read_csv("results/overlap/overlap_human_baseline.csv")
h = pd.DataFrame(dict(child_name=h.speaker.str.replace("_mot", ""), speaker=np.where(h.speaker.str.endswith("_mot"), "mother", "child"),
                      emp=h.empirical, pred=h.naive_zipf_predicted))
y = pd.read_csv("results/baselines/yang_children/yang_children_overlap.csv", index_col=0)
yc = pd.DataFrame(dict(child_name=y.index, speaker="child", emp=y.emp_trim.values, pred=y.pred_trim.values))
# Yang mothers (trimmed), from adults18 numbers
ym = pd.DataFrame(dict(child_name=["Naomi", "Eve", "Sarah", "Adam", "Peter", "Nina"], speaker="mother",
                       emp=[0.221, 0.215, 0.302, 0.352, 0.181, 0.490], pred=[0.191, 0.209, 0.265, 0.281, 0.122, 0.584]))
def test(d):
    t, p = ttest_rel(d.emp, d.pred); return dict(n=len(d), emp=d.emp.mean(), pred=d.pred.mean(), t=t, p=p, bf01=jzs_bf01(t, len(d)))
out = []
for spk, Y in [("child", yc), ("mother", ym)]:
    S6, M = six[six.speaker == spk], h[h.speaker == spk]
    for lab, d in [("TPR-baseline 6 alone", S6), ("Providence 5 alone (no Lara)", S6[S6.child_name != "Lara"]),
                   ("Manchester 12", M), ("Manchester 12 + these 6 (18)", pd.concat([M, S6])),
                   ("Manchester 12 + Yang 6 + these 6 (24)", pd.concat([M, Y, S6]))]:
        out.append(dict(speaker=spk, sample=lab, **test(d)))
pd.set_option("display.width", 250)
print(six.round(3).to_string()); R = pd.DataFrame(out); print(R.round(3).to_string())
six.to_csv("results/baselines/yang_children/tpr_baseline_six_overlap.csv", index=False)
R.to_csv("results/baselines/yang_children/tpr_baseline_six_dxn_tests.csv", index=False)
