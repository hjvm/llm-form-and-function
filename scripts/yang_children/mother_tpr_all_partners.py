#!/usr/bin/env python3
"""Mothers' TPR over all their transitions (following the child AND following other adults), per dyad,
against the leave-one-dyad-out adults-with-adults reference; plus the child-only version and a repetition
check (does the mother's mention repeat the child's utterance?). All-speaker inventories from the shared
pipeline; Yang's six trimmed to his age ranges for the mothers' cases (reference uses every session).
Writes results/baselines/all_children/mother_tpr_all_partners.csv."""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp
from determiner.analytical_metrics import jzs_bf01
from determiner.paths import chdir_to_root
chdir_to_root(); OUT = Path("results/baselines/all_children")
LO = dict(Naomi=13, Eve=18, Sarah=27, Adam=27, Peter=16, Nina=23); HI = dict(Naomi=61, Eve=27, Sarah=61, Adam=58, Peter=34, Nina=47)
M12 = ["Anne", "Aran", "Becky", "Carl", "Dominic", "Gail", "Joel", "John", "Liz", "Nicole", "Ruth", "Warren"]
inv = []
for f in ["manchester_all_speakers", "yang", "tpr6"]:
    d = pd.read_csv(OUT / f"{f}_det_noun_locations.csv.gz"); d["child_name"] = d.child_name.str.capitalize(); inv.append(d)
inv = pd.concat(inv, ignore_index=True); inv = inv[inv.noun.notna()]
inv["in_age"] = [True if c not in LO else LO[c] <= a < HI[c] + 1 for c, a in zip(inv.child_name, inv.get("age_months", np.nan))]
norm = lambda s: re.sub(r"[^a-z' ]", "", str(s).lower()).split()
cases = []
for (child, sess), g in inv.groupby(["child_name", "filename"], sort=False):
    last = {}
    for r in g.sort_values(["line_num", "pair_index"]).itertuples():
        p = last.get(r.noun)
        if p is not None and r.speaker_type == "mother" and p["speaker"] != r.speaker and p["type"] in ("child", "other_adult") and r.in_age:
            mine, theirs = norm(r.sentence), norm(p["sentence"])
            k = len(theirs)
            repeat = k > 0 and any(mine[i:i + k] == theirs for i in range(len(mine) - k + 1))
            cases.append(dict(child_name=child, prev_type=p["type"], switch=int(r.det != p["det"]), repeat=repeat,
                              gap=r.line_num - p["line_num"]))
        last[r.noun] = dict(speaker=r.speaker, type=r.speaker_type, det=r.det, sentence=r.sentence, line_num=r.line_num)
C = pd.DataFrame(cases)
A = pd.read_csv(OUT / "adult_tpr_cases.csv.gz")
rows = []
for lab, kids in [("All 24", None), ("Manchester 12", M12)]:
    for which, sel in [("following child only", C.prev_type == "child"), ("following child or other adult", C.prev_type.notna())]:
        c = C[sel & (True if kids is None else C.child_name.isin(kids))]
        per = c.groupby("child_name").switch.mean()
        loo = pd.Series({k: A[A.child_name != k].switch.mean() for k in per.index})
        t, p = ttest_1samp(per - loo, 0)
        rows.append(dict(sample=lab, cases=which, n_cases=len(c), tpr=per.mean(), loo_ref=loo.mean(), p=p, bf01=jzs_bf01(t, len(per))))
R = pd.DataFrame(rows); R.to_csv(OUT / "mother_tpr_all_partners.csv", index=False)
pd.set_option("display.width", 200); print(R.round(4).to_string())
ch = C[C.prev_type == "child"]
print("\nRepetition (mother following the child): repeats child's whole utterance in", f"{ch.repeat.mean():.1%}", "of", len(ch), "cases")
print(ch.groupby("repeat").switch.agg(["mean", "size"]).round(3).to_string())
print("adjacent (next utterance) share:", f"{(ch.gap == 1).mean():.1%}; following other adult: repeat {C[C.prev_type=='other_adult'].repeat.mean():.1%}")
print(C.groupby("prev_type").switch.agg(["mean", "size"]).round(3).to_string())
