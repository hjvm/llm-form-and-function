#!/usr/bin/env python3
"""Robustness of the main adult TPR reference (C: all adult-to-adult transitions, 24 dyads):
leave-one-dyad-out references, and the reference without transitions where the mother responds.
Needs adult_tpr_reference.py and all_children.py to have run. Writes adult_reference_loo.csv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import pandas as pd
from scipy.stats import ttest_1samp
from determiner.analytical_metrics import jzs_bf01
from determiner.paths import chdir_to_root
chdir_to_root(); OUT = Path("results/baselines/all_children")
A = pd.read_csv(OUT / "adult_tpr_cases.csv.gz"); c = pd.read_csv(OUT / "tpr_cases_24.csv.gz")
M12 = ["Anne", "Aran", "Becky", "Carl", "Dominic", "Gail", "Joel", "John", "Liz", "Nicole", "Ruth", "Warren"]
no_mother = A[A.speaker_type == "other_adult"].switch.mean()
rows = []
for spk in ["mother", "child"]:
    for lab, kids in [("All 24", None), ("Manchester 12", M12)]:
        t = c[c.speaker == spk]
        t = t if kids is None else t[t.child_name.isin(kids)]
        per = t.groupby("child_name").human_switch.mean()
        loo = pd.Series({k: A[A.child_name != k].switch.mean() for k in per.index})
        t1, p1 = ttest_1samp(per - loo, 0); t2, p2 = ttest_1samp(per, no_mother)
        rows.append(dict(speaker=spk, sample=lab, tpr=per.mean(), loo_ref_mean=loo.mean(), loo_p=p1, loo_bf01=jzs_bf01(t1, len(per)),
                         ref_without_mother_responding=no_mother, p_without_mother_responding=p2))
R = pd.DataFrame(rows); R.to_csv(OUT / "adult_reference_loo.csv", index=False); print(R.round(4).to_string())
print("mother responding to another adult:", round(A[A.kind == "B"].switch.mean(), 4), "n", (A.kind == "B").sum(),
      "| other adults responding:", round(no_mother, 4), "n", (A.speaker_type == "other_adult").sum())
