#!/usr/bin/env python3
"""Adult TPR reference from every adult-to-adult transition in all 24 dyads (shared preprocessing,
adult roles only: no TV/media, siblings, playmates or visiting children).

A transition is a D×N site whose noun was last mentioned, earlier in the same session, by a DIFFERENT
adult speaker. Three nested definitions:
  A  other adult <- mother                      (the paper's definition)
  B  A + mother <- other adult                  (both directions between mother and other adults)
  C  B + other adult <- a different other adult (every adult-to-adult transition)
The reference is the pooled switch rate (as in the paper); per-dyad rates are reported for spread.
Writes results/baselines/all_children/adult_tpr_reference.{csv,json} and adult_tpr_cases.csv.gz.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd
from determiner.paths import chdir_to_root

chdir_to_root()
OUT = Path("results/baselines/all_children")
M12 = ["Anne", "Aran", "Becky", "Carl", "Dominic", "Gail", "Joel", "John", "Liz", "Nicole", "Ruth", "Warren"]
invs = []
for f in ["manchester_all_speakers", "yang", "tpr6"]:
    d = pd.read_csv(OUT / f"{f}_det_noun_locations.csv.gz")
    d["child_name"] = d.child_name.str.capitalize()
    invs.append(d)
inv = pd.concat(invs, ignore_index=True)
inv = inv[inv.noun.notna()]

cases = []
for (child, sess), g in inv.groupby(["child_name", "filename"], sort=False):
    last = {}
    for r in g.sort_values(["line_num", "pair_index"]).itertuples():
        prev = last.get(r.noun)
        if prev is not None and r.speaker_type in ("mother", "other_adult") and prev[1] in ("mother", "other_adult") and prev[0] != r.speaker:
            kind = ("A" if (r.speaker_type, prev[1]) == ("other_adult", "mother") else
                    "B" if (r.speaker_type, prev[1]) == ("mother", "other_adult") else "C")
            cases.append((child, sess, r.speaker, r.speaker_type, prev[0], prev[1], r.noun, r.det, prev[2], kind))
        last[r.noun] = (r.speaker, r.speaker_type, r.det)
C = pd.DataFrame(cases, columns=["child_name", "filename", "speaker", "speaker_type", "prev_speaker", "prev_speaker_type",
                                 "noun", "det", "prev_det", "kind"])
C["switch"] = (C.det != C.prev_det).astype(int)
C.to_csv(OUT / "adult_tpr_cases.csv.gz", index=False)

rows, refs = [], {}
for name, kinds in [("A: other adult <- mother", "A"), ("B: + mother <- other adult", "AB"), ("C: all adult-to-adult", "ABC")]:
    for sample, kids in [("all 24", None), ("Manchester 12", M12)]:
        c = C[C.kind.isin(list(kinds)) & (True if kids is None else C.child_name.isin(kids))]
        per = c.groupby("child_name").switch.agg(["mean", "size"])
        per10 = per[per["size"] >= 10]
        rows.append(dict(definition=name, sample=sample, pooled=c.switch.mean(), n_cases=len(c), n_dyads=len(per),
                         dyad_mean=per10["mean"].mean(), dyad_sd=per10["mean"].std(), dyad_min=per10["mean"].min(),
                         dyad_max=per10["mean"].max(), n_dyads_ge10=len(per10)))
        if sample == "all 24":
            refs[name] = float(c.switch.mean())
R = pd.DataFrame(rows); R.to_csv(OUT / "adult_tpr_reference.csv", index=False)
json.dump(refs, open(OUT / "adult_tpr_reference.json", "w"), indent=1)
pd.set_option("display.width", 200)
print(R.round(4).to_string())
per = C[C.kind.isin(["A", "B"])].groupby(["child_name", "kind"]).switch.agg(["mean", "size"]).unstack()
print(per.round(3).to_string())
