#!/usr/bin/env python3
"""Repetition in TPR transitions, by distance. A transition is a repeat when the responder's utterance
contains the antecedent utterance word for word. Distance = utterances (main tiers, any speaker) and turns
(speaker changes) between the two. Repeat-excluded TPR with the repeat window capped at 1, 2, 3 turns.
Writes results/baselines/all_children/tpr_repeat_{distance,window}.csv."""
import re, sys, bisect
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp
from determiner.analytical_metrics import jzs_bf01
from determiner.paths import chdir_to_root
from determiner.preprocess import load_corpus
chdir_to_root(); OUT = Path("results/baselines/all_children")
LO = dict(Naomi=13, Eve=18, Sarah=27, Adam=27, Peter=16, Nina=23); HI = dict(Naomi=61, Eve=27, Sarah=61, Adam=58, Peter=34, Nina=47)
M12 = ["Anne", "Aran", "Becky", "Carl", "Dominic", "Gail", "Joel", "John", "Liz", "Nicole", "Ruth", "Warren"]
ALL = ("child", "mother", "other_adult", "other")
# speaker sequence per session, for turn distances
turns = {}
for prof, path, kids in [("manchester", "data/CHILDES/Manchester", None), ("brown", "data/CHILDES/Yang", ["Adam", "Eve", "Sarah"]),
                         ("bloom", "data/CHILDES/Yang", ["Peter"]), ("sachs", "data/CHILDES/Yang", ["Naomi"]),
                         ("suppes", "data/CHILDES/Yang", ["Nina"]), ("tpr_data", "data/CHILDES/tpr6", ["lara", "naima", "lily", "alex", "william", "ethan"])]:
    for child, df in load_corpus(prof, path, kids, keep=ALL).items():
        for f, g in df.groupby("filename", sort=False):
            g = g.sort_values("line_num")
            turn_id = (g.speaker != g.speaker.shift()).cumsum().values
            turns[(child.capitalize(), f)] = (g.line_num.values, turn_id)
def turn_of(key, ln):
    lines, tid = turns[key]; i = bisect.bisect_right(lines, ln) - 1
    return tid[max(i, 0)]

inv = []
for f in ["manchester_all_speakers", "yang", "tpr6"]:
    d = pd.read_csv(OUT / f"{f}_det_noun_locations.csv.gz"); d["child_name"] = d.child_name.str.capitalize(); inv.append(d)
inv = pd.concat(inv, ignore_index=True); inv = inv[inv.noun.notna()]
inv["in_age"] = [True if c not in LO else LO[c] <= a < HI[c] + 1 for c, a in zip(inv.child_name, inv.get("age_months", np.nan))]
norm = lambda s: re.sub(r"[^a-z' ]", "", str(s).lower()).split()
ADULT = ("mother", "other_adult")
cases = []
for (child, sess), g in inv.groupby(["child_name", "filename"], sort=False):
    last = {}
    for r in g.sort_values(["line_num", "pair_index"]).itertuples():
        p = last.get(r.noun)
        if p is not None and p["speaker"] != r.speaker:
            if r.speaker_type in ADULT and p["type"] in ADULT: role = "adult<-adult"
            elif r.in_age and r.speaker_type == "mother" and p["type"] == "child": role = "mother<-child"
            elif r.in_age and r.speaker_type == "child" and p["type"] == "mother": role = "child<-mother"
            else: role = None
            if role:
                mine, theirs = norm(r.sentence), norm(p["sentence"]); k = len(theirs)
                rep = k > 0 and any(mine[i:i + k] == theirs for i in range(len(mine) - k + 1))
                cases.append(dict(role=role, child_name=child, switch=int(r.det != p["det"]), repeat=rep,
                                  d_utt=r.line_num - p["line_num"], d_turn=turn_of((child, sess), r.line_num) - turn_of((child, sess), p["line_num"])))
        last[r.noun] = dict(speaker=r.speaker, type=r.speaker_type, det=r.det, sentence=r.sentence, line_num=r.line_num)
C = pd.DataFrame(cases)
C.to_csv(OUT / "tpr_cases_with_repeats.csv.gz", index=False)

# distance distribution among repeats
rows = []
for role, g in C[C.repeat].groupby("role"):
    allr = C[C.role == role]
    rows.append(dict(role=role, n_cases=len(allr), n_repeats=len(g), repeat_share=len(g) / len(allr),
                     **{f"utt<={k}": (g.d_utt <= k).mean() for k in (1, 2, 3, 5, 10)},
                     **{f"turn<={k}": (g.d_turn <= k).mean() for k in (1, 2, 3, 5, 10)},
                     utt_median=g.d_utt.median(), turn_median=g.d_turn.median(), switch_in_repeats=g.switch.mean()))
D = pd.DataFrame(rows); D.to_csv(OUT / "tpr_repeat_distance.csv", index=False)
pd.set_option("display.width", 250); print(D.round(3).to_string())

def test(c, ref_cases, kids):
    c = c if kids is None else c[c.child_name.isin(kids)]
    per = c.groupby("child_name").switch.mean()
    loo = pd.Series({k: ref_cases[ref_cases.child_name != k].switch.mean() for k in per.index})
    t, p = ttest_1samp(per - loo, 0); return per.mean(), loo.mean(), p, jzs_bf01(t, len(per))
rows = []
for win in ["none", 1, 2, 3, "any"]:
    drop = (lambda d: np.zeros(len(d), bool)) if win == "none" else (lambda d: d.repeat.values) if win == "any" else (lambda d, w=win: (d.repeat & (d.d_turn <= w)).values)
    K = C[~drop(C)]; ref = K[K.role == "adult<-adult"]
    for lab, kids in [("All 24", None), ("Manchester 12", M12)]:
        for who in ["mother<-child", "child<-mother"]:
            tpr, r, p, bf = test(K[K.role == who], ref, kids)
            rows.append(dict(window=win, filter="symmetric", sample=lab, who=who, tpr=tpr, ref=r, p=p, bf01=bf))
        if win != "none":  # mothers' repeats only, unfiltered reference
            M = C[(C.role == "mother<-child")]; M = M[~drop(M)]
            tpr, r, p, bf = test(M, C[C.role == "adult<-adult"], kids)
            rows.append(dict(window=win, filter="mothers only", sample=lab, who="mother<-child", tpr=tpr, ref=r, p=p, bf01=bf))
W = pd.DataFrame(rows); W.to_csv(OUT / "tpr_repeat_window.csv", index=False); print(W.round(3).to_string())
