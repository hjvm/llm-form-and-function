#!/usr/bin/env python3
"""TPR for the paper's 49 models and the baselines at child sites with the 1-turn repeat exclusion.
A child TPR case is dropped when the child's utterance contains the mother's antecedent utterance word for
word and the mother spoke in the immediately preceding turn. Exclusion is defined on the human transcript,
so the same sites are dropped for every model. Reference: adults-with-adults transitions with the same
exclusion (tpr_repeat_window.csv, window 1). Paper's 12 children (committed inventory, legacy line numbers).
Needs results/baselines/all_children/paper49_child_pthe.csv.gz (child-site P(the) of the 49 models, Mac).
Writes results/baselines/all_children/models_tpr_repeat1.csv."""
import re, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp
from determiner.paths import chdir_to_root
from determiner.tpr import prepare_tpr_cases
from determiner.analytical_metrics import jzs_bf01
from baselines.common import load_tpr_cases, Scorer
from baselines import coins, ngram
chdir_to_root(); OUT = Path("results/baselines/all_children")
inv = pd.read_csv("results/site_inventory/manchester_det_noun_locations.csv")
cases = prepare_tpr_cases(inv)
cases = cases[cases.speaker == "child"].copy()
paper = load_tpr_cases(); paper = paper[paper.speaker == "child"]
# the paper's cached case set; details from prepare_tpr_cases (5,560 of 5,562 match; 2 differ by sort ties
# and are kept, never dropped)
cases = paper[["sentence_id", "child_name", "previous_det"]].merge(
    cases[["sentence_id", "filename", "original_sentence", "previous_sentence", "previous_line_num", "target_line_num"]],
    on="sentence_id", how="left")
# legacy line numbering: row index of the paper loader's tab-separated read (all tiers), per file
speakers = {}
for f in inv.filename.unique():
    t = pd.read_csv(f, sep="\t", on_bad_lines="skip", names=["speaker", "sentence"]).dropna()
    speakers[os.path.basename(f)] = t.speaker.values
def turns_between(fname, a, b):
    s = [x for x in speakers[fname][a:b + 1] if str(x).startswith("*")]
    return sum(1 for i in range(1, len(s)) if s[i] != s[i - 1])
norm = lambda s: re.sub(r"[^a-z' ]", "", str(s).lower()).split()
def is_repeat(r):
    if pd.isna(r.previous_sentence): return False
    mine, theirs = norm(r.original_sentence), norm(r.previous_sentence); k = len(theirs)
    return k > 0 and any(mine[i:i + k] == theirs for i in range(len(mine) - k + 1))
cases["repeat"] = [is_repeat(r) for r in cases.itertuples()]
cases["d_turn"] = [99 if pd.isna(r.filename) else turns_between(r.filename, int(r.previous_line_num), int(r.target_line_num)) for r in cases.itertuples()]
cases["drop"] = cases.repeat & (cases.d_turn <= 1)
cases["d_A"] = (cases.previous_det.str.lower() == "the").astype(int)
print(f"child cases {len(cases)}; repeats {cases.repeat.mean():.1%}; dropped (repeat within 1 turn) {cases['drop'].mean():.1%}")
W = pd.read_csv(OUT / "tpr_repeat_window.csv")
REF1 = float(W[(W.window == "1") & (W["sample"] == "All 24") & (W.who == "child<-mother")].ref.iloc[0])
REF0 = 0.2221
# predictions
P = pd.read_csv(OUT / "paper49_child_pthe.csv.gz")
dup = P.groupby("model")["type"].nunique(); dup = set(dup[dup > 1].index)  # gpt-bert "mixed" run as ar and mlm
P["name"] = [f"{m} ({t})" if m in dup else m for m, t in zip(P.model, P.type)]
preds = {m: g.set_index("sentence_id").P_the for m, g in P.groupby("name")}
sc = Scorer(); sid = sc.sites.sentence_id.values
for lab, fn in [("Fair coin", "fair (P(the)=0.5)"), ("0.82 coin", "0.82 toward 'the'"), ("Noun-level coin", "noun-level bias (corpus P(the|noun))")]:
    preds[lab] = pd.Series(coins.COINS[fn](sc.sites), index=sid)
for m, lab in ngram.NGRAM_MODELS.items():
    preds[lab] = pd.Series(ngram.p_the(pd.read_csv(f"results/baselines/ngram/{m}_cac_predictions.csv.gz"), sid), index=sid)
preds["Humans"] = pd.Series((inv.det == "the").astype(float).values, index=inv.index)
rows = []
for name, p in preds.items():
    for excl, ref in [("none", REF0), ("1 turn", REF1)]:
        c = cases if excl == "none" else cases[~cases["drop"]]
        e = (c.d_A - c.sentence_id.map(p)).abs()
        per = e.groupby(c.child_name).mean()
        t, pv = ttest_1samp(per, ref)
        rows.append(dict(model=name, exclusion=excl, ref=ref, tpr=per.mean(), p=pv, bf01=jzs_bf01(t, len(per)), passes=pv > 0.05))
R = pd.DataFrame(rows); R.to_csv(OUT / "models_tpr_repeat1.csv", index=False)
summ = pd.read_csv("results/tpr/tpr_model_summary_analytical.csv").set_index("model_short")
chk = R[(R.exclusion == "none")].set_index("model").tpr
both = [m for m in chk.index if m in summ.index]
print("max |ours - paper| model TPR:", float((chk[both] - summ.loc[both, "mean_model_tpr"]).abs().max()))
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
piv = R.pivot(index="model", columns="exclusion", values=["tpr", "p"]).round(3)
print(piv.to_string())
print("passing:", R.groupby("exclusion").passes.sum().to_dict())
