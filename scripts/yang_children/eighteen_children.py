"""Every baseline on 18 children (Manchester 12 + Yang's 6), child sites, beside the paper's 12.
Run from the repo root after yang_extract.py and (for the 5-grams) Step 1 on Yang's children:
    python pipeline/01_run_inference.py --models <json> --corpus ./data/CHILDES/Yang --output output/yang_discourse
"""
import sys; sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from determiner.analytical_metrics import jzs_bf01
from determiner.tpr import prepare_tpr_cases
from baselines.common import Scorer, counts_overlap, load_sites, load_tpr_cases
from baselines import ngram
from baselines.memory import adult_pool

OUT = Path("results/baselines/yang_children")
LO = dict(Naomi=13, Eve=18, Sarah=27, Adam=27, Peter=16, Nina=23); HI = dict(Naomi=61, Eve=27, Sarah=61, Adam=58, Peter=34, Nina=47)
ADULT = 0.2154798761609907
OFF = 10 ** 7

# ---- sites and TPR cases
yinv = pd.read_csv("output/yang_discourse/det_noun_locations.csv")
keep = np.array([LO[c] <= a < HI[c] + 1 for c, a in zip(yinv.child_name, yinv.age_months)])
ycases = prepare_tpr_cases(yinv)
ycases = ycases[ycases.sentence_id.isin(np.where(keep)[0])]
ycases["d_A"] = ycases.previous_det.str.lower().map({"a": 0, "an": 0, "the": 1}).astype(int)
ycases["human_switch"] = ((ycases.original_det == "the").astype(int) != ycases.d_A).astype(int)
ycases["sentence_id"] += OFF
ysites = yinv[keep & yinv.noun.notna().values].reset_index().rename(columns={"index": "sentence_id", "speaker_type": "spk"})
ysites = pd.DataFrame(dict(sentence_id=ysites.sentence_id + OFF, child_name=ysites.child_name, speaker=ysites.spk,
                           noun=ysites.noun, orig_the=(ysites.det == "the").astype(int)))
msites = load_sites(); mcases = load_tpr_cases()
sites = pd.concat([msites[["sentence_id", "child_name", "speaker", "noun", "orig_the"]].dropna(subset=["noun"]), ysites], ignore_index=True)
sites = sites[sites.speaker.isin(["child", "mother"])]
cases = pd.concat([mcases, ycases[mcases.columns]], ignore_index=True)
pthe_noun = sites.groupby("noun").orig_the.mean()   # pooled over the 18 dyads, children and mothers

def evaluate(p_the, children):
    d = sites.assign(P_the=np.clip(p_the, 0, 1))
    d = d[(d.speaker == "child") & d.child_name.isin(children)]
    ov = Scorer.overlap_rows(d).set_index("child_name")
    t, p = ttest_rel(ov.empirical, ov.predicted)
    tp = cases[(cases.speaker == "child") & cases.child_name.isin(children)].merge(d[["sentence_id", "P_the"]], on="sentence_id")
    per = (tp.d_A - tp.P_the).abs().groupby(tp.child_name).mean()
    t1, p1 = ttest_1samp(per, ADULT)
    return dict(n=len(ov), emp=ov.empirical.mean(), pred=ov.predicted.mean(), dxn_p=p, dxn_bf01=jzs_bf01(t, len(ov)),
                tpr=per.mean(), tpr_p=p1, n_tpr_cases=len(tp))

M12 = sorted(msites.child_name.unique()); A18 = M12 + list(LO)
models = {"Children (realised)": sites.orig_the.values.astype(float), "Fair coin": np.full(len(sites), 0.5),
          "0.82 coin": np.full(len(sites), 0.82), "Noun-level coin": sites.noun.map(pthe_noun).values}
for m, lab in ngram.NGRAM_MODELS.items():
    yp = Path("output/yang_discourse/ar") / m
    if not yp.exists():
        print(f"{m}: no predictions on Yang's children yet"); continue
    mp = pd.read_csv(f"results/baselines/ngram/{m}_cac_predictions.csv.gz")[["sentence_id", "P_the"]]
    ypp = ngram.collect_predictions(m, Path("output/yang_discourse/ar"))[["sentence_id", "P_the"]].assign(sentence_id=lambda x: x.sentence_id + OFF)
    models[lab] = sites.sentence_id.map(pd.concat([mp, ypp]).set_index("sentence_id").P_the).values
rows = []
for name, p in models.items():
    for lab, ch in [("Manchester 12", M12), ("Manchester 12 + Yang 6", A18), ("Manchester 11 + Yang 6", [c for c in A18 if c != "Carl"])]:
        sub = sites.child_name.isin(ch) & (sites.speaker == "child")
        if np.isnan(p[sub.values]).any():
            continue
        rows.append(dict(model=name, sample=lab, **evaluate(p, ch)))

# memory model: S draws from the Manchester adult pool, per child
pool = adult_pool(); pn, pt = pool.noun.values, pool.the.values
rng = np.random.default_rng(13); sims = []
for c, g in sites[sites.speaker == "child"].groupby("child_name"):
    for k in range(1000):
        idx = rng.integers(0, len(pn), len(g)); e, pr, b, n = counts_overlap(pn[idx], pt[idx]); sims.append((k, c, e, pr))
sims = pd.DataFrame(sims, columns=["sim", "child_name", "emp", "pred"])
for lab, ch in [("Manchester 12", M12), ("Manchester 12 + Yang 6", A18), ("Manchester 11 + Yang 6", [c for c in A18 if c != "Carl"])]:
    s = sims[sims.child_name.isin(ch)]
    pv = s.groupby("sim").apply(lambda x: ttest_rel(x.emp, x.pred)[1], include_groups=False)
    mm = s.groupby("child_name")[["emp", "pred"]].mean(); t, p = ttest_rel(mm.emp, mm.pred)
    rows.append(dict(model="Memory model (Manchester adult pool)", sample=lab, n=len(mm), emp=mm.emp.mean(), pred=mm.pred.mean(),
                     dxn_p=float(pv.median()), dxn_bf01=jzs_bf01(t, len(mm)), dxn_pass_rate=float((pv > 0.05).mean())))
R = pd.DataFrame(rows); R.to_csv(OUT / "baselines_18_children.csv", index=False)
pd.set_option("display.width", 250); print(R.round(4).to_string())
# check: human per-dyad TPR from cases vs committed references
ref = pd.read_csv("results/tpr/tpr_human_perdyad_references.csv"); ref = ref[ref.speaker == "child"].set_index("child_name_norm").tpr_unrestricted
print(cases[cases.speaker == "child"].groupby("child_name").human_switch.mean().round(3).to_dict()); print(ref.round(3).to_dict())
