"""Every analysis on all 24 children: Manchester 12 (committed paper inventory) + Yang's 6 (Brown, Bloom,
Sachs, Suppes; trimmed to his age ranges) + Lara and the 5 Providence children. The 12 new children come
from the shared preprocessing pipeline (determiner.preprocess; inventories in results/baselines/all_children/).
Child and caretaker sites; D×N and TPR; coins, memory, 5-grams (when their predictions on the new
children exist). Other-adult TPR reference: the paper's (tpr-data 17 dyads, as published) and an
adult-roles-only version (drops TV/media, siblings, child visitors) with Eve, Sarah and Naomi's adults added.
Run from the repo root."""
import sys; sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
from determiner.analytical_metrics import jzs_bf01
from determiner.tpr import prepare_tpr_cases
from baselines.common import Scorer, counts_overlap, load_sites, load_tpr_cases
from baselines import ngram
from baselines.memory import adult_pool

OUT = Path("results/baselines/all_children"); INV = OUT
LO = dict(Naomi=13, Eve=18, Sarah=27, Adam=27, Peter=16, Nina=23); HI = dict(Naomi=61, Eve=27, Sarah=61, Adam=58, Peter=34, Nina=47)
SIX = ["lara", "naima", "lily", "alex", "william", "ethan"]
OFF_Y, OFF_T = 10 ** 7, 2 * 10 ** 7
ADULT_ROLES = {"Father", "Investigator", "Grandmother", "Grandfather", "Adult", "Relative", "Aunt", "Uncle", "Visitor",
               "Family_Friend", "Teacher", "Caretaker", "Babysitter", "Student"}

def add_switch(c):
    c = c.copy(); c["d_A"] = c.previous_det.str.lower().map({"a": 0, "an": 0, "the": 1}).astype(int)
    c["human_switch"] = ((c.original_det.str.lower() == "the").astype(int) != c.d_A).astype(int); return c

def new_set(name, off, keep_mask=None):
    """Sites and child/mother TPR cases for a new-pipeline inventory. Cases are built from child and mother
    sites only, like the paper's Manchester inventory (no other adults in between)."""
    inv = pd.read_csv(INV / f"{name}_det_noun_locations.csv.gz")
    keep = np.ones(len(inv), bool) if keep_mask is None else keep_mask(inv)
    cm = inv[inv.speaker_type.isin(["child", "mother"])]
    c = add_switch(prepare_tpr_cases(cm)); c = c[c.sentence_id.isin(np.where(keep)[0])]; c["sentence_id"] += off
    s = inv[keep & inv.noun.notna().values & inv.speaker_type.isin(["child", "mother"]).values].reset_index()
    s = pd.DataFrame(dict(sentence_id=s["index"] + off, child_name=s.child_name.str.capitalize(), speaker=s.speaker_type,
                          noun=s.noun, orig_the=(s.det == "the").astype(int)))
    c["child_name"] = c.child_name.str.capitalize()
    return inv, s, c

yinv, ys, yc = new_set("yang", OFF_Y, lambda d: np.array([LO[c] <= a < HI[c] + 1 for c, a in zip(d.child_name, d.age_months)]))
tinv, ts, tc = new_set("tpr6", OFF_T)
msites = load_sites(); mcases = load_tpr_cases()
sites = pd.concat([msites[["sentence_id", "child_name", "speaker", "noun", "orig_the"]].dropna(subset=["noun"]), ys, ts], ignore_index=True)
sites = sites[sites.speaker.isin(["child", "mother"])].reset_index(drop=True)
cases = pd.concat([mcases, yc[mcases.columns], tc[mcases.columns]], ignore_index=True)

# ---- other-adult references
full_t = pd.read_csv("results/site_inventory/tpr_data_det_noun_locations.csv")
oa = add_switch(prepare_tpr_cases(full_t)); oa = oa[oa.speaker == "other_adult"]
ADULT_PAPER = float(oa.human_switch.mean())
roles = pd.read_csv(OUT / "tpr_data_speaker_roles.csv")  # dyad, code, role (from the tpr-data @ID headers)
adult_codes = set(zip(roles.dyad, roles.code)) - set(zip(roles[~roles.role.isin(ADULT_ROLES)].dyad, roles[~roles.role.isin(ADULT_ROLES)].code))
oa_codes = full_t.loc[oa.sentence_id, "speaker"].str.strip("*:").values
oa_adult = oa[[(d, k) in adult_codes for d, k in zip(oa.child_name, oa_codes)]]
esn = yinv[yinv.child_name.isin(["Eve", "Sarah", "Naomi"])]  # other_adult here is already adult roles only
oa2 = add_switch(prepare_tpr_cases(esn)); oa2 = oa2[oa2.speaker == "other_adult"]
ADULT_ALL = float(pd.concat([oa_adult.human_switch, oa2.human_switch]).mean())
print(f"other-adult pooled TPR: paper (tpr-data 17) = {ADULT_PAPER:.4f} (n={len(oa)}); adult roles only = "
      f"{oa_adult.human_switch.mean():.4f} (n={len(oa_adult)}); + Eve/Sarah/Naomi = {ADULT_ALL:.4f} (n={len(oa_adult)+len(oa2)})")

ADULT_REFS = [("paper", ADULT_PAPER), ("paper, adult roles only", float(oa_adult.human_switch.mean())),
              ("adult roles + Eve/Sarah/Naomi", ADULT_ALL)]
_ref = OUT / "adult_tpr_reference.json"  # from adult_tpr_reference.py: every adult transition in all 24 dyads
if _ref.exists():
    import json
    ADULT_REFS += [(f"24 dyads, {k}", v) for k, v in json.load(open(_ref)).items()]
M12 = sorted(msites.child_name.unique()); Y6 = list(LO); T6 = [c.capitalize() for c in SIX]
SAMPLES = {"Manchester 12 (paper)": M12, "All 24": M12 + Y6 + T6, "All 24 without Carl": [c for c in M12 + Y6 + T6 if c != "Carl"],
           "Manchester 12 + Yang 6": M12 + Y6, "Manchester 12 + Lara/Providence 6": M12 + T6}

def evaluate(p_the, children, speaker, adult):
    d = sites.assign(P_the=np.clip(p_the, 0, 1)); d = d[(d.speaker == speaker) & d.child_name.isin(children)]
    ov = Scorer.overlap_rows(d).set_index("child_name"); t, p = ttest_rel(ov.empirical, ov.predicted)
    tp = cases[(cases.speaker == speaker) & cases.child_name.isin(children)].merge(d[["sentence_id", "P_the"]], on="sentence_id")
    per = (tp.d_A - tp.P_the).abs().groupby(tp.child_name).mean(); t1, p1 = ttest_1samp(per, adult)
    return dict(n=len(ov), emp=ov.empirical.mean(), pred=ov.predicted.mean(), dxn_p=p, dxn_bf01=jzs_bf01(t, len(ov)),
                tpr=per.mean(), tpr_p=p1, tpr_bf01=jzs_bf01(t1, len(per)), n_tpr_dyads=len(per))

pthe_noun = sites.groupby("noun").orig_the.mean()
models = {"Humans": sites.orig_the.values.astype(float), "Fair coin": np.full(len(sites), 0.5), "0.82 coin": np.full(len(sites), 0.82),
          "Noun-level coin": sites.noun.map(pthe_noun).values}
def new_children_ngram(m, setname):
    """P_the per site (orig_index = row of the new inventory) for a 5-gram on a new set. Collected from the
    runner's output dir (output/new_children/<set>_ngram) into results/baselines/all_children/ngram/ once."""
    f = OUT / "ngram" / f"{m}_{setname}.csv.gz"
    run = Path(f"output/new_children/{setname}_ngram")
    if not f.exists() and (run / "ar" / m).exists():
        q = ngram.collect_predictions(m, run / "ar")
        idx = pd.read_csv(run / "det_noun_locations.csv").orig_index
        q = q.assign(orig_index=idx.values[q.sentence_id.values])[["orig_index", "P_the", "truncated"]]
        f.parent.mkdir(exist_ok=True); q.to_csv(f, index=False)
    return pd.read_csv(f) if f.exists() else None

for m, lab in ngram.NGRAM_MODELS.items():
    parts = [pd.read_csv(f"results/baselines/ngram/{m}_cac_predictions.csv.gz")[["sentence_id", "P_the"]]]
    for setname, off in [("yang", OFF_Y), ("tpr6", OFF_T)]:
        q = new_children_ngram(m, setname)
        if q is None:
            print(f"{m}: not yet run on {setname}"); continue
        parts.append(pd.DataFrame(dict(sentence_id=q.orig_index + off, P_the=q.P_the)))
    models[lab] = sites.sentence_id.map(pd.concat(parts).drop_duplicates("sentence_id").set_index("sentence_id").P_the).values
rows = []
for name, p in models.items():
    for spk in ["child", "mother"]:
        for lab, ch in SAMPLES.items():
            sub = (sites.speaker == spk) & sites.child_name.isin(ch)
            if np.isnan(p[sub.values]).any(): continue
            for adult_lab, adult in ADULT_REFS:
                rows.append(dict(model=name, sites=spk, sample=lab, adult_ref=adult_lab, adult_mean=adult, **evaluate(p, ch, spk, adult)))
# memory model (child and caretaker sample sizes), pool = Manchester adults
pool = adult_pool(); pn, pt = pool.noun.values, pool.the.values; rng = np.random.default_rng(13); sims = []
for (c, spk), g in sites.groupby(["child_name", "speaker"]):
    for k in range(1000):
        idx = rng.integers(0, len(pn), len(g)); e, pr, b, n = counts_overlap(pn[idx], pt[idx]); sims.append((k, c, spk, e, pr))
sims = pd.DataFrame(sims, columns=["sim", "child_name", "speaker", "emp", "pred"])
for spk in ["child", "mother"]:
    for lab, ch in SAMPLES.items():
        s = sims[sims.child_name.isin(ch) & (sims.speaker == spk)]
        pv = s.groupby("sim").apply(lambda x: ttest_rel(x.emp, x.pred)[1], include_groups=False)
        mm = s.groupby("child_name")[["emp", "pred"]].mean(); t, p = ttest_rel(mm.emp, mm.pred)
        rows.append(dict(model="Memory model", sites=spk, sample=lab, n=len(mm), emp=mm.emp.mean(), pred=mm.pred.mean(),
                         dxn_p=float(pv.median()), dxn_bf01=jzs_bf01(t, len(mm))))
R = pd.DataFrame(rows); R.to_csv(OUT / "all_24_children.csv", index=False)
sites.to_csv(OUT / "sites_24.csv.gz", index=False); cases.to_csv(OUT / "tpr_cases_24.csv.gz", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(R.round(3).to_string())
