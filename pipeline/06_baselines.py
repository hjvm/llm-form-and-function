#!/usr/bin/env python3
"""Follow-up baselines for the D×N overlap and TPR tests (not part of the paper). See docs/BASELINES.md.

    python pipeline/06_baselines.py                       # everything (about an hour on a laptop)
    python pipeline/06_baselines.py --only coins memory   # some parts
    python pipeline/06_baselines.py --draws 100           # fewer simulated transcripts (quick check)

Needs: Step 2's TPR cases (output/manchester_tpr_childes/tpr_cached_inputs.csv). The 5-grams'
predictions come from the paper's own Step 1 (see src/baselines/ngram.py):
    python pipeline/01_run_inference.py --models configs/ngram_models.json
    python pipeline/06_baselines.py --collect-ngrams      # copy them into results/baselines/ngram/

Writes results/baselines/*.csv and figures/baselines/*.png.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import coins, memory, ngram, zipf_audit                     # noqa: E402
from baselines.common import BASELINE_RESULTS, Scorer                    # noqa: E402
from baselines.figures import overlap_figure, tpr_figure, zipf_audit_figure            # noqa: E402
from determiner.paths import FIGURES_DIR                                   # noqa: E402

PARTS = ["coins", "memory", "ngram", "zipf", "figures"]
OUT, FIG = BASELINE_RESULTS, FIGURES_DIR / "baselines"


def ngram_p_the(sentence_ids):
    """{label: P(the) per site} for every 5-gram whose CAC predictions are committed."""
    out = {}
    for model, label in ngram.NGRAM_MODELS.items():
        path = OUT / "ngram" / f"{model}_cac_predictions.csv.gz"
        if path.exists():
            out[label] = ngram.p_the(pd.read_csv(path), sentence_ids)
        else:
            print(f"  {model}: no CAC predictions yet ({path.name}); skipped", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=PARTS, default=PARTS)
    ap.add_argument("--draws", type=int, default=1000, help="simulated transcripts per baseline")
    ap.add_argument("--collect-ngrams", action="store_true",
                    help="copy the 5-grams' Step 1 predictions from output/ into results/baselines/ngram/")
    a = ap.parse_args()
    for d in (OUT, OUT / "ngram", FIG):
        d.mkdir(parents=True, exist_ok=True)
    scorer = Scorer()
    rng = np.random.default_rng(2611)

    if "coins" in a.only:
        coins.run(scorer, a.draws).to_csv(OUT / "coin_results.csv", index=False)
        print("coins done", flush=True)

    if "memory" in a.only:
        res, per = memory.run(a.draws)
        res.to_csv(OUT / "memory_results.csv", index=False)
        per.to_csv(OUT / "memory_per_dyad.csv", index=False)
        print("memory done", flush=True)

    if "ngram" in a.only:
        if a.collect_ngrams:
            for model in ngram.NGRAM_MODELS:
                try:
                    ngram.collect_predictions(model).drop(columns="child_name").to_csv(
                        OUT / "ngram" / f"{model}_cac_predictions.csv.gz", index=False, float_format="%.6g")
                except FileNotFoundError as e:
                    print(f"  {e}", flush=True)
        out = []
        for name, p in ngram_p_the(scorer.sites.sentence_id).items():
            acc = (p > 0.5) == scorer.sites.orig_the.values
            extra = {lab: dict(accuracy=acc[(scorer.sites.speaker == lab.split()[0]).values].mean())
                     for lab in ("child sites", "mother sites")}
            out.append(scorer.evaluate_with_draws(name, p, a.draws, rng, extra))
        pd.concat(out).to_csv(OUT / "ngram_results.csv", index=False)
        print("ngram done", flush=True)

    if "zipf" in a.only:
        dec = zipf_audit.decomposition()
        dec.to_csv(OUT / "zipf_decomposition.csv", index=False)
        rows, pts = zipf_audit.zipf_noun_coins(dec, a.draws)
        rows.to_csv(OUT / "zipf_noun_coins.csv", index=False)
        pts.to_csv(OUT / "zipf_noun_coins_points.csv", index=False)
        rows, pts = zipf_audit.formula_self_check(dec, max(a.draws // 3, 1))
        rows.to_csv(OUT / "formula_self_check.csv", index=False)
        pts.to_csv(OUT / "formula_self_check_points.csv", index=False)
        zipf_audit.session_consistency().to_csv(OUT / "session_consistency.csv", index=False)
        models = {"children (realised)": scorer.sites.orig_the.values.astype(float),
                  "0.82 coin": np.full(len(scorer.sites), 0.82),
                  "noun-level coin": scorer.sites.noun_pthe.values}
        models.update(ngram_p_the(scorer.sites.sentence_id))
        zipf_audit.overlap_by_frequency(scorer.sites, models).to_csv(OUT / "overlap_by_noun_frequency.csv")
        print("zipf audit done", flush=True)

    if "figures" in a.only:
        points, results = [], []
        mem = pd.read_csv(OUT / "memory_per_dyad.csv")
        m = mem[mem.speaker == "child"]
        points.append(pd.DataFrame(dict(panel="Memory model", label="D×N sampling from caretaker input",
                                        predicted=m.draws_predicted, empirical=m.draws_empirical)))
        mres = pd.read_csv(OUT / "memory_results.csv")
        results.append(dict(label="D×N sampling from caretaker input",
                            dxn_p=float(mres[mres.sites == "child sites"].dxn_p_median.iloc[0])))
        for key, lab in [("fair (P(the)=0.5)", "Fair coin"), ("0.82 toward 'the'", "0.82 coin"),
                         ("noun-level bias (corpus P(the|noun))", "Noun-level-bias coin")]:
            res, ov, _ = scorer.evaluate(coins.COINS[key](scorer.sites))
            ov = ov[ov.speaker == "child"]
            points.append(pd.DataFrame(dict(panel="Coin-toss model", label=lab, predicted=ov.predicted, empirical=ov.empirical)))
            results.append(dict(label=lab, dxn_p=res["child sites"]["dxn_p"]))
        for name, p in ngram_p_the(scorer.sites.sentence_id).items():
            res, ov, _ = scorer.evaluate(p)
            ov = ov[ov.speaker == "child"]
            lab = name.split(" (")[0]
            points.append(pd.DataFrame(dict(panel="N-gram model", label=lab, predicted=ov.predicted, empirical=ov.empirical)))
            results.append(dict(label=lab, dxn_p=res["child sites"]["dxn_p"]))
        points, results = pd.concat(points), pd.DataFrame(results)
        overlap_figure(points, results, FIG / "overlap_baselines.png", ngram=False)
        overlap_figure(points, results, FIG / "overlap_baselines_with_ngram.png", ngram=True)
        tpr_pts = [pd.DataFrame(dict(label="Children\n& caretakers", speaker=spk, child_name=ref.index, tpr=ref.values))
                   for spk, ref in [("child", scorer.child_ref), ("mother", scorer.mother_ref)]]
        tpr_models = {"Fair coin": coins.COINS["fair (P(the)=0.5)"](scorer.sites),
                      "0.82 coin": coins.COINS["0.82 toward 'the'"](scorer.sites),
                      "Noun-level\ncoin": coins.COINS["noun-level bias (corpus P(the|noun))"](scorer.sites)}
        tpr_models.update({k.replace("5-gram, child-directed speech (AO-CHILDES)", "5-gram,\nchild-directed")
                             .replace("5-gram, Wikipedia (KenLM)", "5-gram,\nWikipedia"): v
                           for k, v in ngram_p_the(scorer.sites.sentence_id).items()})
        for lab, p in tpr_models.items():
            _, _, tp = scorer.evaluate(p)
            per = tp.groupby(["speaker", "child_name"]).contrib.mean().reset_index(name="tpr")
            tpr_pts.append(per.assign(label=lab))
        tpr_pts = pd.concat(tpr_pts, ignore_index=True)
        tpr_pts.to_csv(OUT / "tpr_per_dyad.csv", index=False)
        tpr_figure(tpr_pts, scorer.adult_pop, FIG / "tpr_baselines.png")
        cp = pd.read_csv(OUT / "zipf_noun_coins_points.csv")
        cp = cp[(cp.coin == "0.82 toward the") & (cp.speaker == "child")]
        zipf_audit_figure(cp, pd.read_csv(OUT / "formula_self_check_points.csv"), FIG / "zipf_audit.png")
        print("figures done", flush=True)


if __name__ == "__main__":
    main()
