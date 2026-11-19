#!/usr/bin/env python3
"""Follow-up baselines for the D×N overlap and TPR tests (not part of the paper). See docs/BASELINES.md.

    python pipeline/06_baselines.py                       # everything (about an hour on a laptop)
    python pipeline/06_baselines.py --only coins memory   # some parts
    python pipeline/06_baselines.py --draws 100           # fewer simulated transcripts (quick check)

Needs: Step 2's TPR cases (output/manchester_tpr_childes/tpr_cached_inputs.csv); for the
child-directed 5-gram, data/AOCHILDES/aochildes.txt. The Wikipedia 5-gram scores are committed;
re-scoring them needs kenlm and the model file (--kenlm-model).

Writes results/baselines/*.csv and figures/baselines/*.png.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines import coins, memory, ngram, zipf_audit                     # noqa: E402
from baselines.common import BASELINE_RESULTS, Scorer, load_inventory      # noqa: E402
from baselines.figures import overlap_figure, zipf_audit_figure            # noqa: E402
from determiner.paths import FIGURES_DIR                                   # noqa: E402

PARTS = ["coins", "memory", "ngram", "zipf", "figures"]
OUT, FIG = BASELINE_RESULTS, FIGURES_DIR / "baselines"
NGRAM_SCORES = {"5-gram, child-directed speech (AO-CHILDES)": OUT / "ngram" / "aochildes5_site_scores.csv.gz",
                "5-gram, Wikipedia (KenLM)": OUT / "ngram" / "wikipedia5_site_scores.csv.gz"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=PARTS, default=PARTS)
    ap.add_argument("--draws", type=int, default=1000, help="simulated transcripts per baseline")
    ap.add_argument("--rescore-ngrams", action="store_true", help="retrain/re-score the n-grams instead of reading committed scores")
    ap.add_argument("--kenlm-model", help="path to wiki_en_token.arpa.bin, for re-scoring the Wikipedia 5-gram")
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
        inv = load_inventory()
        if a.rescore_ngrams:
            ngram.score_sites_aochildes(inv).to_csv(NGRAM_SCORES["5-gram, child-directed speech (AO-CHILDES)"], index=False)
            if a.kenlm_model:
                ngram.score_sites_kenlm(inv, a.kenlm_model).to_csv(NGRAM_SCORES["5-gram, Wikipedia (KenLM)"], index=False)
        out = []
        for name, path in NGRAM_SCORES.items():
            p = ngram.p_the_from_scores(pd.read_csv(path), scorer.sites.sentence_id)
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
        for name, path in NGRAM_SCORES.items():
            models[name] = ngram.p_the_from_scores(pd.read_csv(path), scorer.sites.sentence_id)
        zipf_audit.overlap_by_frequency(scorer.sites, models).to_csv(OUT / "overlap_by_noun_frequency.csv")
        print("zipf audit done", flush=True)

    if "figures" in a.only:
        points, results = [], []
        mem = pd.read_csv(OUT / "memory_per_dyad.csv")
        m = mem[mem.speaker == "child"]
        points.append(pd.DataFrame(dict(panel="Memory model", label="D×N sampling from caretaker input",
                                        predicted=m.draws_predicted, empirical=m.draws_empirical)))
        for key, lab in [("fair (P(the)=0.5)", "Fair coin"), ("0.82 toward 'the'", "0.82 coin"),
                         ("noun-level bias (corpus P(the|noun))", "Noun-level-bias coin")]:
            res, ov, _ = scorer.evaluate(coins.COINS[key](scorer.sites))
            ov = ov[ov.speaker == "child"]
            points.append(pd.DataFrame(dict(panel="Coin-toss model", label=lab, predicted=ov.predicted, empirical=ov.empirical)))
            results.append(dict(label=lab, dxn_p=res["child sites"]["dxn_p"]))
        for name in NGRAM_SCORES:
            res, ov, _ = scorer.evaluate(ngram.p_the_from_scores(pd.read_csv(NGRAM_SCORES[name]), scorer.sites.sentence_id))
            ov = ov[ov.speaker == "child"]
            lab = name.split(" (")[0]
            points.append(pd.DataFrame(dict(panel="N-gram model", label=lab, predicted=ov.predicted, empirical=ov.empirical)))
            results.append(dict(label=lab, dxn_p=res["child sites"]["dxn_p"]))
        points, results = pd.concat(points), pd.DataFrame(results)
        overlap_figure(points, results, FIG / "overlap_baselines.png", ngram=False)
        overlap_figure(points, results, FIG / "overlap_baselines_with_ngram.png", ngram=True)
        cp = pd.read_csv(OUT / "zipf_noun_coins_points.csv")
        cp = cp[(cp.coin == "0.82 toward the") & (cp.speaker == "child")]
        zipf_audit_figure(cp, pd.read_csv(OUT / "formula_self_check_points.csv"), FIG / "zipf_audit.png")
        print("figures done", flush=True)


if __name__ == "__main__":
    main()
