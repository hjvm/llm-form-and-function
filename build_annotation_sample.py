"""Build a draft human-annotation sample for the discourse-functional (TPR) benchmark.

Selects child-target D×N sites where a fixed 10-utterance human window is guaranteed to contain the
cross-speaker antecedent (`true_dist <= 10`), renders each item from the corpus stream (10 preceding
utterances + the target with its determiner masked), and exports a draft for annotation.

Sampling is **joint over the three passing models** (ltg-bert-bnc, roberta-base, t5-base). Every
eligible context is scored by all three; per-model confidence is that model's own |P(the) - 0.5| * 2
(0 = split between the/a, 1 = certain). Contexts are ranked by mean confidence across the three (ties
broken by the minimum, so consensus-confident sites lead) and we take the top N as the **high** band and
the bottom N as the **low** band. Each row is a unique context and reports all three models' probabilities
and confidences explicitly (`<model>_pthe`, `<model>_conf`) — no averaging is baked into the reported
values; the mean/min are only the ranking keys. `n_pred_the` (0..3) flags directional agreement.

Inputs (all produced upstream by the pipeline):
  - output/manchester_tpr_childes/tpr_cached_inputs.csv   (TPR site metadata; human_baseline.ipynb)
  - output/manchester_discourse_childes/*/*/child_predictions.csv  (Experiment 2; run_experiments.py)

Output: results/human_validation/annotation_sample_draft.csv

This is a *draft* sampler. Open decisions (sample size, child-only vs +caretaker targets, annotator
count/recruitment, platform, optional 3/10/full context manipulation) are intentionally left to the
annotation-run design; scale N_PER_BAND to the target N.
"""
import os, glob, re
import numpy as np, pandas as pd
import cac_utils as cac

DISCOURSE_DIR = "output/manchester_discourse_childes"
TPR_META      = "output/manchester_tpr_childes/tpr_cached_inputs.csv"
OUT_DIR       = "results/human_validation"
KEY           = ["child_name", "speaker", "sentence_id"]
WINDOW        = 10
MODELS        = ["ltg-bert-bnc", "roberta-base", "t5-base"]   # the passing models
N_PER_BAND    = 36                                            # unique contexts in each of the high/low bands
SPK = {"child": "*CHI", "mother": "*MOT", "other_adult": "*OTH"}

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # --- site metadata: transition cell, gold determiner, true antecedent distance (from cache) ---
    meta = pd.read_csv(TPR_META, usecols=[
        "child_name", "filename", "speaker", "sentence_id", "noun", "previous_det", "original_det",
        "target_line_num", "det_char_start", "det_char_end", "original_sentence", "true_dist"])
    meta["d_A"] = meta.previous_det.str.lower().map({"a": 0, "an": 0, "the": 1})
    meta = meta.dropna(subset=["d_A", "true_dist"])
    meta["d_A"] = meta.d_A.astype(int)
    meta["gold_the"] = (meta.original_det.str.lower() == "the").astype(int)
    _C = {(0, 0): "a->a", (0, 1): "a->the", (1, 1): "the->the", (1, 0): "the->a"}
    meta["transition"] = [_C[(a, g)] for a, g in zip(meta.d_A, meta.gold_the)]

    # --- corpus stream per session, for rendering the fixed 10-utterance window ---
    corpus = cac.load_manchester_corpus()
    sessions = {}
    for child, df in corpus.items():
        for fn, g in df.groupby("filename"):
            sessions[(child, os.path.basename(str(fn)))] = g.sort_values("line_num").reset_index(drop=True)

    # --- model predictions over the eligible child sites ---
    long = _load_predictions(meta)
    sites_meta = meta[(meta.speaker == "child") & (meta.true_dist <= WINDOW)].set_index(KEY)
    uni = long[(long.speaker == "child") & (long.true_dist <= WINDOW) & long.model.isin(MODELS)]

    # one row per context, one P(the) column per model (single-arch models: one value/site)
    piv = uni.pivot_table(index=KEY, columns="model", values="P_the_B", aggfunc="mean").dropna(subset=MODELS)
    pthe_cols, conf_cols = [], []
    for m in MODELS:
        piv[f"{m}_pthe"] = piv[m]; pthe_cols.append(f"{m}_pthe")
        piv[f"{m}_conf"] = (piv[m] - 0.5).abs() * 2; conf_cols.append(f"{m}_conf")
    piv["mean_conf"]  = piv[conf_cols].mean(axis=1)
    piv["min_conf"]   = piv[conf_cols].min(axis=1)
    piv["n_pred_the"] = sum((piv[m] > 0.5).astype(int) for m in MODELS)  # 0/3 = consensus direction
    df = piv.drop(columns=MODELS).join(sites_meta, how="inner").reset_index()
    print(f"unique child contexts scored by all 3 models (true_dist<={WINDOW}): {len(df)}")

    # verify the antecedent (topic noun's prior mention) actually appears in the rendered window
    df["prompt_10utt"] = [_render(r, sessions) for r in df.itertuples(index=False)]
    df["antecedent_in_prompt"] = [_ant_present(n, p) for n, p in zip(df.noun, df.prompt_10utt)]
    print("antecedent present in rendered window: %.0f%% of %d contexts"
          % (100 * df.antecedent_in_prompt.mean(), len(df)))
    df = df[df.antecedent_in_prompt].copy()

    # rank by joint confidence (highest from all three first); top -> high band, bottom -> low band
    ranked = df.sort_values(["mean_conf", "min_conf", "child_name", "sentence_id"],
                            ascending=[False, False, True, True])
    sample = pd.concat([ranked.head(N_PER_BAND).assign(conf_band="high"),
                        ranked.tail(N_PER_BAND).assign(conf_band="low")], ignore_index=True)

    cols = KEY + ["conf_band", "noun", "transition", "previous_det", "original_det", "true_dist",
                  "n_pred_the", "mean_conf", "min_conf"] + \
           [c for m in MODELS for c in (f"{m}_pthe", f"{m}_conf")] + ["prompt_10utt"]
    out_path = f"{OUT_DIR}/annotation_sample_draft.csv"
    sample[cols].to_csv(out_path, index=False)
    print(f"saved {out_path}  ({len(sample)} unique contexts: {N_PER_BAND} high + {N_PER_BAND} low)")
    print("mean_conf by band:", sample.groupby("conf_band").mean_conf.agg(["min", "max"]).round(3).to_dict("index"))
    print("transition cells:", dict(sample.transition.value_counts()), "| dyads:", sample.child_name.nunique())


def _load_predictions(meta):
    parts = []
    for md_dir in sorted(glob.glob(f"{DISCOURSE_DIR}/*/*")):
        if not os.path.isdir(md_dir):
            continue
        fr = []
        for dd in glob.glob(md_dir + "/*"):
            if not os.path.isdir(dd):
                continue
            for spk in ["child", "mother"]:
                pf = f"{dd}/{spk}_predictions.csv"
                if os.path.exists(pf):
                    d = pd.read_csv(pf, usecols=["sentence_id", "P_the", "P_a", "context_length"])
                    d["child_name"] = os.path.basename(dd)
                    d["speaker"] = spk
                    fr.append(d)
        if not fr:
            continue
        p = pd.concat(fr, ignore_index=True)
        p["model"] = os.path.basename(md_dir)
        parts.append(p)
    pred = pd.concat(parts, ignore_index=True)
    pred["P_the_B"] = pred.P_the / (pred.P_the + pred.P_a).replace(0, np.nan)
    long = pred.merge(meta[KEY + ["true_dist"]], on=KEY, how="inner").dropna(subset=["P_the_B"])
    long["in_window"] = long.true_dist <= long.context_length
    return long


def _render(r, sessions, k=WINDOW):
    g = sessions.get((r.child_name, os.path.basename(str(r.filename))))
    if g is None:
        return None
    prior = g[g.line_num < r.target_line_num].tail(k)
    lines = [f"{SPK.get(sp, '*OTH')}: {t}" for sp, t in zip(prior.speaker_type, prior.sentence)]
    masked = r.original_sentence[:int(r.det_char_start)] + "____" + r.original_sentence[int(r.det_char_end):]
    tspk = g.loc[g.line_num == r.target_line_num, "speaker_type"]
    lines.append(f"{SPK.get(tspk.iloc[0] if len(tspk) else 'child', '*CHI')}: {masked}")
    return "\n".join(lines)


def _ant_present(noun, prompt):
    if not isinstance(prompt, str):
        return False
    ctx = "\n".join(prompt.split("\n")[:-1])
    return bool(re.search(r"\b" + re.escape(str(noun)) + r"\b", ctx, re.I))


if __name__ == "__main__":
    main()
