#!/usr/bin/env python3
"""Step 1 (CAC inference, unchanged scoring) on the children added after the paper.

    --set yang   Yang's six (data/CHILDES/Yang/<Child>/*.cha), trimmed to Yang's age ranges
    --set tpr6   Lara + 5 Providence children (data/CHILDES/tpr6/<name>.cha, tpr-data format)

Corpora are loaded and sites taken from the shared preprocessing pipeline (determiner.preprocess;
inventories from scripts/preprocessing_audit/extract_new_children.py, committed under
results/baselines/all_children/).

    --speakers child          child sites only (what the model analyses use; halves the cost)
    --speakers child mother   both (for the 5-gram caretaker analyses)

Writes <output>/det_noun_locations.csv (the evaluated sites, with ``orig_index`` = row in the source
inventory) and <output>/<type>/<model>/<child>/<speaker>_predictions.csv, whose sentence_id is the row
in that det_noun_locations.csv. Use a fresh --output per model list: the cached discourse inputs depend
on the model list's context windows.

    python scripts/yang_children/run_inference_new_children.py --set yang --models configs/models.json \
        --speakers child --output output/new_children/yang_lms
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np, pandas as pd
from determiner.paths import chdir_to_root
from determiner.preprocess import load_corpus
from determiner.inference import prepare_discourse_inputs, _run_model

LO = dict(Naomi=13, Eve=18, Sarah=27, Adam=27, Peter=16, Nina=23); HI = dict(Naomi=61, Eve=27, Sarah=61, Adam=58, Peter=34, Nina=47)
SIX = ["lara", "naima", "lily", "alex", "william", "ethan"]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["yang", "tpr6"], required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--speakers", nargs="+", default=["child"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    chdir_to_root()
    inv = pd.read_csv(f"results/baselines/all_children/{a.set}_det_noun_locations.csv.gz")
    if a.set == "yang":
        corpus = {}
        for prof, kids in [("brown", ["Adam", "Eve", "Sarah"]), ("bloom", ["Peter"]), ("sachs", ["Naomi"]), ("suppes", ["Nina"])]:
            corpus.update(load_corpus(prof, "data/CHILDES/Yang", kids, keep=("child", "mother")))
        inv = inv[[LO[c] <= x < HI[c] + 1 for c, x in zip(inv.child_name, inv.age_months)]]
    else:
        corpus = load_corpus("tpr_data", "data/CHILDES/tpr6", SIX, keep=("child", "mother"))
    inv = inv[inv.speaker_type.isin(a.speakers)].rename_axis("orig_index").reset_index()
    # the local transcripts must clean to exactly the sentences the inventory was extracted from
    text = {(c, f, l): t for c, df in corpus.items() for f, l, t in zip(df.filename, df.line_num, df.sentence)}
    bad = sum(text.get((c, f, l)) != t for c, f, l, t in zip(inv.child_name, inv.filename, inv.line_num, inv.sentence))
    if bad:
        sys.exit(f"{bad} of {len(inv)} inventory sentences do not match the local transcripts; check data/CHILDES links")
    print(f"inventory matches the local transcripts ({len(inv)} sites)")
    os.makedirs(a.output, exist_ok=True)
    inv.to_csv(os.path.join(a.output, "det_noun_locations.csv"), index=False)
    prepared = prepare_discourse_inputs(corpus, inv, a.output, a.models)
    print(f"{len(prepared)} sites ({', '.join(a.speakers)}) for {len(corpus)} children")
    sessions = {(c, f): g.sort_values("line_num").reset_index(drop=True)
                for c, df in corpus.items() for f, g in df.groupby("filename", sort=False)}
    for cfg in json.load(open(a.models)):
        print(f"\n=== {cfg['name']} ({cfg['type']})")
        try:
            _run_model(cfg, corpus, prepared, sessions, a.output, skip_existing=not a.force)
        except Exception:
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
