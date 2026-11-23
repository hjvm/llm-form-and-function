#!/usr/bin/env python3
"""Site inventories for the children added after the paper, from the shared preprocessing pipeline.

    yang : Adam, Eve, Sarah (Brown), Peter (Bloom), Naomi (Sachs), Nina (Suppes)  -> + age_months
    tpr6 : Lara and the five Providence children (tpr-data files)

Writes output/new_pipeline/<set>/det_noun_locations.csv with child, mother and other-adult sites
(other_adult = adult roles from the @ID headers only). Manchester keeps its committed inventory.
"""
import os, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import pandas as pd
from determiner.paths import chdir_to_root
from determiner.preprocess import load_corpus
from determiner.sites import extract_det_noun_locations

SETS = {
    "yang": [("brown", "data/CHILDES/Yang", ["Adam", "Eve", "Sarah"]), ("bloom", "data/CHILDES/Yang", ["Peter"]),
             ("sachs", "data/CHILDES/Yang", ["Naomi"]), ("suppes", "data/CHILDES/Yang", ["Nina"])],
    "tpr6": [("tpr_data", "data/CHILDES/tpr6", ["lara", "naima", "lily", "alex", "william", "ethan"])],
}

def main():
    chdir_to_root()
    for name in sys.argv[1:] or SETS:
        corpus = {}
        for prof, path, kids in SETS[name]:
            corpus.update(load_corpus(prof, path, kids))
        out = f"output/new_pipeline/{name}"
        inv = extract_det_noun_locations({k: v for k, v in corpus.items()}, out)
        if name == "yang":
            ages = {}
            for f in inv.filename.unique():
                m = re.search(r"\|CHI\|(\d+);(\d+)(?:\.(\d+))?\|", open(f, encoding="utf-8", errors="ignore").read())
                ages[f] = int(m.group(1)) * 12 + int(m.group(2)) + int(m.group(3) or 0) / 31 if m else None
            inv["age_months"] = inv.filename.map(ages)
            inv.to_csv(os.path.join(out, "det_noun_locations.csv"), index=False)
        print(inv.groupby(["child_name", "speaker_type"]).size().unstack())

if __name__ == "__main__":
    main()
