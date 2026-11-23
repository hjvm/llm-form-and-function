#!/usr/bin/env python3
"""Manchester with the shared pipeline, all speakers (child, mother, adult others), for the adult TPR
reference only. The paper's child/mother analyses keep the committed inventory.
    python scripts/preprocessing_audit/extract_manchester_all_speakers.py Anne Aran ...   (chunks; then --merge)"""
import sys, os, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import pandas as pd
from determiner.paths import chdir_to_root
from determiner.preprocess import load_corpus
from determiner.sites import extract_det_noun_locations
chdir_to_root()
if sys.argv[1] == "--merge":
    inv = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("output/new_pipeline/manchester_*/det_noun_locations.csv"))], ignore_index=True)
    inv.to_csv("results/baselines/all_children/manchester_all_speakers_det_noun_locations.csv.gz", index=False)
    print(inv.groupby(["child_name", "speaker_type"]).size().unstack())
else:
    kids = sys.argv[1:]
    extract_det_noun_locations(load_corpus("manchester", "data/CHILDES/Manchester", kids), f"output/new_pipeline/manchester_{kids[0]}")
