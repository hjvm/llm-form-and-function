import sys, os, re; sys.path.insert(0, "src")
import pandas as pd
from determiner.corpus import load_manchester_corpus
from determiner.sites import extract_det_noun_locations
which = sys.argv[1]
corpus = load_manchester_corpus(f"./data/CHILDES/{which}")
if which == "Manchester":
    corpus = {k: corpus[k] for k in ["Carl", "Anne", "Dominic"]}
inv = extract_det_noun_locations(corpus, f"output/yang_children/{which}")
# child age per file
ages = {}
for f in inv.filename.unique():
    m = re.search(r"\|CHI\|(\d+);(\d+)(?:\.(\d+))?\|", open(f, encoding="utf-8", errors="ignore").read())
    ages[f] = int(m.group(1)) * 12 + int(m.group(2)) + (int(m.group(3) or 0) / 31) if m else None
inv["age_months"] = inv.filename.map(ages)
inv.to_csv(f"output/yang_children/{which}/det_noun_locations.csv", index=False)
print(inv.groupby(["child_name", "speaker_type"]).size().unstack())
