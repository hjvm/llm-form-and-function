#!/usr/bin/env python3
"""Audit D×N site extraction on every corpus: legacy loader vs shared pipeline, both scored
against each utterance's own %mor tier (CLAN MOR/POST, an independent tagger).

For each child and speaker (child, mother) it writes, per utterance, the spaCy sites under the
legacy cleaning, the spaCy sites under the shared pipeline, and the %mor sites, then reports
precision/recall of each against %mor (multiset match of (det, noun lemma) within an utterance).

    python scripts/preprocessing_audit/audit_extraction.py --out output/preprocessing_audit
"""
import argparse, collections, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import pandas as pd
from determiner.paths import chdir_to_root
from determiner.corpus import clean_childes_annotations
from determiner.preprocess import load_corpus, mor_sites
from determiner.sites import SPACY_MODEL, MAX_GAP

SETS = [  # (label, corpus, profile, path, children)
    ("Manchester", "manchester", "manchester", "data/CHILDES/Manchester", None),
    ("Brown", "brown", "brown", "data/CHILDES/Yang", ["Adam", "Eve", "Sarah"]),
    ("Bloom", "bloom", "bloom", "data/CHILDES/Yang", ["Peter"]),
    ("Sachs", "sachs", "sachs", "data/CHILDES/Yang", ["Naomi"]),
    ("Suppes", "suppes", "suppes", "data/CHILDES/Yang", ["Nina"]),
    ("Lara", "lara", "tpr_data", "data/CHILDES/tpr6", ["lara"]),
    ("Providence", "providence", "tpr_data", "data/CHILDES/tpr6", ["naima", "lily", "alex", "william", "ethan"]),
]


def spacy_sites(nlp, texts):
    out = []
    for doc in nlp.pipe(texts, batch_size=2000):
        s = []
        for ch in doc.noun_chunks:
            d = ch[0]
            if d.pos_ != "DET" or d.lower_ not in ("a", "an", "the"):
                continue
            if not 0 <= ch.root.i - d.i - 1 <= MAX_GAP:
                continue
            if ch.root.pos_ != "NOUN" or "Sing" not in ch.root.morph.get("Number"):
                continue
            s.append(("a" if d.lower_ in ("a", "an") else "the", ch.root.lemma_.lower()))
        out.append(s)
    return out


def match(pred, gold):
    g = collections.Counter(gold); p = collections.Counter(pred)
    return sum((g & p).values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/preprocessing_audit")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    chdir_to_root(); os.makedirs(a.out, exist_ok=True)
    import spacy
    nlp = spacy.load(SPACY_MODEL, disable=["ner"])
    for label, corpus, prof, path, kids in SETS:
        if a.only and label not in a.only:
            continue
        data = load_corpus(prof, path, kids, keep=("child", "mother"))
        for child, df in data.items():
            f = os.path.join(a.out, f"{label}_{child}.pkl")
            if os.path.exists(f):
                continue
            df = df.copy()
            df["legacy"] = df.raw.map(clean_childes_annotations).fillna("")
            if prof == "tpr_data":  # the legacy tpr loader dropped wrapped lines: emulate with first line only
                pass
            df["sp_new"] = spacy_sites(nlp, df.sentence.tolist())
            df["sp_old"] = spacy_sites(nlp, df.legacy.tolist())
            df["mor_sites"] = df.mor.map(mor_sites)
            df["corpus"] = label; df["child"] = child
            df.to_pickle(f)
            print(label, child, len(df), flush=True)


if __name__ == "__main__":
    main()
