"""D×N pairs of the Manchester adults other than the mother (input to the memory model).

Reads data/CHILDES/Manchester/<child>/*.cha, keeps the adult tiers below, cleans them with
determiner.corpus.clean_childes_annotations and extracts singular a/an/the + noun pairs with
the same rule as the site inventory (spaCy noun chunk, determiner first, at most one word
between determiner and head, singular common-noun head, lemma lower-cased).

The committed results/baselines/inputs/manchester_other_adult_det_noun.csv was extracted with
spaCy 3.8 / en_core_web_sm 3.8.0, not the 3.7.1 pinned for the paper's inventory; expect small
differences if it is re-extracted under ``pip install -e ".[sites]"``.
"""
import os

import pandas as pd

from determiner.corpus import clean_childes_annotations
from determiner.paths import MANCHESTER_DIR

# Excluded: *BRO, *NIC (siblings), *FRI, *WAY (friends of unknown age).
ADULT_TIERS = ["*INV:", "*RAC:", "*JUI:", "*DAD:", "*FAT:", "*GRA:", "*ADU:"]


def load_utterances(root=MANCHESTER_DIR) -> pd.DataFrame:
    rows = []
    for child in sorted(os.listdir(root)):
        p = os.path.join(root, child)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if not f.endswith(".cha"):
                continue
            t = pd.read_csv(os.path.join(p, f), sep="\t", on_bad_lines="skip", names=["speaker", "sentence"]).dropna()
            t = t[t.speaker.isin(ADULT_TIERS)].copy()
            t["sentence"] = t.sentence.apply(clean_childes_annotations)
            t = t[t.sentence.str.strip() != ""]
            t["child_name"], t["filename"] = child, f
            rows.append(t)
    return pd.concat(rows, ignore_index=True)


def extract(utterances: pd.DataFrame) -> pd.DataFrame:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    out = []
    for doc, r in zip(nlp.pipe(utterances.sentence.astype(str).tolist(), batch_size=1000), utterances.itertuples()):
        for chunk in doc.noun_chunks:
            if chunk[0].pos_ == "DET" and chunk[0].lower_ in ("a", "an", "the"):
                gap = chunk.root.i - chunk[0].i - 1
                if 0 <= gap <= 1 and chunk.root.pos_ == "NOUN" and "Sing" in chunk.root.morph.get("Number"):
                    out.append(dict(child_name=r.child_name, speaker=r.speaker, speaker_type="other_adult",
                                    det="a" if chunk[0].lower_ in ("a", "an") else "the", noun=chunk.root.lemma_.lower()))
    return pd.DataFrame(out)
