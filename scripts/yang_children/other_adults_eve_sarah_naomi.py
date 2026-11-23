import sys, os, re, glob; sys.path.insert(0, "src")
import pandas as pd
from determiner.corpus import clean_childes_annotations
from determiner.sites import extract_det_noun_locations
from determiner.tpr import prepare_tpr_cases
ADULT_ROLES = {"Father", "Investigator", "Grandmother", "Grandfather", "Adult", "Student", "Relative", "Aunt", "Uncle", "Teacher", "Visitor"}
LO = dict(Naomi=13, Eve=18, Sarah=27, Adam=27, Peter=16, Nina=23); HI = dict(Naomi=61, Eve=27, Sarah=61, Adam=58, Peter=34, Nina=47)
corpus = {}
for c in ["Eve", "Sarah", "Naomi"]:
    rows = []
    for f in sorted(glob.glob(f"./data/CHILDES/Yang/{c}/*.cha")):
        txt = open(f, encoding="utf-8").read()
        m = re.search(r"\|CHI\|(\d+);(\d+)", txt); age = int(m.group(1)) * 12 + int(m.group(2))
        if not (LO[c] <= age <= HI[c]): continue
        part = re.search(r"@Participants:\s*(.*)", txt).group(1)
        adults = {p.split()[0] for p in part.split(",") if len(p.split()) >= 2 and p.split()[-1] in ADULT_ROLES}
        t = pd.read_csv(f, sep="\t", on_bad_lines="skip", names=["speaker", "sentence"]).dropna()
        t["filename"] = f; t["line_num"] = range(len(t))
        code = t.speaker.str.extract(r"^\*([A-Z]+):")[0]
        t["speaker_type"] = code.map(lambda x: "child" if x == "CHI" else "mother" if x == "MOT" else "other_adult" if x in adults else None)
        rows.append(t[t.speaker_type.notna()])
    d = pd.concat(rows, ignore_index=True); d["sentence"] = d.sentence.apply(clean_childes_annotations)
    corpus[c] = d[d.sentence.str.strip() != ""][["filename", "speaker", "sentence", "line_num", "speaker_type"]]
inv = extract_det_noun_locations(corpus, "output/yang_children/other_adults")
cases = prepare_tpr_cases(inv)
print(inv.groupby(["child_name", "speaker_type"]).size().unstack())
oa = cases[cases.speaker == "other_adult"]
oa = oa.assign(sw=((oa.original_det == "the").astype(int) != oa.previous_det.map({"a": 0, "the": 1})).astype(int))
print(oa.groupby("child_name").agg(n_cases=("sw", "size"), tpr=("sw", "mean"), sessions=("filename", "nunique")))
