#!/usr/bin/env python3
"""Precision/recall of the legacy and shared-pipeline extractions against %mor, per corpus and speaker."""
import collections, glob, sys
import pandas as pd
out = sys.argv[1] if len(sys.argv) > 1 else "output/preprocessing_audit"
import re
def norm(sites):  # compare noun lemmas without spaces/underscores; %mor's pronominal "one" is not a noun site
    return [(d, re.sub(r"[^a-z]", "", n)) for d, n in sites if n != "one"]
def m(p, g):
    """Matches; nouns agree if equal or one ends the other (sailboat/boat: compounds are split differently)."""
    g = list(norm(g)); hit = 0
    for d, n in norm(p):
        for k, (d2, n2) in enumerate(g):
            if d == d2 and n and n2 and (n.endswith(n2) or n2.endswith(n)):
                hit += 1; del g[k]; break
    return hit
rows = []
for f in sorted(glob.glob(f"{out}/*.pkl")):
    df = pd.read_pickle(f); df = df[df.mor.notna()]
    for st, g in df.groupby("speaker_type"):
        r = dict(corpus=g.corpus.iat[0], child=g.child.iat[0], speaker=st, utts=len(g),
                 mor=g.mor_sites.map(lambda x: len(norm(x))).sum(), old=g.sp_old.map(len).sum(), new=g.sp_new.map(len).sum())
        r["old_hit"] = sum(m(p, q) for p, q in zip(g.sp_old, g.mor_sites))
        r["new_hit"] = sum(m(p, q) for p, q in zip(g.sp_new, g.mor_sites))
        # determiner-only agreement (ignores noun-lemma differences)
        r["new_hit_det"] = sum(sum((collections.Counter(d for d, _ in norm(p)) & collections.Counter(d for d, _ in norm(q))).values())
                             for p, q in zip(g.sp_new, g.mor_sites))
        rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(f"{out}/audit_by_child.csv", index=False)
agg = R.groupby(["corpus", "speaker"])[["utts", "mor", "old", "new", "old_hit", "new_hit", "new_hit_det"]].sum()
agg["old_P"] = agg.old_hit / agg.old; agg["old_R"] = agg.old_hit / agg.mor
agg["new_P"] = agg.new_hit / agg.new; agg["new_R"] = agg.new_hit / agg.mor
agg["new_change_%"] = 100 * (agg.new - agg.old) / agg.old
agg.to_csv(f"{out}/audit_by_corpus.csv")
pd.set_option("display.width", 200)
print(agg.round(3).to_string())
