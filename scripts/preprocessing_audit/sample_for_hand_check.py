#!/usr/bin/env python3
"""Random samples of shared-pipeline sites per corpus (and of disagreements with %mor) for hand checking."""
import collections, glob, random, sys
import pandas as pd
out = sys.argv[1] if len(sys.argv) > 1 else "output/preprocessing_audit"
k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
random.seed(2026)
rows = []
for f in sorted(glob.glob(f"{out}/*.pkl")):
    df = pd.read_pickle(f)
    for _, r in df.iterrows():
        g = collections.Counter(r.mor_sites)
        for s in r.sp_new:
            rows.append(dict(corpus=r.corpus, child=r.child, speaker=r.speaker_type, raw=r.raw, clean=r.sentence,
                             det=s[0], noun=s[1], in_mor=g[s] > 0, kind="site"))
        for s in (g - collections.Counter(r.sp_new)):
            rows.append(dict(corpus=r.corpus, child=r.child, speaker=r.speaker_type, raw=r.raw, clean=r.sentence,
                             det=s[0], noun=s[1], in_mor=True, kind="mor_only"))
R = pd.DataFrame(rows)
S = pd.concat([g.sample(min(k, len(g)), random_state=7) for _, g in R.groupby(["corpus", "kind"])])
S.to_csv(f"{out}/hand_check_sample.csv", index=False)
print(len(S))
