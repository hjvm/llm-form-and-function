"""D×N per child from %mor (CLAN MOR) sites vs spaCy sites, same utterances (those with %mor)."""
import sys, glob, pickle, re, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from determiner.overlap import overlap_stats
from scipy import stats
rows = []
for f in sorted(glob.glob('output/preprocessing_audit/*.pkl')):
    d = pickle.load(open(f, 'rb'))
    corpus, child = d.corpus.iloc[0], d.child.iloc[0]
    for spk in ['child', 'mother']:
        g = d[(d.speaker_type == spk) & d.mor.notna() & (d.mor.astype(str).str.len() > 0)]
        cov = len(g) / max(1, (d.speaker_type == spk).sum())
        out = dict(corpus=corpus, child=child, speaker=spk, mor_coverage=round(cov, 3))
        for col in ['sp_new', 'mor_sites']:
            pairs = [f'{a} {n}' for L in g[col] for (a, n) in L if n != 'one']
            st = overlap_stats(pairs)
            out.update({f'{col}_S': st['S'], f'{col}_N': st['N'], f'{col}_emp': st['empirical'], f'{col}_pred': st['predicted']})
        rows.append(out)
r = pd.DataFrame(rows)
r.to_csv('results/baselines/all_children/providence_audit/mor_vs_spacy.csv', index=False)
print(r.round(3).to_string())
for (c, sp), g in r.groupby(['corpus', 'speaker']):
    if len(g) < 2: continue
    for col in ['sp_new', 'mor_sites']:
        gap = g[f'{col}_emp'] - g[f'{col}_pred']
        print(f'{c:11s} {sp:6s} {col:9s} n={len(g)} emp {g[col+"_emp"].mean():.3f} pred {g[col+"_pred"].mean():.3f} gap {gap.mean():+.3f} p {stats.ttest_1samp(gap,0).pvalue:.3f} pos {int((gap>0).sum())}/{len(g)}')
