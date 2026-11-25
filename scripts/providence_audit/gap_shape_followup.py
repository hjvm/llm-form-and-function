"""Follow-up: is the flatter noun distribution (shape component) a sampling-design effect?
(1) correlation of shape with top-10 share, sessions, age span across all children;
(2) Providence children re-sampled like Manchester: 34 sessions drawn from 2;0–3;0 (20 draws)."""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/providence_audit'); sys.path.insert(0, 'src')
import gap_decomposition as G
from scipy import stats
G.R = 100
r = pd.read_csv('results/baselines/all_children/providence_audit/gap_decomposition.csv')
c = r[r.speaker == 'child']
for v in ['top10', 'hapax', 'sessions', 'S', 'N']:
    print(v, 'vs shape r=%.2f p=%.3f' % stats.pearsonr(c[v], c['shape']), '| vs gap r=%.2f p=%.3f' % stats.pearsonr(c[v], c['gap']))
inv = pd.read_csv('results/baselines/all_children/tpr6_det_noun_locations.csv.gz', dtype={'filename': str})
age = lambda s: int(re.sub(r'\D', '', s)[:2]) * 12 + int(re.sub(r'\D', '', s)[2:4])
rows = []
rng = np.random.default_rng(7)
for ch in ['naima', 'lily', 'alex', 'william', 'ethan']:
    g = inv[(inv.child_name == ch) & (inv.speaker_type == 'child')].copy(); g['age'] = g.filename.map(age)
    g = g[(g.age >= 24) & (g.age < 36)]
    ses = g.filename.unique()
    for d in range(20):
        pick = rng.choice(ses, size=min(34, len(ses)), replace=False)
        res = G.decompose(g[g.filename.isin(pick)]); rows.append(dict(child=ch, draw=d, n_sessions=len(pick), **res))
    print(ch, len(ses), 'sessions in 2;0-3;0', flush=True)
s = pd.DataFrame(rows); s.to_csv('results/baselines/all_children/providence_audit/gap_manchester_like_samples.csv', index=False)
m = s.groupby('child')[['S', 'N', 'b', 'emp', 'pred', 'gap', 'artifact', 'shape', 'det_use', 'top10']].mean()
print(m.round(3)); print('mean', m.mean().round(3).to_dict())
# share of draws where the 5 children pass a paired t
pv = [stats.ttest_rel(s[s.draw == d].emp, s[s.draw == d].pred).pvalue for d in range(20)]
print('paired t pass rate over draws: %.2f, median p %.3f' % (np.mean(np.array(pv) > .05), np.median(pv)))
