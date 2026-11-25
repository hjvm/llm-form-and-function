"""Yang & Valian (2020) Analysis 2, Table 2 ('phase one': from age 2;0, child Det-N pairs until 50
unique pairs, plus the rest of that session), applied to Providence. Extraction = re-implemented
overlap.py readindata; variants: as written vs speaker-attribution fixed; lines joined.
Also Manchester from the repo's files (start-at-2;0 rule applied explicitly) for comparison."""
import sys, glob, os, re; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/providence_audit')
import numpy as np, pandas as pd
from scipy import stats
from determiner.analytical_metrics import jzs_bf01
import yang_valian_reimpl as Y

def read(path, key, fixed):
    data = []; seen = 0
    for line in open(path, encoding='utf-8', errors='replace'):
        t = line.split()
        if fixed and len(t) > 1 and t[1].startswith('*'): seen = 1 if key in line else 0
        elif not fixed and key in line: seen = 1
        if '%mor:' in line and seen == 1:
            w = t; fn = w[0]
            for i in range(len(w) - 2):
                if w[i] in Y.ART and Y.sg_noun(w[i+1]) and not Y.sg_noun(w[i+2]): data.append(('a' if w[i] != 'det:art|the' else 'the', w[i+1][2:], fn))
            for j in range(len(w) - 2):
                if w[j] in Y.ART and Y.sg_noun(w[j+2]): data.append(('a' if w[j] != 'det:art|the' else 'the', w[j+2][2:], fn))
            i = len(w) - 2
            if i >= 0 and w[i] in Y.ART and Y.sg_noun(w[i+1]): data.append(('a' if w[i] != 'det:art|the' else 'the', w[i+1][2:], fn))
            seen = 0
    return data

def age_of(fn):  # '010102' / 'anne/011007a.cha' -> months
    m = re.search(r'(\d{2})(\d{2})(\d{2})[a-z]?(\.cha)?$', fn); return int(m.group(1)) * 12 + int(m.group(2))
def sess(fn): return re.sub(r'[a-z]?(\.cha)?$', '', fn)

def phase1(data, start_age=24, cut=50):
    data = [x for x in data if age_of(x[2]) >= start_age]
    u = set()
    for i, (d, n, fn) in enumerate(data):
        u.add((d, n))
        if len(u) == cut:
            s = sess(fn); return data[:i+1] + [x for x in data[i+1:] if sess(x[2]) == s]
    return data

rows = []
for src, d in [('Manchester (repo files)', '../yang_repo/src/data'), ('Providence+Lara (tpr-data, lines joined)', '../tpr6_joined')]:
    for f in sorted(glob.glob(d + '/*.cha')):
        for fixed in [False, True]:
            for start in [0, 24]:
                p = phase1(read(f, '*CHI:', fixed), start_age=start)
                s = Y.score([(a, b) for a, b, _ in p]); s.pop('pred_23')
                rows.append(dict(source=src, child=os.path.basename(f)[:-4], speaker_fix=fixed, start_age=start,
                                 first=p[0][2], last=p[-1][2], **s))
r = pd.DataFrame(rows)
r.to_csv('results/baselines/all_children/providence_audit/yv2020_analysis2_phase1.csv', index=False)
print(r[(r.start_age == 24)].round(3).to_string())
P = ['alex', 'ethan', 'lily', 'naima', 'william']
for (src, fx, st), g in r.groupby(['source', 'speaker_fix', 'start_age']):
    for lab, h in ([('all', g)] if src.startswith('Man') else [('Providence 5', g[g.child.isin(P)]), ('+Lara', g)]):
        t, pv = stats.ttest_rel(h.emp, h.pred)
        print(f'{src[:10]:10s} {lab:12s} speaker_fix={fx!s:5s} start={st:2d} n={len(h):2d} S {h.S.mean():5.0f} emp {h.emp.mean():.3f} pred {h.pred.mean():.3f} p {pv:.3f} BF01 {jzs_bf01(t, len(h)):.2f} over {(h.emp > h.pred).sum()}/{len(h)}')
