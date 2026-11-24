"""Literal CHILDES-%mor det-noun extraction (no cleaning, no spaCy): an article tag
(det:art|a/an/the; UD-style det|a-...-Art in Manchester's 2024 %mor) immediately followed by a
common-noun tag (n| ; noun|). Noun = %mor stem. Variants: (a) any number, (b) singular only,
(c) one adjective allowed in between, singular only. Scored with per-noun bias and fixed 2/3."""
import sys, glob, pickle, re, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from determiner.overlap import overlap_stats, average_expected_overlap as E
from determiner.analytical_metrics import jzs_bf01
from scipy import stats

ART = {'a', 'an', 'the'}
def parse(tok):
    tok = tok.split('~')[0]
    if '|' not in tok: return None, None, False
    tag, rest = tok.split('|', 1)
    stem = re.split(r'[-&]', rest)[0].lower()
    plural = bool(re.search(r'-PL\b|-Plur', rest))
    return tag, stem, plural
def is_art(tag, stem):
    return (tag == 'det:art' or (tag == 'det' and stem in ART)) and stem in ART
def is_noun(tag): return tag in ('n', 'noun')
def is_adj(tag): return tag in ('adj', 'adj:pred') or tag.startswith('adj')

def extract(mor, variant):
    toks = [parse(t) for t in str(mor).split()]
    out = []
    for i, (tag, stem, _) in enumerate(toks):
        if tag is None or not is_art(tag, stem): continue
        j = i + 1
        if variant == 'adj1' and j < len(toks) and toks[j][0] and is_adj(toks[j][0]): j += 1
        if j < len(toks) and toks[j][0] and is_noun(toks[j][0]):
            if variant != 'any_number' and toks[j][2]: continue
            out.append(('a' if stem in ('a', 'an') else 'the', toks[j][1]))
    return out

rows = []
for f in sorted(glob.glob('output/preprocessing_audit/*.pkl')):
    d = pickle.load(open(f, 'rb'))
    corpus, child = d.corpus.iloc[0], d.child.iloc[0]
    for spk in ['child', 'mother']:
        g = d[(d.speaker_type == spk) & d.mor.notna()]
        ours = [f'{a} {n}' for L in g.sp_new for (a, n) in L]
        so = overlap_stats(ours)
        row = dict(corpus=corpus, child=child, speaker=spk, ours_S=so['S'], ours_N=so['N'], ours_emp=so['empirical'],
                   ours_pred_noun=so['predicted'], ours_pred_23=E(so['N'], so['S'], b=2 / 3))
        for v in ['any_number', 'singular', 'adj1']:
            pairs = [f'{a} {n}' for m in g.mor for (a, n) in extract(m, v)]
            st = overlap_stats(pairs)
            row.update({f'{v}_S': st['S'], f'{v}_N': st['N'], f'{v}_bias': st['bias'], f'{v}_emp': st['empirical'],
                        f'{v}_pred_noun': st['predicted'], f'{v}_pred_23': E(st['N'], st['S'], b=2 / 3)})
        rows.append(row)
r = pd.DataFrame(rows)
r.to_csv('results/baselines/all_children/providence_audit/mor_literal_dxn.csv', index=False)
pd.set_option('display.width', 250)
cols = ['corpus', 'child', 'ours_S', 'ours_N', 'ours_emp', 'ours_pred_noun', 'singular_S', 'singular_N', 'singular_emp', 'singular_pred_noun', 'singular_pred_23', 'any_number_S', 'any_number_N', 'any_number_emp']
print(r[(r.speaker == 'child') & r.corpus.isin(['Providence', 'Manchester'])][cols].round(3).to_string())
def T(h, e, p):
    t, pv = stats.ttest_rel(h[e], h[p]); return f'{h[e].mean():.3f} vs {h[p].mean():.3f} p {pv:.3f} BF01 {jzs_bf01(t, len(h)):.2f} (+{int((h[e] > h[p]).sum())}/{len(h)})'
for corp in [['Providence'], ['Manchester'], ['Providence', 'Manchester', 'Brown', 'Bloom', 'Sachs', 'Suppes', 'Lara']]:
    for spk in ['child', 'mother']:
        h = r[r.corpus.isin(corp) & (r.speaker == spk)]
        print(f'\n{"+".join(corp) if len(corp) < 3 else "All 24"} {spk} n={len(h)}')
        print('  ours          per-noun:', T(h, 'ours_emp', 'ours_pred_noun'), '| 2/3:', T(h, 'ours_emp', 'ours_pred_23'))
        for v in ['any_number', 'singular', 'adj1']:
            print(f'  %mor {v:10s} per-noun:', T(h, f'{v}_emp', f'{v}_pred_noun'), '| 2/3:', T(h, f'{v}_emp', f'{v}_pred_23'))
