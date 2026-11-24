"""D×N under three bias definitions: per-noun dominant share (paper, GM&Y 2017 text), overall
determiner share max(P(the), P(a)) per speaker, and Yang's fixed 2/3 (implied by every expected
value in Yang 2011/2013 Table 1)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from determiner.overlap import overlap_stats, average_expected_overlap as E
from determiner.analytical_metrics import jzs_bf01
from scipy import stats
inv = pd.read_csv('results/baselines/all_children/tpr6_det_noun_locations.csv.gz')
man = pd.read_csv('results/site_inventory/manchester_det_noun_locations.csv')
yang = pd.read_csv('results/baselines/all_children/yang_det_noun_locations.csv.gz')
rows = []
for corpus, df in [('Providence', inv[inv.child_name != 'lara']), ('Lara', inv[inv.child_name == 'lara']), ('Manchester', man), ('Yang6', yang)]:
    for (ch, spk), s in df[df.speaker_type.isin(['child', 'mother'])].groupby(['child_name', 'speaker_type']):
        st = overlap_stats((s.det + ' ' + s.noun).tolist())
        pthe = (s.det == 'the').mean(); bo = max(pthe, 1 - pthe)
        rows.append(dict(corpus=corpus, child=ch, speaker=spk, S=st['S'], N=st['N'], emp=st['empirical'], p_the=pthe,
                         b_noun=st['bias'], pred_noun=st['predicted'], b_overall=bo, pred_overall=E(st['N'], st['S'], b=bo),
                         pred_two_thirds=E(st['N'], st['S'], b=2 / 3)))
r = pd.DataFrame(rows)
r.to_csv('results/baselines/all_children/providence_audit/bias_definition.csv', index=False)
print(r.round(3).to_string())
def T(g, col):
    t, p = stats.ttest_rel(g.emp, g[col]); return f'{g[col].mean():.3f} p {p:.3f} BF01 {jzs_bf01(t, len(g)):.2f}'
for lab, g in [('Providence 5', r[r.corpus == 'Providence']), ('Manchester 12', r[r.corpus == 'Manchester']), ('Yang 6 (full ages)', r[r.corpus == 'Yang6']), ('All 24', r)]:
    for sp in ['child', 'mother']:
        h = g[g.speaker == sp]
        print(f'{lab:18s} {sp:6s} n={len(h):2d} emp {h.emp.mean():.3f} | per-noun bias: {T(h,"pred_noun")} | overall bias: {T(h,"pred_overall")} | b=2/3: {T(h,"pred_two_thirds")}')
