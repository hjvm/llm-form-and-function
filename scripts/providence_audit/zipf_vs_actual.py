"""Split the D×N gap into (a) noun-frequency shape vs Zipf and (b) determiner use given frequency.
pred_actual = mean over noun types of 1 - b^n - (1-b)^n with each noun's observed count n
(Yang's formula with the Zipf frequencies replaced by the observed ones)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from determiner.overlap import overlap_stats, average_expected_overlap
from scipy import stats
inv = pd.read_csv('results/baselines/all_children/tpr6_det_noun_locations.csv.gz')
man = pd.read_csv('results/site_inventory/manchester_det_noun_locations.csv')
rows = []
for corpus, df in [('Providence', inv[inv.child_name != 'lara']), ('Lara', inv[inv.child_name == 'lara']), ('Manchester', man)]:
    for (ch, spk), s in df[df.speaker_type.isin(['child', 'mother'])].groupby(['child_name', 'speaker_type']):
        st = overlap_stats((s.det + ' ' + s.noun).tolist())
        n = s.noun.value_counts().values; b = st['bias']
        pa = np.mean(1 - b**n - (1 - b)**n)
        # Zipf top-10 share vs observed
        N = len(n); H = np.sum(1 / np.arange(1, N + 1))
        rows.append(dict(corpus=corpus, child=ch, speaker=spk, N=N, S=st['S'], emp=st['empirical'], pred_zipf=st['predicted'], pred_actual=pa,
                         hapax_obs=np.mean(n == 1), top10_obs=n[:10].sum() / n.sum(), top10_zipf=np.sum(1 / np.arange(1, 11)) / H))
r = pd.DataFrame(rows)
r.to_csv('results/baselines/all_children/providence_audit/zipf_vs_actual.csv', index=False)
print(r.round(3).to_string())
for (c, sp), g in r.groupby(['corpus', 'speaker']):
    print(c, sp, len(g), 'emp %.3f zipf %.3f actual %.3f | p(emp=zipf) %.3f p(emp=actual) %.3f | top10 obs %.2f zipf %.2f' % (
        g.emp.mean(), g.pred_zipf.mean(), g.pred_actual.mean(), stats.ttest_rel(g.emp, g.pred_zipf).pvalue if len(g) > 1 else np.nan,
        stats.ttest_rel(g.emp, g.pred_actual).pvalue if len(g) > 1 else np.nan, g.top10_obs.mean(), g.top10_zipf.mean()))
