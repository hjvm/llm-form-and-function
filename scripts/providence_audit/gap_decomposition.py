"""Why do children over-overlap? Per-speaker decomposition of the D×N gap (emp − pred, paper method)
   gap = artifact + shape + det_use
 artifact: E[emp − pred] for a perfectly productive Zipf speaker with the same N (observed types),
           S and expected b̂ (Yang's own model, scored exactly like the child): the test's own offset.
 shape:    E[emp | child's real noun counts, binomial dets] − E[emp | Zipf counts]: what the real
           noun-frequency distribution adds relative to Zipf, with determiners still chosen freely.
 det_use:  emp − E[emp | child's real noun counts, binomial dets]: how the child actually spreads
           a/the over each noun vs a free binomial choice (negative = nouns stick to one determiner).
 True b in both nulls is calibrated so that E[b̂] equals the speaker's observed b̂."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from determiner.overlap import overlap_stats, average_expected_overlap as E
rng = np.random.default_rng(1)
R = 300

def pred_fast(N, S, b):
    r = np.arange(1, N + 1); p = 1 / (r * np.sum(1 / r))
    return np.mean(1 - (p * b + 1 - p) ** S - (p * (1 - b) + 1 - p) ** S + (1 - p) ** S)

def stats_counts(n, the):  # n: counts per noun type (>0), the: 'the' counts
    S = n.sum(); a = n - the
    b = np.maximum(the, a).sum() / S
    emp = np.mean((the > 0) & (a > 0))
    return len(n), S, b, emp

def exp_bhat(n, b):  # analytic E[b̂] for fixed counts n under Bin(n, b) per noun (favored det random)
    tot = 0.0
    from scipy.stats import binom
    for k in np.unique(n):
        x = np.arange(k + 1); pm = binom.pmf(x, k, b)
        tot += (n == k).sum() * np.sum(pm * np.maximum(x, k - x))
    return tot / n.sum()

def solve_b(n, target):
    lo, hi = 0.5, 0.9999
    for _ in range(40):
        m = (lo + hi) / 2
        (lo, hi) = (m, hi) if exp_bhat(n, m) < target else (lo, m)
    return (lo + hi) / 2

def zipf_draw(Nt, S):
    r = np.arange(1, Nt + 1); p = 1 / r; p /= p.sum()
    c = rng.multinomial(S, p); return c[c > 0]

def solve_Nt(Nobs, S):  # Zipf type count whose expected observed types = Nobs
    def eN(Nt):
        r = np.arange(1, Nt + 1); p = 1 / r; p /= p.sum(); return np.sum(1 - (1 - p) ** S)
    lo, hi = Nobs, Nobs * 50
    while eN(hi) < Nobs: hi *= 2
    for _ in range(40):
        m = (lo + hi) // 2
        if eN(m) < Nobs: lo = m + 1
        else: hi = m
        if lo >= hi: break
    return lo

def dets(n, b):
    fav = rng.random(len(n)) < 0.5
    k = rng.binomial(n, b)
    return np.where(fav, k, n - k)

def decompose(df):
    n = df.noun.value_counts().values; the = df[df.det == 'the'].noun.value_counts().reindex(df.noun.value_counts().index, fill_value=0).values
    N, S, b, emp = stats_counts(n, the); pred = pred_fast(N, S, b)
    # null B: real counts
    bB = solve_b(n, b)
    embB = np.array([stats_counts(n, dets(n, bB))[3] for _ in range(R)])
    # null A: Zipf counts, same expected N and b̂
    Nt = solve_Nt(N, S)
    A = []
    nz0 = zipf_draw(Nt, S); bA = solve_b(nz0, b)
    for _ in range(R):
        nz = zipf_draw(Nt, S); NN, SS, bb, ee = stats_counts(nz, dets(nz, bA))
        A.append((NN, bb, ee, pred_fast(NN, SS, bb)))
    A = np.array(A)
    top10 = n[:10].sum() / S; H = np.sum(1 / np.arange(1, N + 1)); top10z = np.sum(1 / np.arange(1, 11)) / H
    return dict(N=N, S=S, b=b, emp=emp, pred=pred, gap=emp - pred,
                artifact=np.mean(A[:, 2] - A[:, 3]), artifact_sd=np.std(A[:, 2] - A[:, 3]),
                shape=embB.mean() - A[:, 2].mean(), det_use=emp - embB.mean(),
                nullA_N=A[:, 0].mean(), nullA_b=A[:, 1].mean(), b_true=bA,
                hapax=np.mean(n == 1), top10=top10, top10_zipf=top10z, sessions=df.filename.nunique())

inv = pd.read_csv('results/baselines/all_children/tpr6_det_noun_locations.csv.gz', dtype={'filename': str})
man = pd.read_csv('results/site_inventory/manchester_det_noun_locations.csv')
yang = pd.read_csv('results/baselines/all_children/yang_det_noun_locations.csv.gz')
rows = []
for corpus, D in [('Providence', inv[inv.child_name != 'lara']), ('Lara', inv[inv.child_name == 'lara']), ('Manchester', man), ('Yang6', yang)]:
    for (ch, spk), g in D[D.speaker_type.isin(['child', 'mother'])].groupby(['child_name', 'speaker_type']):
        rows.append(dict(corpus=corpus, child=ch, speaker=spk, **decompose(g))); print(rows[-1]['child'], spk, round(rows[-1]['gap'], 3), flush=True)
r = pd.DataFrame(rows)
r.to_csv('results/baselines/all_children/providence_audit/gap_decomposition.csv', index=False)
