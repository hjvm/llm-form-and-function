"""Parroting rate per child (24 children; child + mother utterances, shared pipeline, all sessions).
Turn = run of consecutive utterances by one speaker. For each child det-noun token:
  rep1: the same det+noun occurs in the mother's immediately preceding turn;
  rep3: ... in any of the mother's turns among the previous 3 turns (i.e. the last 2 mother turns);
  noun1: the same noun (any determiner) in the mother's preceding turn.
Utterance level: share of child utterances (>=2 words) contained verbatim in the mother's preceding turn.
Gap (emp − pred) and its decomposition from gap_decomposition.csv."""
import glob, pickle, re, sys, numpy as np, pandas as pd
from scipy import stats
norm = lambda s: ' ' + re.sub(r'[^a-z\' ]', ' ', str(s).lower()).replace("'", "'") + ' '
rows = []
for f in sorted(glob.glob('output/preprocessing_audit/*.pkl')):
    d = pickle.load(open(f, 'rb')); corpus, child = d.corpus.iloc[0], d.child.iloc[0]
    tok = {k: 0 for k in ['n', 'rep1', 'rep3', 'noun1']}; ut = ut_rep = 0
    for fn, g in d.groupby('filename', sort=False):
        g = g.sort_values('line_num')
        turns = []  # (speaker_type, [utterances], [pairs])
        for st, s, pr in zip(g.speaker_type, g.sentence, g.sp_new):
            if turns and turns[-1][0] == st: turns[-1][1].append(norm(s)); turns[-1][2].extend(pr)
            else: turns.append([st, [norm(s)], list(pr)])
        for i, (st, utts, prs) in enumerate(turns):
            if st != 'child': continue
            prev = turns[i - 1] if i >= 1 and turns[i - 1][0] == 'mother' else None
            prev3 = [t for t in turns[max(0, i - 3):i] if t[0] == 'mother']
            p1 = set(prev[2]) if prev else set(); p3 = set(x for t in prev3 for x in t[2]); n1 = {n for _, n in p1}
            for pair in prs:
                tok['n'] += 1; tok['rep1'] += pair in p1; tok['rep3'] += pair in p3; tok['noun1'] += pair[1] in n1
            for u in utts:
                if len(u.split()) >= 2:
                    ut += 1; ut_rep += bool(prev) and any(re.sub(r'\s+', ' ', u) in re.sub(r'\s+', ' ', m) for m in prev[1])
    rows.append(dict(corpus=corpus, child=child, sites=tok['n'], det_noun_1turn=tok['rep1'] / tok['n'], det_noun_3turns=tok['rep3'] / tok['n'],
                     noun_1turn=tok['noun1'] / tok['n'], utterance_1turn=ut_rep / ut))
r = pd.DataFrame(rows)
gd = pd.read_csv('results/baselines/all_children/providence_audit/gap_decomposition.csv')
gd = gd[gd.speaker == 'child'].assign(key=lambda x: x.child.str.lower())[['key', 'gap', 'shape', 'det_use']]
r = r.assign(key=r.child.str.lower()).merge(gd, on='key', how='left').drop(columns='key')
r.to_csv('results/baselines/all_children/providence_audit/parroting.csv', index=False)
print(r.round(3).to_string())
print(r.groupby('corpus')[['det_noun_1turn', 'det_noun_3turns', 'utterance_1turn', 'gap']].mean().round(3))
for v in ['det_noun_1turn', 'det_noun_3turns', 'utterance_1turn']:
    for y in ['gap', 'shape', 'det_use']:
        print(v, y, 'r=%.2f p=%.3f' % stats.pearsonr(r[v], r[y]))
