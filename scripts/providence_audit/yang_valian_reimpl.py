"""Re-implementation (not a copy) of the extraction and scoring in Yang & Valian (2020)'s
'Analysis 1 & 2/overlap.py' (github.com/charles-yang-upenn/Determiner-Productivity), read line by
line. Input: CHAT files with a filename prefix on each line ('<file> %mor: ...'), as in that repo's
data/*.cha and in tpr-data.
  readindata: after a line containing the speaker key, the next %mor line is taken (the flag is
  only reset by a %mor line). Pairs, all on raw %mor tokens:
   (1) det:art|a/an/the + n|X (X without 'PL'), unless the token after is also a singular n|
   (2) det:art + any token + n|Y (Y without 'PL') -> Y   (adjectives, compounds' second noun)
   (3) last two tokens det:art + n|X (X without 'PL')
  Noun = the token after 'n|' verbatim (suffixes like -DIM kept). Only n| (not n:prop, n:let...).
  Bias = per-noun dominant share (find_bias); expected overlap = Yang's Zipf formula with that bias."""
import sys, glob, os, pandas as pd
sys.path.insert(0, 'src')
from determiner.overlap import average_expected_overlap as E

ART = ('det:art|a', 'det:art|an', 'det:art|the')
def sg_noun(w): return w.startswith('n|') and 'PL' not in w

def readindata(path, key):
    data, seen = [], 0
    for line in open(path, encoding='utf-8', errors='replace'):
        if key in line: seen = 1
        if '%mor:' in line and seen == 1:
            w = line.split()
            for i in range(len(w) - 2):
                if w[i] in ART and sg_noun(w[i + 1]) and not sg_noun(w[i + 2]):
                    data.append(('a' if w[i] != 'det:art|the' else 'the', w[i + 1][2:]))
            for j in range(len(w) - 2):
                if w[j] in ART and sg_noun(w[j + 2]):
                    data.append(('a' if w[j] != 'det:art|the' else 'the', w[j + 2][2:]))
            i = len(w) - 2
            if i >= 0 and w[i] in ART and sg_noun(w[i + 1]):
                data.append(('a' if w[i] != 'det:art|the' else 'the', w[i + 1][2:]))
            seen = 0
    return data

def score(data):
    S = len(data); per = {}
    for d, n in data: per.setdefault(n, {'a': 0, 'the': 0})[d] += 1
    N = len(per); both = sum(1 for c in per.values() if c['a'] and c['the'])
    bias = sum(max(c['a'], c['the']) for c in per.values()) / S   # ties: a>=the -> a counted big, same total
    return dict(N=N, S=S, bias=bias, emp=both / N, pred=E(N, S, b=bias), pred_23=E(N, S, b=2 / 3))

if __name__ == '__main__':
    rows = []
    for label, files in [('YV2020 Manchester', sorted(glob.glob(sys.argv[1] + '/*.cha'))), ('tpr6', sorted(glob.glob(sys.argv[2] + '/*.cha')))]:
        for f in files:
            for spk, key in [('mother', '*MOT:'), ('child', '*CHI:')]:
                rows.append(dict(source=label, child=os.path.basename(f)[:-4], speaker=spk, **score(readindata(f, key))))
    r = pd.DataFrame(rows); r.to_csv(sys.argv[3], index=False); print(r.round(3).to_string())
