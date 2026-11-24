"""Audit of Providence child det-noun sites (shared pipeline): ages, imitation, letters,
plurals/mass, hapaxes, and D×N under different selections. Manchester children for contrast."""
import re, sys, collections
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from determiner.preprocess import load_corpus
from determiner.overlap import overlap_stats

PROV = ['naima', 'lily', 'alex', 'william', 'ethan']
inv = pd.read_csv('results/baselines/all_children/tpr6_det_noun_locations.csv.gz', dtype={'filename': str})
man = pd.read_csv('results/site_inventory/manchester_det_noun_locations.csv')
corp = load_corpus('tpr_data', 'data/CHILDES/tpr6', PROV + ['lara'])

def age_months(sid):
    s = re.sub(r'\D', '', str(sid))[:6]
    return int(s[:2]) * 12 + int(s[2:4]) + int(s[4:6]) / 30.0

def stats(df):
    st = overlap_stats((df.det + ' ' + df.noun).tolist())
    return st

# --- imitation: child site whose det+noun appeared in an adult utterance within the previous k utterances
def flag_imitation(child, sites, k=2):
    c = corp[child].copy(); c['filename'] = c.filename.astype(str)
    key = {}
    for fn, g in c.groupby('filename'):
        g = g.sort_values('line_num')
        sents = list(zip(g.line_num, g.speaker_type, g.sentence.fillna('').str.lower()))
        idx = {ln: i for i, (ln, _, _) in enumerate(sents)}
        key[fn] = (sents, idx)
    flags, verb = [], []
    for _, r in sites.iterrows():
        sents, idx = key[r.filename]
        i = idx.get(r.line_num)
        pat = re.compile(r'\b(a|an|the)\s+(\w+\s+)?' + re.escape(r.noun) + r'\b')
        prev = [s for (ln, sp, s) in sents[max(0, i - k):i] if sp != 'child'] if i is not None else []
        flags.append(any(pat.search(s) for s in prev))
        verb.append(any(r.sentence.lower().strip(' .?!') in s for s in prev) if prev else False)
    return np.array(flags), np.array(verb)

rows, samp = [], []
LETTER = set('abcdefghijklmnopqrstuvwxyz')
for ch in PROV + ['lara']:
    s = inv[(inv.child_name == ch) & (inv.speaker_type == 'child')].copy()
    s['age'] = s.filename.map(age_months)
    s['imit'], s['verbatim'] = flag_imitation(ch, s)
    s['letter'] = s.noun.isin(LETTER)
    s['plural_form'] = s.apply(lambda r: bool(re.search(r'\b' + re.escape(r.noun) + r'(s|es)\b', r.sentence.lower())) and not re.search(r'\b' + re.escape(r.noun) + r'\b', r.sentence.lower()), axis=1)
    cnt = s.noun.value_counts()
    base = stats(s)
    def row(label, d):
        st = stats(d); return dict(child=ch, selection=label, n_sites=len(d), N=st['N'], S=st['S'], bias=round(st['bias'], 3), emp=round(st['empirical'], 3), pred=round(st['predicted'], 3), gap=round(st['empirical'] - st['predicted'], 3))
    rows.append(row('all', s))
    rows.append(row('no imitation (adult det+noun in prev 2 utts)', s[~s.imit]))
    rows.append(row('no letters', s[~s.letter]))
    rows.append(row('age < 36 mo', s[s.age < 36]))
    rows.append(row('age 24-36 mo', s[(s.age >= 24) & (s.age < 36)]))
    rows.append(row('age < 30 mo', s[s.age < 30]))
    s['ageband'] = (s.age // 6 * 6).astype(int)
    for b, g in s.groupby('ageband'):
        if len(g) >= 100:
            st = stats(g); samp.append(dict(child=ch, corpus='Providence' if ch != 'lara' else 'Lara', ageband=b, S=st['S'], N=st['N'], emp=st['empirical'], pred=st['predicted']))
    print(f'\n== {ch}: ages {s.age.min():.0f}-{s.age.max():.0f} mo, sites {len(s)}, imitation {s.imit.mean():.1%} (verbatim {s.verbatim.mean():.1%}), letters {s.letter.sum()}, hapax nouns {(cnt==1).sum()}/{len(cnt)}')
    print('  top nouns:', ', '.join(f'{n} {c}' for n, c in cnt.head(15).items()))
    print('  top frames:', collections.Counter(s.sentence.str.lower().str.strip(' .?!')).most_common(8))
    s.to_csv(f'output/providence_audit_{ch}.csv.gz', index=False)

# Manchester children for contrast, same age bands
for ch, s in man[man.speaker_type == 'child'].groupby('child_name'):
    m = s.filename.astype(str).str.extract(r'(\d{6})')[0]
    s = s.assign(age=m.map(lambda x: age_months(x) if isinstance(x, str) else np.nan))
    cnt = s.noun.value_counts()
    st = stats(s)
    rows.append(dict(child=ch, selection='Manchester all', n_sites=len(s), N=st['N'], S=st['S'], bias=round(st['bias'], 3), emp=round(st['empirical'], 3), pred=round(st['predicted'], 3), gap=round(st['empirical'] - st['predicted'], 3)))
    if s.age.notna().any():
        s['ageband'] = (s.age // 6 * 6)
        for b, g in s.groupby('ageband'):
            if len(g) >= 100:
                st = stats(g); samp.append(dict(child=ch, corpus='Manchester', ageband=int(b), S=st['S'], N=st['N'], emp=st['empirical'], pred=st['predicted']))
pd.DataFrame(rows).to_csv('results/baselines/all_children/providence_audit/selections.csv', index=False)
pd.DataFrame(samp).to_csv('results/baselines/all_children/providence_audit/age_band_samples.csv', index=False)
print(pd.DataFrame(rows).to_string())
