"""CHAT preprocessing for any CHILDES corpus: one shared pipeline plus a small per-corpus step.

The paper's loader (``determiner.corpus``) was written for the Manchester files as distributed
with the 2024 CHILDES release and is frozen so the published site inventories reproduce. On
other corpora it has gaps that Manchester happens not to exercise (replacements such as Brown's
``de [: the]``, omitted sounds such as ``(th)e`` and ``an(d)``, wrapped main-tier lines in the
tpr-data files, speaker roles beyond CHI/MOT). This module is the general version.

Shared steps (every corpus)
    1. read      join tab-continuation lines; keep each utterance's %mor tier (for auditing);
                 record each speaker's role from the @ID headers.
    2. roles     Target_Child -> child, Mother -> mother, adult roles -> other_adult,
                 everything else (siblings, playmates, media, environment...) -> other.
    3. clean     ``clean_utterance``: CHAT conventions resolved to the intended words, the way
                 CLAN's MOR reads them (replacement targets, omitted sounds filled in, retraced
                 material dropped, codes and markers removed).

Per-corpus step (``PROFILES``)
    file layout (standard .cha per session, or tpr-data's one file per dyad with a session id
    prefix), speaker-code overrides, and any extra regex rules a corpus needs before or after
    the shared cleaning.
"""
import glob
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------------------------
ADULT_ROLES = {
    'Father', 'Investigator', 'Grandmother', 'Grandfather', 'Adult', 'Relative', 'Aunt', 'Uncle',
    'Teacher', 'Caretaker', 'Babysitter', 'Visitor', 'Student', 'Observer', 'Housekeeper',
    'Therapist', 'Nurse', 'Family_Friend',
}  # not adults or not known to be: Sibling/Brother/Sister/Cousin/Child/Playmate/Friend/Participant/
   # Unidentified, and non-speakers Media/Environment


def role_to_speaker_type(role: Optional[str], code: str) -> str:
    if role == 'Target_Child':
        return 'child'
    if role == 'Mother':
        return 'mother'
    if role in ADULT_ROLES:
        return 'other_adult'
    if role is None:  # no header: fall back to the code
        return {'CHI': 'child', 'MOT': 'mother'}.get(code, 'other')
    return 'other'


# ---------------------------------------------------------------------------------------------
# Shared cleaning
# ---------------------------------------------------------------------------------------------
_BULLET = re.compile(r'\x15[^\x15]*\x15')
# Omitted words. Manchester writes them &=0is or [* &=0is], the other corpora 0is; the paper's loader
# inserted the Manchester ones (it helps the parser on telegraphic speech), so every notation is now
# inserted the same way -- except a/an/the, which would create a site the speaker never produced,
# and bare suffixes (&=0es, 0s).
_NO_INSERT = {'a', 'an', 'the', 's', 'es', 'ed', 'ing', 'en', 'er'}


def _omitted(m):
    w = m.group(1)
    return ' ' if w.lower() in _NO_INSERT else f' {w} '


_OMITTED = [
    re.compile(r'\[\*[^\]]*?&?=?0(\w+)[^\]]*\]'),
    re.compile(r'&=0(\w+)'),
    re.compile(r"(?<!\S)0([A-Za-z]\w*)"),
]
_SUB = [
    # postcodes and scoped annotations that carry no words: [+ PI], [=! laughs], [% ...], [!], [?]
    (re.compile(r'\[(?:\+|=!?|%|\^|!+|\?|\*)[^\]]*\]'), ''),
    # replacements: <a b> [: x y] / word [: x y]  -> x y  (also [:: x] and [:=_ x])
    (re.compile(r'<[^>]*>\s*\[::? ([^\]]+)\]'), r' \1 '),
    (re.compile(r"[^\s<>\[\]]+\s*\[::? ([^\]]+)\]"), r' \1 '),
    # retracing / repetition: drop the retraced material  <a b> [/] , word [//] , [///] , [/-] , [/?]
    (re.compile(r'<[^>]*>\s*\[/{1,3}[-?]?\]'), ' '),
    (re.compile(r"[^\s<>\[\]]+\s*\[/{1,3}[-?]?\]"), ' '),
    # explanations, alternatives, paralinguistics etc. that remain: drop the bracket, keep the words
    (re.compile(r'\[[^\]]*\]'), ' '),
    (re.compile(r'[<>]'), ' '),
]
_WORD_LEVEL = [
    (re.compile(r'\b(?:xxx|yyy|www)\b'), ' '),         # unintelligible / phonological / untranscribed
    (re.compile(r'(?<!\S)0\S*'), ' '),                  # omitted words (0is) and action-only "0"
    (re.compile(r'(?<!\S)&\S*'), ' '),                  # &=laughs &-um &+fr &~nonword &word
    (re.compile(r'@[\w:$*]+'), ''),                     # dog@c, ba@wp, mira@s:spa
    (re.compile(r"\(([\w']+)\)"), r'\1'),                # omitted sounds: goin(g) -> going, (th)e -> the, doll('s)
    (re.compile(r'(\w)\+(\w)'), r'\1 \2'),              # word+word compounds -> two words (as before)
    (re.compile(r'(\w):+'), r'\1'),                     # prolongation  no:o -> noo, hm: -> hm
    (re.compile(r'[‹›]'), ' '),                          # Providence/tpr-data group brackets ‹the ,›
    (re.compile(r'[\^≠↫↑↓ˈˌ‡„≈∆∇°▔▁☺♋⁎∬Ϋ∲§∾↻⇗↗→↘⇘∞≋↓↑‖]'), ''),
]
_TERMINATORS = re.compile(r'\+(?:\.\.\.|\.\.\?|!\?|/\.|/\?|//\.|//\?|"/\.|"\.|\.|\^|,|\+|<|"|/)')
_PAUSES = re.compile(r'\((?:\.+|\d+(?::\d+)?\.?\d*)\)')


def clean_utterance(text: str) -> str:
    """Resolve CHAT conventions in a main-tier utterance to the intended plain words."""
    if not isinstance(text, str):
        return ''
    t = _BULLET.sub(' ', text)
    for pat in _OMITTED:
        t = pat.sub(_omitted, t)
    t = _PAUSES.sub(' ', t)
    t = _TERMINATORS.sub(' ', t)
    for pat, rep in _SUB:
        t = pat.sub(rep, t)
    for pat, rep in _WORD_LEVEL:
        t = pat.sub(rep, t)
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'\s+([.,!?])', r'\1', t)
    t = re.sub(r'^[.,!?\s]+', '', t)
    return t.strip()


# ---------------------------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------------------------
def _parse_id(line: str) -> Optional[Tuple[str, str]]:
    parts = line.split('\t', 1)[-1].split('|')
    if len(parts) >= 8:
        return parts[2].strip(), parts[7].strip()
    return None


def _read_tiers(lines, prefix_session: bool, default_session: str):
    """Yield (session, line_num, tier_line) with continuation lines joined."""
    cur = None
    for i, raw in enumerate(lines):
        raw = raw.rstrip('\n\r')
        if prefix_session:
            sess, _, body = raw.partition(' ')
            body = body.lstrip(' ')
        else:
            sess, body = default_session, raw
        if body.startswith('\t') and cur is not None:
            cur[2] += ' ' + body.strip()
            continue
        if cur is not None:
            yield tuple(cur)
        cur = [sess, i, body] if body[:1] in ('*', '%', '@') else None
    if cur is not None:
        yield tuple(cur)


def read_chat(path: str, prefix_session: bool = False, code_overrides: Dict[str, str] = None) -> pd.DataFrame:
    """One row per main-tier utterance: session, line_num, code, role, speaker_type, raw, mor."""
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    rows, roles = [], {}
    default_session = path
    n_main = 0  # line_num = ordinal of the main tier in the file, so dependent tiers (%mor, %gra...)
                # being present or stripped never changes it
    for sess, _, tier in _read_tiers(lines, prefix_session, default_session):
        if tier.startswith('@ID:'):
            p = _parse_id(tier)
            if p:
                roles[(sess, p[0])] = p[1]
        elif tier.startswith('*'):
            code, _, txt = tier[1:].partition(':')
            role = roles.get((sess, code))
            st = (code_overrides or {}).get(code) or role_to_speaker_type(role, code)
            ln = n_main; n_main += 1
            rows.append(dict(filename=sess, line_num=ln, code=code, role=role, speaker='*' + code + ':',
                             speaker_type=st, raw=txt.strip(), mor=None))
        elif tier.startswith('%mor:') and rows and rows[-1]['mor'] is None and rows[-1]['filename'] == sess:
            rows[-1]['mor'] = tier.split(':', 1)[1].strip()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Per-corpus profiles
# ---------------------------------------------------------------------------------------------
@dataclass
class CorpusProfile:
    name: str
    layout: str                                  # 'per_child_dirs' (Child/*.cha) or 'tpr_data' (child.cha)
    code_overrides: Dict[str, str] = field(default_factory=dict)
    pre_rules: List[Tuple[str, str]] = field(default_factory=list)    # applied to the raw tier
    post_rules: List[Tuple[str, str]] = field(default_factory=list)   # applied after clean_utterance
    notes: str = ''


PROFILES = {
    'manchester': CorpusProfile('manchester', 'per_child_dirs',
        notes='Paper corpus. Its committed inventory (legacy loader) stays the reference.'),
    'brown': CorpusProfile('brown', 'per_child_dirs',
        notes='Adam: child forms transcribed with replacements (de [: the], 1,700x); many omitted '
              'sounds (an(d), (th)e, (a)nother). Shared rules resolve both.'),
    'sachs': CorpusProfile('sachs', 'per_child_dirs', notes='Babbling @b forms; shared rules.'),
    'suppes': CorpusProfile('suppes', 'per_child_dirs', notes='Shared rules.'),
    'bloom': CorpusProfile('bloom', 'per_child_dirs',
        notes='Peter is recorded mostly with investigators (PAT, LOI); his mother speaks in few sessions.'),
    'tpr_data': CorpusProfile('tpr_data', 'tpr_data',
        # media-time bullets were flattened to plain "123_456" tokens in these files
        pre_rules=[(r'(?<!\S)\d+_\d+(?!\S)', ' ')],
        notes='Lara + Providence (and Manchester/Brown copies): one file per dyad, session id prefix, '
              'wrapped main-tier lines, flattened media bullets.'),
}


def load_corpus(profile: str, path: str, children: Optional[List[str]] = None,
                keep=('child', 'mother', 'other_adult')) -> Dict[str, pd.DataFrame]:
    """Load a corpus with the shared pipeline and its profile.

    Returns {child: DataFrame} with the loader columns the rest of the code expects
    (filename, speaker, sentence, line_num, speaker_type) plus code, role, raw and mor.
    """
    prof = PROFILES[profile]
    pre = [(re.compile(a), b) for a, b in prof.pre_rules]
    post = [(re.compile(a), b) for a, b in prof.post_rules]
    if prof.layout == 'per_child_dirs':
        units = {c: sorted(glob.glob(os.path.join(path, c, '*.cha')))
                 for c in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, c))}
        prefix = False
    else:
        units = {os.path.basename(p)[:-4]: [p] for p in sorted(glob.glob(os.path.join(path, '*.cha')))}
        prefix = True
    out = {}
    for child, files in units.items():
        if children and child not in children:
            continue
        df = pd.concat([read_chat(f, prefix, prof.code_overrides) for f in files], ignore_index=True)
        df = df[df.speaker_type.isin(keep)].copy()

        def _clean(t):
            for p, r in pre:
                t = p.sub(r, t)
            t = clean_utterance(t)
            for p, r in post:
                t = p.sub(r, t)
            return t
        df['sentence'] = df['raw'].map(_clean)
        df = df[df.sentence.str.strip().str.strip('.,!?').str.len() > 0]
        out[child] = df[['filename', 'speaker', 'sentence', 'line_num', 'speaker_type',
                         'code', 'role', 'raw', 'mor']].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------------------------
# %mor reference sites (for auditing the spaCy extraction)
# ---------------------------------------------------------------------------------------------
_DET = re.compile(r'^det(?::art)?\|(the|a|an)(?:-(?:Def|Ind)-Art)?$')


def _noun(tok: str) -> Optional[str]:
    """Singular common-noun lemma from a %mor token (old n|dog or new noun|dog style), else None."""
    tok = tok.split('~')[0]
    m = re.match(r'^(?:n|noun)\|(.+)$', tok)
    if not m:
        return None
    lemma = m.group(1)
    if re.search(r'-PL\b|&PL|-Plur|&dn', lemma):
        return None
    if lemma.startswith('+'):  # compound n|+n|fire+n|truck -> firetruck
        lemma = ''.join(re.split(r'[-&]', part.split('|')[-1])[0] for part in lemma[1:].split('+'))
    lemma = re.split(r'[-&]', lemma)[0]
    return lemma.lower().replace('_', '') if lemma else None


def mor_sites(mor: Optional[str]) -> List[Tuple[str, str]]:
    """(det, noun) sites in a %mor tier: a/an/the then a singular noun, at most one word between."""
    if not isinstance(mor, str):
        return []
    toks = [t for t in mor.split() if '|' in t]
    sites = []
    for i, t in enumerate(toks):
        m = _DET.match(t.split('~')[0])
        if not m:
            continue
        det = 'the' if m.group(1) == 'the' else 'a'
        for j in (i + 1, i + 2):
            if j >= len(toks):
                break
            n = _noun(toks[j])
            if n:
                if j == i + 1 and j + 1 < len(toks) and _noun(toks[j + 1]):   # the fire truck -> truck
                    n = _noun(toks[j + 1])
                sites.append((det, n))
                break
            if j == i + 1 and not re.match(r'^(adj|n|noun|qn|part|num|adv|det:num)\|', toks[j]):
                break
    return sites
