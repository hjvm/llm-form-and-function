"""5-gram baselines.

* Child-directed speech: interpolated Kneser-Ney 5-gram trained here on AO-CHILDES (Huebner &
  Willits 2021; 893,989 American English child-directed utterances, data/AOCHILDES/aochildes.txt).
  Manchester (British English) is not part of AO-CHILDES. One discount per order,
  D = n1 / (n1 + 2 n2) (Chen & Goodman 1998), continuation counts for lower orders, uniform
  floor over the vocabulary. Tokenisation mirrors AO-CHILDES (lower case, contractions whole,
  punctuation split off).
* Wikipedia: the public KenLM model BramVanroy/kenlm_wikipedia_en (wiki_en_token.arpa.bin), scored
  with ``score_sites_kenlm`` (needs ``pip install kenlm spacy`` and the model file).

Each site's utterance is scored whole, with sentence boundaries, with the determiner span
replaced by "a", "an" and "the". As in src/cac/scoring.py, P(a) merges "a" and "an" in log space
and P(the) = softmax over the two categories. Utterances are scored alone: a 5-gram sees four
words of context, so earlier utterances could not reach it anyway.
"""
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from determiner.paths import DATA_DIR

AOCHILDES = DATA_DIR / "AOCHILDES" / "aochildes.txt"
_TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)*|[^\sa-z0-9]")


def tokenize(s: str):
    return _TOK.findall(s.lower())


class KneserNey:
    def __init__(self, sentences, order: int = 5):
        self.order = order
        self.vocab = Counter(w for s in sentences for w in s)
        self.V = len(self.vocab) + 2                                   # + </s>, <unk>
        top = Counter()
        for s in sentences:
            w = ["<s>"] * (order - 1) + s + ["</s>"]
            for i in range(order - 1, len(w)):
                top[tuple(w[i - order + 1:i + 1])] += 1
        self.counts = {order: top}
        for k in range(order - 1, 0, -1):                             # continuation counts
            c = Counter()
            for g in self.counts[k + 1]:
                c[g[1:]] += 1
            self.counts[k] = c
        self.D, self.ctx_tot, self.ctx_typ = {}, {}, {}
        for k in range(1, order + 1):
            vals = Counter(self.counts[k].values())
            self.D[k] = vals[1] / (vals[1] + 2 * vals[2])
            tot, typ = defaultdict(int), defaultdict(int)
            for g, c in self.counts[k].items():
                tot[g[:-1]] += c
                typ[g[:-1]] += 1
            self.ctx_tot[k], self.ctx_typ[k] = dict(tot), dict(typ)

    def prob(self, w, h) -> float:
        p = 1.0 / self.V
        for k in range(1, self.order + 1):
            hh = h[len(h) - (k - 1):] if k > 1 else ()
            if len(hh) < k - 1:
                break
            tot = self.ctx_tot[k].get(hh)
            if not tot:
                break
            c = self.counts[k].get(hh + (w,), 0)
            p = max(c - self.D[k], 0) / tot + self.D[k] * self.ctx_typ[k][hh] / tot * p
        return p

    def logp10(self, tokens) -> float:
        o = self.order
        w = ["<s>"] * (o - 1) + [t if t in self.vocab else "<unk>" for t in tokens] + ["</s>"]
        return sum(math.log10(self.prob(w[i], tuple(w[i - o + 1:i]))) for i in range(o - 1, len(w)))


def _substitutions(inv: pd.DataFrame):
    for r in inv.itertuples(index=False):
        s, a, b = r.sentence, int(r.det_char_start), int(r.det_char_end)
        assert s[a:b].lower() in ("a", "an", "the"), s
        yield {d: s[:a] + d + s[b:] for d in ("a", "the", "an")}


def score_sites_aochildes(inv: pd.DataFrame, corpus: Path = AOCHILDES) -> pd.DataFrame:
    if not Path(corpus).exists():
        raise FileNotFoundError(f"{corpus} not found; see docs/BASELINES.md for where to get AO-CHILDES.")
    sents = [l.split() for l in open(corpus, encoding="utf-8") if l.strip()]
    lm = KneserNey(sents)
    rows = [(i, *(lm.logp10(tokenize(v[d])) for d in ("a", "the", "an"))) for i, v in enumerate(_substitutions(inv))]
    return pd.DataFrame(rows, columns=["row", "logp_a", "logp_the", "logp_an"])


def score_sites_kenlm(inv: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Wikipedia 5-gram, as run for the talk: spaCy en_core_web_sm tokenizer only, kenlm bos/eos."""
    import kenlm
    import spacy
    m = kenlm.Model(model_path)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"])
    cache = {}

    def sc(s):
        if s not in cache:
            cache[s] = m.score(" ".join(t.text for t in nlp.make_doc(s.strip())), bos=True, eos=True)
        return cache[s]
    rows = [(i, *(sc(v[d]) for d in ("a", "the", "an"))) for i, v in enumerate(_substitutions(inv))]
    return pd.DataFrame(rows, columns=["row", "logp_a", "logp_the", "logp_an"])


def p_the_from_scores(scores: pd.DataFrame, sentence_ids) -> np.ndarray:
    """P(the) per site; 'a' = logaddexp(a, an) as in src/cac/scoring.py. ``row`` = sentence_id."""
    ln10 = np.log(10)
    sc = scores.set_index("row")
    la = np.logaddexp(sc.logp_a * ln10, sc.logp_an * ln10)
    p = (1 / (1 + np.exp(la - sc.logp_the * ln10))).reindex(sentence_ids).values
    assert not np.isnan(p).any(), "scores do not cover every site"
    return p
