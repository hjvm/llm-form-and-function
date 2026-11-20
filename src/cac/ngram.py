"""N-gram language models as autoregressive CAC scorers (follow-up baselines, not in the paper).

An n-gram model is plugged into the unchanged CAC pipeline as an ``ar`` model: Step 1 builds the
same discourse prompt as for every autoregressive model (preceding session lines labelled
``*CHI:``/``*MOT:``, the target line with the determiner replaced by ``___``), splices each
candidate form into the blank, scores the whole prompt and normalises with
``merge_and_normalize`` ({'a': ['a', 'an'], 'the': ['the']}).

Scoring a prompt: the n-gram models were trained one utterance per line, so a line break is an
utterance boundary. The prompt's log-probability is the sum over its lines of each line's
sentence log-probability with sentence boundaries (natural log). Lines before the target are
identical for every candidate, so they add the same constant to every candidate's score and
cancel in the normalisation; only the target line, speaker label included, decides P(the).

Backends
    kn      interpolated Kneser-Ney trained on a one-utterance-per-line corpus (AO-CHILDES),
            tokenised like AO-CHILDES (lower case, contractions whole, punctuation split off).
    kenlm   a KenLM binary (e.g. BramVanroy/kenlm_wikipedia_en wiki_en_token.arpa.bin),
            tokenised with spaCy en_core_web_sm's tokenizer, as used for that model in the talk.

Model config (configs/ngram_models.json):
    {"name": "ngram/aochildes-kn5", "type": "ar",
     "ngram": {"backend": "kn", "corpus": "data/AOCHILDES/aochildes.txt", "order": 5}}
    {"name": "ngram/wikipedia-kenlm5", "type": "ar",
     "ngram": {"backend": "kenlm", "path": "/path/to/wiki_en_token.arpa.bin"}}
"""
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Sequence

LN10 = math.log(10)
CONTEXT_TOKENS = 1024     # same budget as the GPT-2-family models in the paper
_KN_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)*|[^\sa-z0-9]")


class KneserNey:
    """Interpolated Kneser-Ney; one discount per order D = n1 / (n1 + 2 n2) (Chen & Goodman 1998)."""

    def __init__(self, sentences: List[List[str]], order: int = 5):
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

    def sentence_log10(self, tokens: Sequence[str]) -> float:
        o = self.order
        w = ["<s>"] * (o - 1) + [t if t in self.vocab else "<unk>" for t in tokens] + ["</s>"]
        return sum(math.log10(self.prob(w[i], tuple(w[i - o + 1:i]))) for i in range(o - 1, len(w)))


class NgramTokenizer:
    """The minimal tokenizer interface Step 1 uses (encode, model_max_length, mask_token)."""
    mask_token = None
    pad_token = eos_token = "</s>"

    def __init__(self, tokenize, model_max_length: int = CONTEXT_TOKENS):
        self._tokenize = tokenize
        self.model_max_length = model_max_length

    def tokenize_line(self, line: str) -> List[str]:
        return self._tokenize(line)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[str]:
        toks = []
        for line in str(text).split("\n"):
            toks += self._tokenize(line) + (["</s>"] if add_special_tokens else [])
        return toks


class NgramLM:
    """Model object for cac.score_candidates: log P(prompt) = sum of line sentence scores."""
    is_ngram = True
    config = None

    def __init__(self, line_log10, tokenizer: NgramTokenizer):
        self._line_log10 = line_log10
        self.tokenizer = tokenizer
        self._cache: Dict[str, float] = {}

    def line_logprob(self, line: str) -> float:
        v = self._cache.get(line)
        if v is None:
            if len(self._cache) > 500_000:
                self._cache.clear()
            v = self._cache[line] = self._line_log10(self.tokenizer.tokenize_line(line)) * LN10
        return v

    def prompt_logprob(self, prompt: str) -> float:
        return sum(self.line_logprob(line) for line in prompt.split("\n"))

    def eval(self):
        return self


def score_ngram(model: NgramLM, prompt: str, candidates: Mapping[str, Sequence[str]], blank: str) -> Dict[str, float]:
    from cac.scoring import _forms, _uniform, merge_and_normalize
    if prompt.find(blank) == -1:
        return _uniform(candidates)
    scores = {form: model.prompt_logprob(prompt.replace(blank, form)) for form in _forms(candidates)}
    return merge_and_normalize(scores, candidates)


def load_ngram(model_config: Dict):
    spec = model_config["ngram"]
    if spec["backend"] == "kn":
        tok = NgramTokenizer(lambda s: _KN_TOKEN.findall(s.lower()))
        print(f"Training {spec.get('order', 5)}-gram Kneser-Ney on {spec['corpus']} ...")
        sents = [l.split() for l in open(spec["corpus"], encoding="utf-8") if l.strip()]
        lm = KneserNey(sents, order=spec.get("order", 5))
        return NgramLM(lm.sentence_log10, tok), tok
    if spec["backend"] == "kenlm":
        import kenlm
        import spacy
        m = kenlm.Model(spec["path"])
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"])
        tok = NgramTokenizer(lambda s: [t.text for t in nlp.make_doc(s.strip()) if not t.is_space])
        return NgramLM(lambda toks: m.score(" ".join(toks), bos=True, eos=True), tok), tok
    raise ValueError(f"unknown n-gram backend {spec['backend']!r}")
