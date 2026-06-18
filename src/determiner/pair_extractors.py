# pair_extractors.py

from abc import ABC, abstractmethod
import spacy

class PairExtractor(ABC):
    """
    Abstract base class for extracting closed-open class pairs.
    Subclasses must implement extract() to return list of (closed, open) tuples.
    """
    @abstractmethod
    def extract(self, text: str) -> list:
        pass


class DeterminerNounExtractor(PairExtractor):
    """
    Extracts D×N pairs directly from lists of text lines,
    encapsulating all spaCy interactions (including batching).
    """
    def __init__(self, spacy_model="en_core_web_sm", max_gap=1, pipe_batch_size=1000):
        self.nlp = spacy.load(spacy_model, disable=["ner"])
        self.max_gap = max_gap
        self.pipe_batch_size = pipe_batch_size

    def extract(self, text):
        """
        Process a text string; returns list of 'det noun' strings.
        """
        return self._extract_from_doc(self.nlp(text))

    def extract_from_lines(self, lines):
        """
        Process a list of text lines; returns list of 'det noun' strings.
        """
        pairs = []
        docs = self.nlp.pipe(lines, batch_size=self.pipe_batch_size)

        for doc in docs:
            pairs.extend(self._extract_from_doc(doc))

        return pairs

    def _extract_from_doc(self, doc):
        pairs = []
        for chunk in doc.noun_chunks:
            if chunk[0].pos_ == "DET" and chunk[0].lower_ in ("a", "an", "the"):
                det = "a" if chunk[0].lower_ in ("a", "an") else "the"

                det_idx = chunk[0].i
                noun_idx = chunk.root.i
                gap = noun_idx - det_idx - 1

                if 0 <= gap <= self.max_gap:
                    if chunk.root.pos_ == "NOUN" and "Sing" in chunk.root.morph.get("Number"):
                        pairs.append(f"{det} {chunk.root.lemma_}")
        return pairs
