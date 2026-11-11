# Measuring Form and Function in Language Models

Code, results and paper artifacts for

> Hector Javier Vázquez Martínez and Charles Yang. 2026. **Measuring Form and Function in Language Models.** In *Proceedings of the Second BabyLM Workshop (BabyLM 2026)*, pages 438–458, Budapest, Hungary. Association for Computational Linguistics. [aclanthology.org/2026.babylm-main.35](https://aclanthology.org/2026.babylm-main.35/)

We test whether language models use the English determiners *a* and *the* the way young children do, on two benchmarks from child language research, both run on the child–caretaker conversations of the Manchester corpus (CHILDES):

* **Form: the D×N overlap test** (Yang 2013). A productive grammar combines both determiners with nouns freely, which predicts how many noun types should appear with both. Children match the prediction; a memorized inventory falls short of it.
* **Function: the Transitional Probability of Reference, TPR** (Gleitman & Yang 2022). When one speaker refers back to a noun the other speaker just used, how often do they switch determiner? This tracks the discourse function of definiteness.

Models are evaluated with **Contextual Alternative Choice (CAC)** prompting. At every attested determiner–noun site, the model sees the conversation so far, with the determiner blanked out, and we read off its probability of *a* versus *the*. The same closed choice is scored for masked, autoregressive and encoder–decoder models. All metrics are computed analytically from those probabilities; nothing is sampled.

Across 49 evaluations of 47 models, 30 pass the D×N test and 3 pass the TPR test. Only ltg-bert-bnc and roberta-base pass both, and with JZS Bayes factors only ltg-bert-bnc shows evidence for human-like behavior on both tests. Both models are trained on far more data than a 2–3-year-old has heard.

---

## Reproducing the results

```bash
git clone https://github.com/hjvm/llm-form-and-function.git
cd llm-form-and-function
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -e .                                      # Steps 2-5
pip install -e ".[inference]"                         # add this for Step 1 (PyTorch + Transformers)
```

The pipeline has five steps. `run_pipeline.py` runs any of them in order, from any directory:

| Step | What it does | Needs | Writes |
|---|---|---|---|
| 1 `pipeline/01_run_inference.py` | CAC inference: P(*a*), P(*the*) at every site, for every model | `data/`, a GPU or Apple silicon, about two weeks for all 49 | `output/manchester_discourse_childes/` |
| 2 `pipeline/02_human_baseline.ipynb` | TPR cases; human TPR per dyad; the adult reference (0.215); human tests; antecedent distance | `data/` | `results/tpr/`, `figures/human_*` |
| 3 `pipeline/03_compute_metrics.py` | Analytical D×N overlap and TPR for humans and every model | Step 1 predictions, Step 2 | `results/overlap/`, `results/tpr/analytical_tpr_all_results.csv` |
| 4 `pipeline/04_model_tpr.ipynb` | Per-model TPR tests; context-window audit | Step 1 predictions, Steps 2–3 | `results/tpr/tpr_model_*`, `figures/*_corrected*` |
| 5 `pipeline/05_paper_artifacts.ipynb` | Every table and figure in the paper | committed `results/` only | `figures/` |

There are three ways in, depending on how far back you want to go:

```bash
python run_pipeline.py --only 5     # paper tables and figures from the committed results (seconds; no data)
python run_pipeline.py --from 2     # everything after inference (needs data/ and the model predictions)
python run_pipeline.py              # everything, including inference
python run_pipeline.py --list       # show the steps
```

Each step can also be run by hand, e.g. `python pipeline/03_compute_metrics.py --help` or by opening a notebook in Jupyter. Notebooks change to the repository root on their first cell, so they work from any directory.

**Data.** Steps 1 and 2 need the Manchester corpus and the tpr-data transcripts under `data/CHILDES/`; see [`data/README.md`](data/README.md).

**Determiner–noun sites.** The sites every benchmark is computed over were extracted with spaCy `en_core_web_sm` 3.7.1, and newer spaCy models tag the same sentences differently. The inventories used in the paper are committed in `results/site_inventory/` and copied into the pipeline caches automatically, so spaCy is not needed. To re-extract from scratch, `pip install -e ".[sites]"` installs the pinned model.

**Determinism.** Steps 2–5 are deterministic: the paper tables come out byte-identical, and other results agree to floating-point precision across library versions. Step 1 is deterministic given the model weights on the Hugging Face Hub.

See [`docs/REPLICATION.md`](docs/REPLICATION.md) for where every committed file comes from and which step reads it.

---

## Repository layout

```
run_pipeline.py            runs Steps 1-5, or any subset, in order
pipeline/                  the five steps, numbered in the order they run
configs/models.json        the 49 evaluations (Hugging Face id, architecture type)
src/
  cac/                     CAC prompting and scoring; study-independent
  determiner/              this study
    paths.py               repository paths
    corpus.py              CHAT transcript loading and cleaning
    sites.py               determiner-noun site inventory (spaCy)
    inference.py           Step 1: prompts and model scoring
    tpr.py                 TPR cases and human TPR
    overlap.py             Yang (2013) expected overlap
    analytical_metrics.py  analytical overlap, bias, TPR; JZS Bayes factors
results/                   committed outputs of Steps 2-4
  site_inventory/          the frozen site inventories (spaCy 3.7.1)
  overlap/                 per-model D×N summaries, human baseline, Bayes factors
  tpr/                     model and human TPR summaries
figures/                   paper tables (.tex/.csv) and figures (.png)
data/                      corpora (not committed; see data/README.md)
output/                    caches and model predictions (not committed)
```

`configs/models.json` has 49 entries for 47 models: the two mixed-objective gpt-bert baselines are scored both as masked (`mlm`) and as autoregressive (`ar`) models. The `type` field selects how candidates are scored: `mlm` reads the mask token's distribution, `ar` scores each completed prompt, and `seq2seq` scores the sentinel span.

The overlap formula also works as a standalone tool on a list of `det noun` pairs:

```bash
python -m determiner.overlap pairs.txt
```

---

## Citation

```bibtex
@inproceedings{vazquez-martinez-yang-2026-measuring,
    title     = {Measuring Form and Function in Language Models},
    author    = {V{\'a}zquez Mart{\'i}nez, Hector Javier and Yang, Charles},
    booktitle = {Proceedings of the Second BabyLM Workshop (BabyLM 2026)},
    address   = {Budapest, Hungary},
    publisher = {Association for Computational Linguistics},
    year      = {2026},
    pages     = {438--458},
    url       = {https://aclanthology.org/2026.babylm-main.35/},
    doi       = {10.18653/2026.babylm-main.35},
}
```

MIT license; see `LICENSE`.
