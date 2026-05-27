# Measuring Form and Function in Language Models

Code for the paper **"Measuring Form and Function in Language Models"** (Vázquez Martínez & Yang, UPenn Linguistics).

We evaluate 45 language models on two benchmarks for English determiner usage derived from child language research: a formal test of syntactic productivity (Yang 2013) and a functional test of discourse-driven reference (Gleitman & Yang 2022). Both benchmarks are applied to child-directed speech from the Manchester CHILDES corpus and evaluated via a novel **Contextual Alternative Choice (CAC)** prompting paradigm.

**Key finding:** 30/45 models pass the formal D×N test; only 3/45 pass the TPR test (*ltg-bert-bnc*, *roberta-base*, *t5-base*). Only two models—*ltg-bert-bnc* (100M-word BNC) and *roberta-base* (30B words)—pass both, but both are trained on far more data than is available to 2–3-year-old children (~10–20M words). No model trained on a developmentally plausible amount of data passes both tests simultaneously.

---

## The Two Benchmarks

### 1. Formal Productivity — D×N Overlap (Yang 2013)

Children demonstrate *syntactic productivity* by distributing both determiners (*a* and *the*) broadly across noun types. Yang (2013) formalizes this: given a sample of determiner–noun (D×N) tokens, we can predict how many noun types should appear with *both* determiners, based on the sample size, the number of unique noun types, and the empirical determiner bias *b* (~0.82, stable across English corpora).

**Pass criterion:** the model's empirical overlap across 12 dyads is statistically indistinguishable from the Yang (2013) prediction (paired t-test, *p* > 0.05).

### 2. Discourse-Functional Grammar — TPR (Gleitman & Yang 2022)

The *Transitional Probability of Reference* (TPR) captures the core discourse function of the English article system: speakers use *the* (definite) when the referent was just introduced by the *other* speaker, and *a* (indefinite) otherwise. This cross-speaker pattern is acquired early and is stable across adults.

**Pass criterion:** the model's TPR is not significantly different from the human baseline — assessed via a 1-sample t-test against the adult population mean, a Welch t-test against other-adult dyads, and paired t-tests against Manchester children and caretakers (all *p* > 0.05).

### The CAC Prompting Method

At every attested D×N site in a child utterance, the model receives the full preceding transcript session as discourse context and assigns probabilities to *a* vs. *the*. This makes the evaluation:
- **Identical across architectures** (MLM, AR, seq2seq all score the same candidates)
- **Grounded in real child-directed speech** (no templated stimuli)
- **Discourse-sensitive** (context encodes prior mentions of each noun)

The paper uses **probabilistic (sampled) predictions** rather than argmax to avoid the model collapsing to a memorized dominant choice at each attested site.

---

## Repository Structure

```
.
├── run_experiments.py              # Step 1 — CAC inference for all models
├── compute_analytical_metrics.py   # Step 2 — D×N overlap and TPR from stored probabilities
├── human_baseline.ipynb            # Step 3 — TPR human baseline + model statistical tests
├── analysis.ipynb                  # Step 4 — paper figures and tables
├── cac_utils.py                    # Core library (corpus, model loading, scoring, metrics)
├── analytical_metrics.py           # Analytical formulas for overlap and TPR
├── pair_extractors.py              # spaCy D×N extraction (DeterminerNounExtractor)
├── overlap_list_v2.py              # Yang (2013) expected overlap formula
├── model_configs.json              # 45 evaluated models (name, type, description)
├── figures/                        # Paper figures, tables (.png/.tex/.csv), and methodology assets
├── results/
│   ├── overlap/                    # Per-model D×N overlap summaries + human baseline
│   └── tpr/                        # TPR model results + human baseline CSVs
└── data/
    └── dn.txt                      # Flat D×N pair list for standalone overlap_list_v2.py use
```

---

## Setup

```bash
pip install transformers torch scipy numpy pandas tqdm spacy nltk
python -m spacy download en_core_web_sm
```

Device selection is automatic: MPS (Apple Silicon) → CUDA → CPU.

### Data

The Manchester CHILDES corpus and the tpr-data corpus must be downloaded from [TalkBank/CHILDES](https://childes.talkbank.org/) and placed at:

```
data/CHILDES/Manchester/     # 12 mother-child dyads (Anne, Aran, Becky, ...)
data/CHILDES/tpr-data/       # 17 dyads with other-adult speakers (used for TPR baseline)
```

---

## Running the Pipeline

The analysis runs in four sequential steps. Each step depends on the outputs of the previous one.

### Step 1 — Model Inference: `run_experiments.py`

Runs CAC prompting for all models on the Manchester corpus. For each model × dyad × utterance, scores *a* and *the* in context and saves the result.

The paper uses CHILDES-style speaker labels (`*CHI`/`*MOT`). Pass `--discourse-label-style childes` to reproduce paper results.

```bash
# Experiment 1 (ablation): single utterance, no discourse context
python run_experiments.py isolated --models model_configs.json

# Experiment 2 (main): full discourse context with CHILDES speaker labels
python run_experiments.py discourse --models model_configs.json --discourse-label-style childes

# Overwrite existing results (default: skip if output exists)
python run_experiments.py discourse --models model_configs.json --discourse-label-style childes --force
```

**Outputs** (per-model, per-dyad CSV files):

| Directory | Experiment |
|---|---|
| `output/manchester_masked/` | Experiment 1 — isolated (no context) |
| `output/manchester_discourse_childes/` | Experiment 2 — discourse context, CHILDES labels (**main, used in paper**) |

Within each: `{model_type}/{model_name}/{child_name}/child_predictions.csv` and `mother_predictions.csv`. Key columns: `predicted_det_argmax` (argmax), `predicted_det_sample` (probabilistic draw, used in paper), `P_a`, `P_the`.

### Step 2 — Human Baseline: `human_baseline.ipynb`

Run this notebook **after** Step 1 and **before** `compute_analytical_metrics.py`. It prepares Manchester corpus metadata (TPR case index, human TPR baseline), processes the tpr-data corpus to establish the adult-to-adult TPR reference, and runs all four statistical comparisons per model.

**What it does:**

1. Loads Manchester corpus and generates `output/manchester_tpr_childes/tpr_cached_inputs.csv` (TPR case metadata consumed by Step 3)
2. Computes the restricted-noun Manchester human TPR baseline (`results/tpr/tpr_human_baseline.csv`)
3. Computes TPR for tpr-data other-adult speakers (17 dyads)
4. **Validates** that tpr-data is statistically equivalent to Manchester (Welch tests)
5. Establishes the global adult TPR population mean (~0.215) used as the 1-sample test reference
6. Runs four comparisons per model:
   - 1-sample t-test vs. adult population mean
   - Welch t-test vs. 17 other-adult dyad TPRs
   - Paired t-test vs. Manchester children (12 dyads)
   - Paired t-test vs. Manchester caretakers (12 dyads)
7. Exports two CSVs consumed by `analysis.ipynb`:
   - `results/tpr/tpr_human_summary.csv` — human baseline statistics
   - `results/tpr/tpr_model_summary_analytical.csv` — per-model test results

### Step 3 — Analytical Metrics: `compute_analytical_metrics.py`

Reads the stored probability distributions from Step 1 and computes all D×N overlap and TPR metrics analytically — no model inference needed. Requires `tpr_cached_inputs.csv` from Step 2.

```bash
python compute_analytical_metrics.py
# or with explicit paths:
python compute_analytical_metrics.py \
    --discourse-dir output/manchester_discourse_childes \
    --tpr-dir output/manchester_tpr_childes
```

**Outputs** (written to `results/` and tracked in git):
- `results/overlap/{type}/{model}/analytical_overlap_summary.csv` — per-model D×N overlap (one file per model)
- `results/tpr/analytical_tpr_all_results.csv` — TPR for all models (one aggregate file)

### Step 4 — Paper Artifacts: `analysis.ipynb`

Loads all results and generates every figure and table in the paper. Configuration is in **cell 2**:

```python
TARGET_METHOD  = "prob"    # "prob" (paper) | "det" (deterministic, for comparison)
TARGET_SPEAKER = "child"   # "child" (paper) | "mother" (for comparison)
```

Changing these two variables and re-running from cell 6 onward switches the entire analysis to a different subset without touching any other code.

| Notebook section | Content |
|---|---|
| Experiment 1: Isolated | Family-grouped D×N overlap scatter (context ablation) |
| Experiment 2: Discourse | Human overlap figure + family-grouped model overlap figure |
| Experiment 3: TPR | Full TPR figure with SD bars, pass shading, human reference lines |
| Final Summary | Cross-experiment significance table; full results table (all 45 models) |
| Tables and Figures for paper | Paper artifacts written to `figures/` |

**Paper artifacts saved to `figures/`:**

| File | Description |
|---|---|
| `main_three_facet_figure.png` | Main figure: representative overlap + TPR + accuracy |
| `appendix_full_overlap_models_figure.png` | Appendix: all models by family (Exp. 2) |
| `appendix_full_overlap_human_figure.png` | Appendix: human baseline overlap |
| `appendix_full_tpr_figure.png` | Appendix: full TPR results for all models |
| `appendix_full_results_table.csv/.tex` | Appendix: complete results for all 45 models |
| `main_joint_table.csv/.tex` | Main paper: selected representative model table |
| `human_baseline_overlap.tex` | Human baseline overlap statistics |

---

## How Each Claim Is Tested

### Formal Productivity

For each model × dyad: extract D×N pairs from CAC predictions, compute the expected empirical overlap analytically from stored probability distributions (Eq. 2 in paper), and compute the Yang (2013) prediction from the analytical bias and corpus statistics (N unique nouns, S total tokens). The 12 (empirical, predicted) pairs are submitted to a paired t-test. Pass if *p* > 0.05.

Implementation: `overlap_list_v2.py` (Yang 2013 formula), `analytical_metrics.py` (analytical overlap and bias), `compute_analytical_metrics.py` (per-model aggregation), `analysis.ipynb` cells 10–17 (testing and figures).

### Discourse-Functional Grammar

For each model × dyad: find CAC sites where the same noun was mentioned by the *other* speaker (with any determiner) earlier in the same session. TPR = expected probability of switching from the prior human determiner, computed analytically from stored *P*(the) values. Four statistical comparisons are run against the human baseline.

Implementation: `analytical_metrics.py` (analytical TPR), `compute_analytical_metrics.py` (per-model aggregation), `human_baseline.ipynb` (all four statistical comparisons and export), `analysis.ipynb` cells 18–19 (TPR figure).

---

## Model Configurations

`model_configs.json` lists all 45 evaluated models. Each entry:

```json
{"name": "HuggingFace/repo-id", "type": "mlm|ar|seq2seq", "description": "..."}
```

The `type` field controls the scoring method:
- `mlm` — mask the determiner position, score via masked token probabilities
- `ar` — score each full candidate string (*a …* vs. *the …*) and normalize
- `seq2seq` — use span sentinel tokens, compare encoder–decoder scores

Models include BabyLM competition baselines and winning submissions (2023–2025), NYU-MLL RoBERTa training-data ablations (1M–1B tokens), and large reference models (GPT-2, OPT-125M, RoBERTa-base/large, T5-base, ltg-bert-bnc).

---

## Citation

```bibtex
@inproceedings{vazquezmartinez-yang-2025-form-function,
    title     = {Measuring Form and Function in Language Models},
    author    = {V\'{a}zquez Mart\'{i}nez, Hector Javier and Yang, Charles},
    booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics},
    year      = {2025},
}
```
