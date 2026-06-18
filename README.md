# Measuring Form and Function in Language Models

Code for the paper **"Measuring Form and Function in Language Models"** (Vázquez Martínez & Yang, UPenn Linguistics).

We evaluate 47 language models on two benchmarks for English determiner usage derived from child language research: a formal test of syntactic productivity (Yang 2013) and a functional test of discourse-driven reference (Gleitman & Yang 2022). Both benchmarks are applied to child-directed speech from the Manchester CHILDES corpus and evaluated via a novel **Contextual Alternative Choice (CAC)** prompting paradigm.

**Key finding:** across 49 model–architecture evaluations (47 distinct models; the two mixed-objective gpt-bert models are scored under both MLM and AR), 30 pass the formal D×N test and only 3 pass the discourse-functional TPR test (*ltg-bert-bnc*, *roberta-base*, *t5-base*). Only two models—*ltg-bert-bnc* (100M-word BNC) and *roberta-base* (30B words)—pass both, but both are trained on far more data than is available to 2–3-year-old children (~10–20M words). No model trained on a developmentally plausible amount of data passes both tests simultaneously.

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
├── run_pipeline.py                 # one-command driver (runs Steps 1–5 in order)
├── run_experiments.py              # Step 1 — CAC inference for all models
├── human_baseline.ipynb            # Step 2 — human TPR baseline, distance analysis, exports
├── compute_analytical_metrics.py   # Step 3 — D×N overlap and TPR from stored probabilities
├── lm_tpr_analysis.ipynb           # Step 4 — per-model TPR tests + context-corrected analyses
├── analysis.ipynb                  # Step 5 — paper figures and tables
├── build_annotation_sample.py      # optional — draft human-annotation sample (after Step 4)
├── cac_utils.py                    # determiner-study glue (corpus, extraction, TPR prep, processing)
├── src/
│   ├── cac/                        # reusable CAC core (pip-installable): scoring, models, readout, context
│   └── determiner/                 # this study's client: pair_extractors, overlap_list_v2, analytical_metrics
├── model_configs.json              # evaluated models (name, type, description)
├── pyproject.toml                  # installs the `cac` + `determiner` packages (pip install -e .)
├── figures/                        # paper figures + tables (.png/.tex/.csv)
├── results/
│   ├── overlap/                    # per-model D×N overlap summaries + human baseline
│   └── tpr/                        # TPR model results + human baseline CSVs
├── data/
│   └── dn.txt                      # flat D×N pair list for standalone overlap_list_v2 use
└── docs/REORG.md                   # reorganization design record (CAC carve, decisions, status)
```

---

## Setup

```bash
pip install transformers torch scipy numpy pandas tqdm spacy nltk
python -m spacy download en_core_web_sm
pip install -e .          # installs the `cac` core and `determiner` study packages
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

The analysis runs in five sequential steps, each depending on the previous. The simplest path is the driver:

```bash
python run_pipeline.py            # full pipeline (Steps 1–5)
python run_pipeline.py --from 2   # skip the expensive inference; re-run analysis from cached predictions
python run_pipeline.py --list     # show the step plan
```

The individual steps, for reference:

### Step 1 — Model Inference: `run_experiments.py`

Runs CAC prompting for all models on the Manchester corpus. For each model × dyad × utterance, scores *a* and *the* in context and saves the result.

CHILDES-style speaker labels (`*CHI`/`*MOT`) are the default (the paper setting); pass `--discourse-label-style spoken` for `MOTHER`/`CHILD` labels instead.

```bash
# Experiment 1 (ablation): single utterance, no discourse context
python run_experiments.py isolated --models model_configs.json

# Experiment 2 (main): full discourse context (CHILDES labels by default)
python run_experiments.py discourse --models model_configs.json

# Overwrite existing results (default: skip if output exists)
python run_experiments.py discourse --models model_configs.json --force
```

**Outputs** (per-model, per-dyad CSV files):

| Directory | Experiment |
|---|---|
| `output/manchester_masked/` | Experiment 1 — isolated (no context) |
| `output/manchester_discourse_childes/` | Experiment 2 — discourse context, CHILDES labels (**main, used in paper**) |

Within each: `{model_type}/{model_name}/{child_name}/child_predictions.csv` and `mother_predictions.csv`. Key columns: `predicted_det_argmax` (argmax), `predicted_det_sample` (probabilistic draw, used in paper), `P_a`, `P_the`.

### Step 2 — Human Baseline: `human_baseline.ipynb`

Run **after** Step 1 and **before** `compute_analytical_metrics.py`. It prepares Manchester corpus metadata and computes the **human-side** baselines — the per-model TPR tests now live in Step 4.

**What it does:**

1. Loads Manchester corpus and generates `output/manchester_tpr_childes/tpr_cached_inputs.csv` (TPR case metadata, with `true_dist`, consumed by Steps 3–4)
2. Computes the restricted-noun Manchester human TPR baseline (`results/tpr/tpr_human_baseline.csv`) and the unrestricted baseline used for reported results
3. Computes TPR for tpr-data other-adult speakers (17 dyads), **validates** corpus equivalence (Welch tests), and establishes the pooled adult TPR population mean (~0.215)
4. Characterizes the human baseline by antecedent distance / context window → `figures/human_tpr_vs_window.png`, `figures/human_antecedent_distance_hist.png`, `results/tpr/human_tpr_by_window.csv` (and `human_switch_rate_by_window.csv`, `human_antecedent_distance_counts.csv`, `human_tpr_by_distance_summary.csv`)
5. Exports the human artifacts consumed downstream:
   - `results/tpr/tpr_human_summary.csv` — per-speaker aggregate TPR + tests
   - `results/tpr/tpr_human_perdyad_references.csv` — per-dyad reference TPRs (consumed by Step 4)

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

### Step 4 — Model TPR Analysis: `lm_tpr_analysis.ipynb`

Corpus-free, model-side TPR analysis. Consumes Step 3's `analytical_tpr_all_results.csv`, the Step 1 prediction distributions, and the Step 2 human artifacts. Produces:

- the canonical **four-test per-model TPR summary** → `results/tpr/tpr_model_summary_analytical.csv` (one-sample vs the pooled adult mean, Welch vs other-adult dyads, paired vs children, paired vs caretakers) — consumed by Step 5;
- a **context-window validity audit** (`in_window` = antecedent within each model's window, verified against the models' actual inputs) and antecedent-visibility breakdown;
- the **corrected, context-aware comparison** (each model on its in-window child sites, paired vs children on the same sites) → `results/tpr/tpr_model_corrected_in_window.csv`.

```bash
jupyter nbconvert --to notebook --execute --inplace lm_tpr_analysis.ipynb
```

**Outputs** (the TPR audits + corrected figures, tracked in git):

| File | Description |
|---|---|
| `results/tpr/tpr_model_summary_analytical.csv` | Canonical four-test per-model TPR summary (→ Step 5) |
| `results/tpr/tpr_model_corrected_in_window.csv` | Per-model in-window (context-corrected) TPR + matched test |
| `figures/appendix_full_results_table_corrected.tex` | Appendix: corrected results table (in-window TPR, matched test) |
| `figures/appendix_full_tpr_figure_corrected.png` | Appendix: corrected TPR by context-window tier |
| `figures/appendix_full_tpr_vs_window_models_figure.png` | Appendix: per-family model TPR vs context window *K* |

### Step 5 — Paper Artifacts: `analysis.ipynb`

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
| Final Summary | Cross-experiment significance table; full results table (all evaluated models) |
| Tables and Figures for paper | Paper artifacts written to `figures/` |

**Paper artifacts saved to `figures/`:**

| File | Description |
|---|---|
| `main_three_facet_figure.png` | Main figure: representative overlap + TPR + accuracy |
| `appendix_full_overlap_models_figure.png` | Appendix: all models by family (Exp. 2) |
| `appendix_full_overlap_human_figure.png` | Appendix: human baseline overlap |
| `appendix_full_tpr_figure.png` | Appendix: full TPR results for all models |
| `appendix_accuracy_figure.png` | Appendix: per-model determiner accuracy |
| `appendix_full_results_table.csv/.tex` | Appendix: complete results for all evaluated models |
| `main_joint_table.csv/.tex` | Main paper: selected representative model table |
| `human_baseline_overlap.tex` | Human baseline overlap statistics |

The **context-corrected TPR audits/figures** (`appendix_full_results_table_corrected.tex`, `appendix_full_tpr_figure_corrected.png`, `appendix_full_tpr_vs_window_models_figure.png`) come from **Step 4**, and the **human antecedent-distance figures** (`human_tpr_vs_window.png`, `human_antecedent_distance_hist.png`) from **Step 2**; `analysis.ipynb` preserves them when regenerating its own artifacts, so a full `run_pipeline.py` produces the complete figure set.

---

## How Each Claim Is Tested

### Formal Productivity

For each model × dyad: extract D×N pairs from CAC predictions, compute the expected empirical overlap analytically from stored probability distributions (Eq. 2 in paper), and compute the Yang (2013) prediction from the analytical bias and corpus statistics (N unique nouns, S total tokens). The 12 (empirical, predicted) pairs are submitted to a paired t-test. Pass if *p* > 0.05.

Implementation: `src/determiner/overlap_list_v2.py` (Yang 2013 formula), `src/determiner/analytical_metrics.py` (analytical overlap and bias), `compute_analytical_metrics.py` (per-model aggregation), `analysis.ipynb` (testing and figures).

### Discourse-Functional Grammar

For each model × dyad: find CAC sites where the same noun was mentioned by the *other* speaker (with any determiner) earlier in the same session. TPR = expected probability of switching from the prior human determiner, computed analytically from stored *P*(the) values. Four statistical comparisons are run against the human baseline.

Implementation: `src/determiner/analytical_metrics.py` (analytical TPR), `compute_analytical_metrics.py` (per-model aggregation), `human_baseline.ipynb` (human baseline + per-dyad references), `lm_tpr_analysis.ipynb` (the four per-model statistical comparisons, context-window audit, and corrected analyses), `analysis.ipynb` (TPR figures/tables).

---

## Model Configurations

`model_configs.json` lists the evaluated models — **49 entries for 47 distinct models**, since the two mixed-objective models (`*-gpt-bert-mixed`) are listed twice, once as `mlm` and once as `ar`, to score them under both architectures. Each entry:

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
