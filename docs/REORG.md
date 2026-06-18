# Pipeline reorganization & CAC carve — design record

Living record of the repository overhaul: extracting a reusable **CAC** core from the
determiner study, separating producers from viewers, and adding a one-command driver.
Update this file as decisions change.

## Why

The pipeline had grown to 5 ordered steps that **alternate `.py` and `.ipynb`**, with the
notebooks *load-bearing* (they wrote canonical artifacts other steps consumed) and several
implicit cross-step contracts (run order, `figures/` allowlist wipe, the
`--discourse-label-style` flag). Hard to replicate, easy to run wrong. Separately, a second
author group wants to reuse the CAC method for **Root Infinitives** across multiple
languages, corpora, and candidate sets (see "External consumer" below) — which requires a
corpus/candidate/language-agnostic CAC core, not the determiner-specific tangle.

## Target structure

```
src/
├── cac/                     # reusable, corpus/candidate/language-agnostic CORE (pip-installable)
│   ├── scoring.py           # score_candidates(...) + merge_and_normalize + per-arch scorers
│   ├── models.py            # load_model(..., revision=None)  (checkpoint support)
│   ├── readout.py           # mass_on, surprisal, choose_prediction  (generic, N-category)
│   └── context.py           # prepare_context_windows + reconstruct_context + format_context_input  (TODO)
└── determiner/              # OUR study as a CLIENT of cac/ (moved; modules kept as-is)
    ├── pair_extractors.py   # spaCy D×N extraction
    ├── overlap_list_v2.py   # Yang (2013) expected-overlap formula
    └── analytical_metrics.py# analytical overlap + TPR formulas
    # NOTE: moved byte-identical (git mv). Finer merge/rename into extraction.py/overlap.py/
    # tpr.py is a deferred cosmetic step, not done (would change content → re-verify).

pipeline/                    # .py ARTIFACT PRODUCERS — write results/ (intermediate, consumed) (TODO)
notebooks/                   # .ipynb VISUALIZERS — read results/, write figures/ (.png AND .tex tables) (TODO)
run_pipeline.py              # one-command driver (.py, not make) (TODO)
data/ output/ results/ figures/   # unchanged roles (output/ + data/ gitignored)
```

## Decisions (with rationale)

1. **Producer/viewer split.** Every committed *intermediate* artifact (the `results/*.csv`
   other steps read) is produced by `.py`; notebooks are non-load-bearing viewers.
2. **Paper artifacts (figures AND `.tex`/`.csv` tables) live in the notebooks**, together,
   since both are paper deliverables. (Reverses an earlier idea of a separate table script.)
3. **Driver = `run_pipeline.py`** (Python), not a Makefile — for a non-`make` audience.
4. **`--discourse-label-style` defaults to `childes`** (`*CHI`/`*MOT`, the paper setting →
   output `manchester_discourse_childes`). Flag kept for `spoken`/custom. (DONE)
5. **`cac` is pip-installable** (`pyproject.toml`, src-layout). Kills the `sys.path` hack.
6. **Core boundary = the distribution over candidates.** `cac` scores and stops at the
   normalized distribution (`*_predictions.csv`). Behavioral conversion is task-specific:
   - overlap / predicted-overlap / bias / TPR are **binary, determiner-construction-bound** →
     stay in `determiner/` (do NOT generalize to N categories).
   - generic surprisal-style readouts (`mass_on`, `surprisal`, `choose_prediction`) are
     N-category-general → `cac.readout`. (`mass_on` is the unifying abstraction: our `P(the)`
     and their RI rate are both `mass_on(subset)`.)
7. **`score_candidates` API:**
   - `target_span` = **character offsets** `(start, end)` (keeps the site table
     model/tokenizer-agnostic — one site definition reused across all models/checkpoints).
   - `candidates` = **category → surface-form list** mapping. Within a category the member
     forms' log-scores are merged in log-space (`logaddexp` = summing probability mass); the
     merged category scores are softmax-normalized. The **union of all forms is the closed
     candidate set** (the normalization denominator).
   - Collapsing is opt-in: singleton categories ⇒ per-form distribution; merged categories ⇒
     grouped. Because softmax commutes with the log-space merge, `mass_on(subset)` on the
     per-form output equals the merged-category result.
   - Our determiner spec: `{"a": ["a","an"], "the": ["the"]}` (the a/an merge).
8. **Architecture handling.** AR (full-candidate sentence) and seq2seq (sentinel infill) build
   prompts at the **string level** → multiple multi-token forms per category prompt correctly
   today. MLM uses a single mask token (one subword/form) → multi-token forms need the
   **deferred** pseudo-log-likelihood path. Length bias across unequal-length forms is a
   *scoring* (not prompting) question, decided with the multi-token work.
9. **`cac.context` design:** corpus (sessions of ordered utterances) + target sites (session
   key + line number) → a **model-agnostic master context** boundary (`context_start_line_num`,
   capped at the largest model's window), stored compactly as a line number (no text). At
   inference each model **trims** to its own `max_position_embeddings`. Build-once / trim-per-model
   (ideal for the RI checkpoint sweep). Speaker labeling via a **`speaker_label` resolver,
   default = passthrough on the corpus `speaker_type` column**; determiner study overrides to
   `*CHI`/`*MOT` (bit-for-bit); markerless = explicit `lambda s: ""`.
10. **Multi-token MLM masking (PLL) is deferred** until after the full bit-for-bit gate, per
    instruction.

## Bit-for-bit invariants (must hold for the determiner study)

The carve must change **no** committed number. Anchors:
- within-category merge stays `logaddexp` then `softmax` over categories (== old
  `_normalize_det_log_scores`); verified on 200,007 randomized cases incl. all `-inf` combos.
- token resolution: `a`/`an`/`the` → same ids (`convert_tokens_to_ids`).
- masking / candidate-string assembly byte-identical to today.
- sampling RNG: determiner study keeps `choose_determiner_predictions` (ties → `the`); the
  generic `cac.choose_prediction` (first-max ties) is for adopters only.
- `load_model` passes `revision=None` by default ⇒ identical `from_pretrained` calls.

## External consumer (drives the carve)

RI study ("The Value of Getting It Wrong"): CAC applied to the finite/non-finite alternation
across **English, French, Italian**, over **training checkpoints**, with **per-language,
multi-form candidate categories** (e.g. Italian non-finite = {infinitive, participle,
imperative}). Needs: pluggable corpus + target-site extractor, N-ary multi-form categories,
checkpoint loading, multilingual/tokenizer-agnostic. They write their own behavioral layer
(RI rate = `mass_on(non_finite)`); `cac.readout` + `score_candidates` are shared.

## Status

**Done & verified bit-for-bit**
- `cac` package (scoring, models, readout, context); pip-installed.
- `merge_and_normalize` == `_normalize_det_log_scores` (200k cases).
- `score_candidates` == `extract_determiner_probabilities` end-to-end on real MLM/AR/seq2seq.
- `cac_utils.load_model` / `extract_determiner_probabilities` repointed to `cac`; imports intact.
- `cac.context` carved (`estimate_max_context_lines`, `compute_context_starts`,
  `format_context_input` w/ `speaker_label` resolver, `reconstruct_context`); `cac_utils`'s
  `_estimate_max_context_lines`, `format_discourse_input`, and the estimate+sweep inside
  `prepare_discourse_inputs_all_sessions` repointed. Verified: format battery (36 cases, both
  styles/shapes/budgets) identical; discourse cache boundaries (75,133 rows) identical.
- Determiner study moved to `src/determiner/` (`pair_extractors`, `overlap_list_v2`,
  `analytical_metrics` via `git mv`); imports repointed in `compute_analytical_metrics.py`,
  `cac_utils.py`, `human_baseline.ipynb`. Verified: `analytical_tpr_all_results.csv` md5
  unchanged after a full re-run; all imports resolve.
- Dead code removed from `cac_utils` (the 6 per-arch scorers + helpers superseded by `cac`);
  parses, imports, and the `extract_determiner_probabilities` delegation intact.
- `run_pipeline.py` driver added (Steps 1-5 in order; notebooks via nbconvert; `--from/--to/
  --only/--force/--list`). Verified: `--list` plan, range selection, and an orchestrated Step 3.

**Decision — no physical `pipeline/` + `notebooks/` subdirs (for now).** Relocating
`run_experiments.py` / `compute_analytical_metrics.py` into `pipeline/` would break their
cwd-relative paths (`./output`, `./data`) and `from cac_utils import *` (root, not installed),
and the notebooks remain *load-bearing producers* (decision 2: tables+figures live in them), so a
clean producer/viewer **directory** split isn't achievable without bigger surgery. `run_pipeline.py`
delivers the ordering/one-command value without that risk. The physical relocation is a deferred,
optional cosmetic step.

- Full replication pass (Steps 2-5 via `run_pipeline.py --from 2`, against cached predictions):
  **all 63 `results/*.csv` and all 6 `figures/*.{csv,tex}` bit-identical**; PNGs regenerate clean.
  (Step 1 inference not re-run — impractical for 49 models and only adds MPS nondeterminism; the
  scoring path was already proven bit-identical end-to-end on all 3 architectures.)
- `README.md` updated (5 steps, `cac`/`determiner` packages, `run_pipeline.py`, `childes` default,
  model tests in Step 4).

**Remaining**
- Deferred multi-token MLM (PLL) scoring work (for the RI team's multi-token candidates).
- **Model count (resolved):** `model_configs.json` has **49 entries = 47 distinct models**; the two
  `*-gpt-bert-mixed` models are listed twice (`mlm` + `ar`) to score both architectures → 49
  model×arch evaluation rows. Pass counts (verified from the committed appendix table): D×N 30,
  TPR 3 (ltg-bert-bnc, roberta-base, t5-base), both 2. README updated to 47 models / 49 evaluations.
- Optional: physical `pipeline/`+`notebooks/` relocation; finer `determiner` rename/merge.
