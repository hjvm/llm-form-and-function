# Replication guide

Where every committed result comes from, which step reads it, and what has to stay fixed for the numbers to come out the same.

## The chain

```
data/CHILDES/ transcripts (not committed; data/README.md)
  + results/site_inventory/*.csv        frozen determiner–noun sites (spaCy en_core_web_sm 3.7.1)
    ├─ Step 1  01_run_inference.py       → output/manchester_discourse_childes/   P(a), P(the) per site and model
    ├─ Step 2  02_human_baseline.ipynb   → output/manchester_tpr_childes/tpr_cached_inputs.csv
    │                                    → results/tpr/ human files, figures/human_*
    ├─ Step 3  03_compute_metrics.py     → results/overlap/, results/tpr/analytical_tpr_all_results.csv
    ├─ Step 4  04_model_tpr.ipynb        → results/tpr/tpr_model_*.csv, figures/*_corrected*
    └─ Step 5  05_paper_artifacts.ipynb  → figures/ (paper tables and figures)
```

Steps 2–5 are deterministic. Step 1 is deterministic given the model weights (models are loaded from the Hugging Face Hub at their current revision; a `"revision"` key in `configs/models.json` pins one).

## What has to stay fixed

**The site inventories.** Every benchmark is computed over the singular `a/an/the + noun` sites in the transcripts. They were extracted with spaCy 3.7 and `en_core_web_sm` 3.7.1. Newer models tag the same sentences differently: spaCy 3.8 with `en_core_web_sm` 3.8.0 finds 976 sites where 3.7.1 finds 1,013 in a test sample of the same sentences. The inventories used in the paper are therefore committed in `results/site_inventory/`, and `determiner.sites.load_or_extract_det_noun_locations` copies them into the `output/` caches before anything else runs. Extraction from scratch only happens when no inventory exists, prints a warning, and reproduces the committed files only under `pip install -e ".[sites]"`.

The inventory stores transcript paths relative to the repository root (`./data/CHILDES/Manchester/Gail/020609.cha`), which is why every step changes to the root first.

**The joins.** Step 3 joins model predictions to TPR cases on `(child_name, speaker, sentence_id)`, where `sentence_id` is the row number of the site in the Manchester inventory. Predictions and TPR cases built from different inventories will not line up.

## Committed results

| File | Written by | Read by |
|---|---|---|
| `results/site_inventory/manchester_det_noun_locations.csv` | spaCy extraction (frozen) | Steps 1–3 |
| `results/site_inventory/tpr_data_det_noun_locations.csv` | spaCy extraction (frozen) | Step 2 |
| `results/overlap/overlap_human_baseline.csv` | Step 3 | Step 5 |
| `results/overlap/{mlm,ar,seq2seq}/<model>/analytical_overlap_summary.csv` | Step 3 | Step 5 |
| `results/overlap/dxn_bayes_factors.csv` | Step 5 | paper (Table 1, appendix) |
| `results/tpr/analytical_tpr_all_results.csv` | Step 3 | Step 4 |
| `results/tpr/tpr_human_summary.csv` | Step 2 | Steps 4–5 |
| `results/tpr/tpr_human_perdyad_references.csv` | Step 2 | Step 4 |
| `results/tpr/tpr_human_baseline_unrestricted.csv` | Step 2 | Step 5 (appendix human table) |
| `results/tpr/tpr_human_baseline.csv` | Step 2 | not used for reported results (attested-noun subset) |
| `results/tpr/human_tpr_by_window.csv` | Step 2 | Step 4 |
| `results/tpr/human_switch_rate_by_window.csv`, `human_antecedent_distance_counts.csv`, `human_tpr_by_distance_summary.csv` | Step 2 | supplementary |
| `results/tpr/tpr_model_summary_analytical.csv` | Step 4 | Step 5 |
| `results/tpr/tpr_model_corrected_in_window.csv` | Step 4 | supplementary |

## Paper artifacts

| Paper | File | Step |
|---|---|---|
| Table 1 | `figures/main_joint_table.tex` | 5 |
| Figure 2 | `figures/main_three_facet_figure.png` | 5 |
| Appendix: human D×N table | `figures/human_baseline_overlap.tex` | 5 |
| Appendix: human overlap figure | `figures/appendix_full_overlap_human_figure.png` | 5 |
| Appendix: full results table | `figures/appendix_full_results_table.tex` | 5 |
| Appendix: model overlap, accuracy, TPR figures | `figures/appendix_full_overlap_models_figure.png`, `appendix_accuracy_figure.png`, `appendix_full_tpr_figure.png` | 5 |
| Supplementary: context-window analyses | `figures/*_corrected*`, `appendix_full_tpr_vs_window_models_figure.png` | 4 |
| Supplementary: antecedent distance | `figures/human_tpr_vs_window.png`, `human_antecedent_distance_hist.png` | 2 |

The CAC diagram (Figure 1) and the model information tables were made by hand. Captions in the paper were edited after generation; the numbers are the generated ones.

## Checking a re-run

After `python run_pipeline.py --from 2`, `git status` should show only the executed notebooks as changed: every CSV and `.tex` file in `results/` and `figures/` should be byte-identical. PNGs can differ at the byte level across matplotlib versions, and Bayes factors can differ in the 12th significant digit across SciPy versions (numerical integration).
