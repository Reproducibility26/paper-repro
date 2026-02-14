# Coalition Divergence Law

This repository contains the code to reproduce the experiments and figures from the paper Coalition Divergence Law. It includes a 4-stage pipeline for synthetic rotation-model data, plus per-figure scripts that generate the plots used in the paper. Figures 5-6 use the Amazon Books review dataset and require external data.

## Repository Layout

- `Figure1/` - `Figure4/` and `Figure7/` - `Figure17/`
  - `fig1.py` - `fig4.py` and `fig7.py` - `fig17.py` — Scripts to generate the synthetic-experiment figures (Figures 1–4 and 7–17).
- `Figure5_6/`
  - `amazon_books_appendix.py`, `filter_meta_books.py`, `filter_meta_by_asins.py`, `scan_reviews_books_eligibility.py` — Amazon Books appendix pipeline and helper scripts for generating Figures 5 and 6.
  - `README.md` — Step-by-step instructions for generating Figures 5 and 6.

- `rotation_generator.py` — Generates synthetic rotation-model data.
- `evidence_and_baselines.py` — Computes evidence increments and baseline tables.
- `exposure_attribution.py` — Computes per-account attribution scores.
- `metric_eval.py` — Computes metrics and curves.
- `requirements.txt` — Version-constrained Python dependencies.

Run commands from the repository root (the folder containing `Figure1/`, `Figure2/`, `metric_eval.py`, etc.).

## System Requirements

- OS tested: Ubuntu 24.04.3 LTS
- Python: 3.12
- Hardware: Intel Core Ultra 9 285K × 24, RAM 32GB
- Expected runtime:
  | Figure | Expected Runtime |
  | --- | --- |
  | Fig. 1 | ~18s |
  | Fig. 2 | ~23s |
  | Fig. 3 | ~11.2m |
  | Fig. 4 | ~11.1m |
  | Figs. 5–6 | ~7.2m (excludes one-time data download) |
  | Fig. 7 | ~23s |
  | Fig. 8 | ~23.0m |
  | Fig. 9 | ~12.4m |
  | Fig. 10 | ~21.6m |
  | Fig. 11 | ~26s |
  | Fig. 12 | ~87.3m |
  | Fig. 13 | ~11.1m |
  | Fig. 14 | ~18.9m |
  | Fig. 15 | ~39.4m |
  | Fig. 16 | ~46s |
  | Fig. 17 | ~1.8m |
- Notes:
  - Saving plots to PDF does not require LaTeX unless `matplotlib` is configured with `usetex=True`.
    If you see LaTeX errors, install a TeX distribution (TeX Live / MacTeX / MiKTeX) or disable `usetex`.


## Setup (pip + venv)

From the root of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Sanity check:

```bash
python -c "import numpy, matplotlib; print('OK')"
```

## Reproducing Paper Results

> All commands below assume you are in the repository root and your environment is activated.

### Figures 1–17 (excluding Figures 5 and 6)

Run a figure script from the repository root, for example:
```bash
python3 Figure1/fig1.py
```

Outputs are written to `Figure1/data` and `Figure1/figure`.

- For auditability, each synthetic experiment also saves its configuration and manifest files alongside the generated figures.

### Figure 5 and 6

See `Figure5_6/README.md` for the Figure 5–6 pipeline and instructions.

## Rotation Model Pipeline (Optional)

Figure scripts call the same 4-stage pipeline in order:

`rotation_generator.py` → `evidence_and_baselines.py` → `exposure_attribution.py` → `metric_eval.py`

This section is for understanding or partial runs; for reproduction, run the figure scripts directly.

---