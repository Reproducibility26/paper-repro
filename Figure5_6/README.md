## Amazon Books Appendix (Figures 5-6)

These figures use the Amazon Reviews Books dataset and metadata. The data is not included in this repository.

**Required input files**
- `Books.json.gz` (reviews)
- `meta_Books.json.gz` (metadata)

## Dataset download
The Amazon Review Data (2018) page hosts the per-category files. The Books reviews and metadata are provided via a Google Form on that page (select the Books category and download the two files listed above). If you cannot access the form, follow the contact instructions on the dataset page.

Dataset page:
```text
https://nijianmo.github.io/amazon/index
```

**Suggested directory layout (from the repository root)**
```
data/amazon_books/Books.json.gz
data/amazon_books/meta_Books.json.gz
```
---
> All commands below assume you are in the repository root and your environment is activated.

## Step 1: Identify eligible ASINs
This step scans reviews and writes an eligibility list.

```bash
python Figure5_6/scan_reviews_books_eligibility.py \
  --input data/amazon_books/Books.json.gz \
  --outdir out/
```

Optional filters (defaults shown in the script): `--n-min 30`, `--min-total 500`, `--min-valid-weeks 52`.

**Outputs (in `out`)**
- `eligible_asins.txt`
- `asin_stats.csv`

**Note:** this script currently builds output paths with Windows-style backslashes. On Linux/macOS, the files may be created in the current directory with names like `out\\eligible_asins.txt`. If you do not see files under `out/`, check for those filenames in the repo root.

If that happens, you can move them into `out/` without changing any Python code:

```bash
mkdir -p out
mv "out\\eligible_asins.txt" "out/eligible_asins.txt" 2>/dev/null || true
mv "out\\asin_stats.csv" "out/asin_stats.csv" 2>/dev/null || true
```

Alternative (Linux/macOS): set `--outdir .` to avoid backslash paths, then move the files into `out/` if you want them there.

## Step 2: Build a compact metadata CSV
Recommended: restrict metadata to eligible ASINs during extraction.

```bash
python Figure5_6/filter_meta_books.py \
  --input data/amazon_books/meta_Books.json.gz \
  --output out/usable_meta_books.csv \
  --asin-list out/eligible_asins.txt
```

If you skipped `--asin-list`, you can filter afterward:

```bash
python Figure5_6/filter_meta_by_asins.py \
  --meta-csv out/usable_meta_books.csv \
  --asin-list out/eligible_asins.txt \
  --out-csv out/eligible_meta_books.csv
```

## Step 3: Generate Figures 5 and 6

```bash
python Figure5_6/amazon_books_appendix.py \
  --reviews data/amazon_books/Books.json.gz \
  --eligible out/eligible_asins.txt \
  --meta out/usable_meta_books.csv \
  --outdir out \
  --mode legacy
```

Notes:
- `--mode legacy` runs only the appendix pipeline needed for Figures 5-6.
- Default mode is `oracle`, which also exports per-exemplar run directories.
- Use `--no-tex` if TeX is not installed.

**Outputs (in `out`)**
- `amazon_weekly_hist.csv`
- `amazon_weekly_scores.csv`
- `amazon_item_metrics.csv`
- `fig5_Amazon_exemplars.png` and `fig5_Amazon_exemplars.pdf`
- `fig6_Amazon_cdf.png` and `fig6_Amazon_cdf.pdf`

## Plot-only mode
If the CSVs above already exist, you can regenerate the figures without rescanning:

```bash
python Figure5_6/amazon_books_appendix.py \
  --eligible out/eligible_asins.txt \
  --meta out/usable_meta_books.csv \
  --outdir out \
  --plot-only \
  --mode legacy
```
