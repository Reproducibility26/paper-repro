#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
amazon_books_appendix.py — Amazon Books appendix pipeline (Figures 5–6).

This script builds aggregate-time signals for Amazon Books items and produces:
  - Fig. 5: exemplar time series (weekly n_t, raw W1, corrected d_t, cumulative D_t)
  - Fig. 6: cohort CDF summaries across eligible items

Modes:
  1) Full pipeline (default):
     a) Read eligible ASINs (eligible_asins.txt) and titles (usable_meta_books.csv)
     b) Stream-scan Books.json.gz reviews to build per-ASIN weekly 5-star histograms
     c) Build a global reference histogram H0 from the first K valid weeks per ASIN
     d) Estimate baseline table b(n) via multinomial Monte Carlo under H0
     e) For each ASIN-week: compute W1, corrected increment d_t = W1 - b(n_t),
        and cumulative D_t = sum_{s<=t} d_s
     f) Write weekly and item-level CSVs, select exemplars, and plot Fig. 5 + Fig. 6

  2) Plot-only mode (--plot-only):
     - Skip scan/MC/metrics computation.
     - Read existing CSVs in --outdir and re-plot Fig. 5 + Fig. 6.
     - Exemplar selection is repeated from amazon_item_metrics.csv and uses title
       canonicalization to avoid duplicate editions in controls.

Inputs:
  - Books.json.gz                 (required in full mode)
  - eligible_asins.txt            (one ASIN per line)
  - usable_meta_books.csv         (must include columns: asin,title)

Outputs (written to --outdir):
  - amazon_weekly_hist.csv        (full mode)
  - amazon_weekly_scores.csv      (full mode; per-ASIN weekly W1/d_t/D_t)
  - amazon_item_metrics.csv       (full mode; per-ASIN summary metrics for selection)
  - fig5_Amazon_exemplars.(png|pdf)
  - fig6_Amazon_cdf.(png|pdf)
  - (optional) oracle_runs/<ASIN>/{meta.json,intervals.csv}  (see --mode)

Usage:
  # Full pipeline (Linux/macOS)
  python amazon_books_appendix.py --reviews Books.json.gz --eligible eligible_asins.txt \
      --meta usable_meta_books.csv --outdir out/

  # Full pipeline (Windows)
  py amazon_books_appendix.py --reviews Books.json.gz --eligible eligible_asins.txt ^
      --meta usable_meta_books.csv --outdir out

  # Plot-only mode (reuse existing CSVs)
  python amazon_books_appendix.py --eligible eligible_asins.txt --meta usable_meta_books.csv \
      --outdir out/ --plot-only

Notes:
  - Week aggregation uses unixReviewTime binned to weeks; only weeks with n_t >= --min-weekly-n are kept.
  - W1 is computed on the 5-star line via CDF differences (bins are {1,2,3,4,5}).
  - Baseline correction uses b(n) estimated under H0 and nearest-n lookup for unseen n_t.
  - If LaTeX text rendering fails, rerun with --no-tex.

"""

import argparse
import csv
import gzip
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# Time helpers
# -----------------------------
def now_s() -> float:
    return time.time()

def fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s/60:.1f}min"
    return f"{s/3600:.2f}h"


# -----------------------------
# Plot styling (publication quality)
# -----------------------------
def setup_pub_rcparams(use_tex: bool = True):
    """
    Configure matplotlib for publication-quality figures with LaTeX fonts.
    If LaTeX isn't installed, run with --no-tex.
    """
    mpl.rcParams.update(
        {
            "text.usetex": bool(use_tex),
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,

            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,

            "lines.linewidth": 1.4,
            "axes.linewidth": 0.8,

            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,

            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.titlesize": 13,
            "legend.fontsize": 11,

            # embed fonts in PDF
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def nice_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out")
    ax.margins(x=0.01)


def save_png_and_pdf(fig, base, dpi=300):
    fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.02)


def short_title(t: str, maxlen: int = 52) -> str:
    t = (t or "").strip()
    if len(t) <= maxlen:
        return t
    return t[: maxlen - 3] + "..."


# -----------------------------
# Title canonicalization & selection helpers
# -----------------------------
def canonicalize_title(title: str) -> str:
    """
    Normalize titles so different editions don't count as different "books"
    for control-selection purposes.

    - lower
    - strip bracketed (...) and [...]
    - remove common edition words
    - remove punctuation -> whitespace
    - collapse whitespace
    """
    if not title:
        return ""
    t = title.lower().strip()
    t = re.sub(r"\([^)]*\)", " ", t)   # remove (...)
    t = re.sub(r"\[[^\]]*\]", " ", t)  # remove [...]
    t = re.sub(
        r"\b(large\s+print|paperback|hardcover|kindle|audio|audiobook|edition|revised|illustrated|boxed\s+set|box\s+set|volume|vol\.?)\b",
        " ",
        t,
    )
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_excluded_title(title: str, exclude_keywords: List[str]) -> bool:
    t = (title or "").lower()
    return any((kw.lower() in t) for kw in exclude_keywords)


def read_asin_set(path: str) -> set:
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            a = line.strip()
            if a:
                s.add(a)
    return s


def read_meta_titles(path_csv: str) -> Dict[str, str]:
    """
    Reads usable_meta_books.csv (must contain asin/title columns).
    """
    asin2title: Dict[str, str] = {}
    with open(path_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = {c.lower(): c for c in (r.fieldnames or [])}
        asin_col = cols.get("asin") or cols.get("asins") or cols.get("asin_id")
        title_col = cols.get("title")
        if not asin_col or not title_col:
            raise RuntimeError(f"Meta CSV must contain columns 'asin' and 'title'. Found: {r.fieldnames}")
        for row in r:
            asin = (row.get(asin_col) or "").strip()
            title = (row.get(title_col) or "").strip()
            if asin:
                asin2title[asin] = title
    return asin2title


# -----------------------------
# Weekly aggregation
# -----------------------------
def parse_unix_time_to_week(ts: int) -> int:
    return int(ts // (7 * 24 * 3600))

def star_to_bin(star: float) -> int:
    s = int(round(float(star)))
    s = min(5, max(1, s))
    return s - 1


@dataclass
class WeekCell:
    n: int
    counts: np.ndarray  # shape (5,)


def scan_reviews_build_weekly_hist(
    reviews_gz: str,
    eligible_asins: set,
    progress_every: int = 2_000_000,
    min_weekly_n: int = 30,
) -> Dict[str, Dict[int, WeekCell]]:
    t0 = now_s()
    per_asin_week = defaultdict(lambda: defaultdict(lambda: WeekCell(0, np.zeros(5, dtype=np.int64))))
    scanned = 0
    kept = 0
    bad = 0

    print("[INFO] Scanning reviews to build weekly histograms")
    print(f"[INFO] input: {reviews_gz}")
    print(f"[INFO] eligible asins: {len(eligible_asins):,}")
    print(f"[INFO] weekly valid if: n_t >= {min_weekly_n}")

    with gzip.open(reviews_gz, "rt", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if progress_every and scanned % progress_every == 0:
                dt = now_s() - t0
                rate = scanned / max(dt, 1e-9)
                print(f"[PROGRESS] scanned={scanned:,} kept={kept:,} bad={bad:,} ({rate:,.0f} lines/sec)")

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue

            asin = obj.get("asin")
            if asin not in eligible_asins:
                continue

            if "overall" not in obj or "unixReviewTime" not in obj:
                bad += 1
                continue

            try:
                star = float(obj["overall"])
                ts = int(obj["unixReviewTime"])
            except Exception:
                bad += 1
                continue

            wk = parse_unix_time_to_week(ts)
            b = star_to_bin(star)

            cell = per_asin_week[asin][wk]
            cell.n += 1
            cell.counts[b] += 1
            kept += 1

    print("[DONE] Review scan complete")
    print(f"  scanned: {scanned:,}")
    print(f"  kept  : {kept:,}")
    print(f"  bad   : {bad:,}")
    print(f"  elapsed: {fmt_secs(now_s()-t0)}")

    return per_asin_week


def extract_valid_week_rows(
    per_asin_week: Dict[str, Dict[int, WeekCell]],
    min_weekly_n: int = 30,
) -> Dict[str, List[Tuple[int, int, np.ndarray]]]:
    per = {}
    for asin, wkmap in per_asin_week.items():
        rows = []
        for wk, cell in wkmap.items():
            if cell.n >= min_weekly_n:
                rows.append((wk, cell.n, cell.counts.copy()))
        rows.sort(key=lambda x: x[0])
        if rows:
            per[asin] = rows
    return per


# -----------------------------
# Geometry + baseline
# -----------------------------
def w1_distance_from_counts(counts: np.ndarray, ref_counts: np.ndarray) -> float:
    a = counts.astype(np.float64)
    b = ref_counts.astype(np.float64)
    sa = a.sum()
    sb = b.sum()
    if sa <= 0 or sb <= 0:
        return 0.0
    p = a / sa
    q = b / sb
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.sum(np.abs(cdf_p - cdf_q)))


def estimate_baseline_table_multinomial(
    ref_counts: np.ndarray,
    n_values: Iterable[int],
    mc: int = 200,
    seed: int = 999,
) -> Dict[int, float]:
    rng = np.random.default_rng(seed)
    ref = ref_counts.astype(np.int64)
    s = ref.sum()
    if s <= 0:
        raise RuntimeError("ref_counts sum is zero; cannot build baseline.")
    p0 = (ref / s).astype(np.float64)

    table: Dict[int, float] = {}
    n_values = sorted(set(int(x) for x in n_values if int(x) > 0))
    for n in n_values:
        vals = np.empty(mc, dtype=np.float64)
        for j in range(mc):
            samp = rng.multinomial(n, p0)
            vals[j] = w1_distance_from_counts(samp, ref)
        table[n] = float(np.mean(vals))
    return table


def baseline_lookup_nearest(b_table: Dict[int, float], n: int) -> float:
    if n in b_table:
        return b_table[n]
    keys = np.array(sorted(b_table.keys()), dtype=np.int64)
    idx = int(np.argmin(np.abs(keys - int(n))))
    return float(b_table[int(keys[idx])])


# -----------------------------
# Metrics + selection
# -----------------------------
def quantile(x: List[float], q: float) -> float:
    if not x:
        return float("nan")
    return float(np.quantile(np.array(x, dtype=np.float64), q))

def control_score(dlist: List[float]) -> Tuple[float, float]:
    if not dlist:
        return (float("inf"), float("inf"))
    arr = np.array(dlist, dtype=np.float64)
    s1 = float(np.sum(arr))
    sabs = float(np.sum(np.abs(arr))) + 1e-12
    drift = abs(s1) / sabs
    energy = float(np.quantile(np.abs(arr), 0.90))
    return drift, energy


def pick_exemplars(
    item_rows: List[Tuple],
    asin2title: Dict[str, str],
    top_k: int = 2,
    control_k: int = 2,
    exclude_keywords: Optional[List[str]] = None,
    control_drift_max: float = 0.1,
    control_energy_max: float = 0.20,
) -> List[str]:
    """
    item_rows tuples are:
      (asin, Q, qh, tot, mean_star, drift, energy, canon_title)
    """
    exclude_keywords = exclude_keywords or []

    rows_by_Q_desc = sorted(item_rows, key=lambda x: x[1], reverse=True)
    tops: List[str] = []
    used_canon_top: set = set()

    for row in rows_by_Q_desc:
        asin, Qv, qh, tot, mean_star, drift, energy, canon = row
        title = asin2title.get(asin, "")

        if is_excluded_title(title, exclude_keywords):
            continue

        if not canon:
            canon = f"asin:{asin}"

        if canon in used_canon_top:
            continue

        tops.append(asin)
        used_canon_top.add(canon)

        if len(tops) >= top_k:
            break

    if len(tops) < top_k:
        raise RuntimeError(
            "Could not find enough top exemplars after exclusions/uniqueness. "
            "Relax --exclude or expand eligible set."
        )

    forbidden_canon = set(used_canon_top)
    rows_ctrl_ranked = sorted(item_rows, key=lambda x: (x[6], x[5], x[1]))

    controls: List[str] = []
    used_canon_ctrl: set = set()

    for row in rows_ctrl_ranked:
        asin, Qv, qh, tot, mean_star, drift, energy, canon = row

        if (drift > control_drift_max) and (energy > control_energy_max):
            continue

        if not canon:
            canon = f"asin:{asin}"

        if canon in forbidden_canon:
            continue

        if canon in used_canon_ctrl:
            continue

        controls.append(asin)
        used_canon_ctrl.add(canon)

        if len(controls) >= control_k:
            break

    if len(controls) < control_k:
        raise RuntimeError(
            f"Could not find {control_k} distinct-title controls under "
            f"(drift <= {control_drift_max}) OR (energy <= {control_energy_max}). "
            f"Try increasing --control-energy-max or expanding eligible set."
        )

    return tops + controls


# -----------------------------
# Plotting
# -----------------------------
def plot_fig5_exemplars(
    outdir: str,
    exemplars: List[str],
    per_asin_rows: Dict[str, List[Tuple[int, int, float, float, float]]],
    asin2title: Dict[str, str],
    dpi: int = 300,
):
    all_W1, all_d, all_D = [], [], []
    for asin in exemplars:
        for _, _, W1, d, D in per_asin_rows.get(asin, []):
            all_W1.append(W1)
            all_d.append(d)
            all_D.append(D)

    W1_min, W1_max = float(np.min(all_W1)), float(np.max(all_W1))
    W1_pad = 0.06 * (W1_max - W1_min) if W1_max > W1_min else 0.5
    W1_ylim = (W1_min - W1_pad, W1_max + W1_pad)

    M = float(np.max(np.abs(all_d))) if len(all_d) else 1.0
    if M <= 0:
        M = 1.0
    d_ylim = (-1.08 * M, 1.08 * M)

    D_min, D_max = float(np.min(all_D)), float(np.max(all_D))
    D_pad = 0.06 * (D_max - D_min) if D_max > D_min else 1.0
    D_ylim = (D_min - D_pad, D_max + D_pad)

    ncols = len(exemplars)
    fig, axes = plt.subplots(
        nrows=4,
        ncols=ncols,
        figsize=(15.5, 9.6),
        sharex="col",
        constrained_layout=True,
    )

    fig.suptitle(
        r"Amazon Books exemplars (weekly): $n_t$, raw $W_1$, corrected $d_t$, cumulative $D_t$",
        fontsize=12,
    )

    for j, asin in enumerate(exemplars):
        rows = per_asin_rows.get(asin, [])
        x = np.arange(len(rows))

        n_t = np.array([r[1] for r in rows], dtype=np.int64)
        W1 = np.array([r[2] for r in rows], dtype=np.float64)
        d_t = np.array([r[3] for r in rows], dtype=np.float64)
        D_t = np.array([r[4] for r in rows], dtype=np.float64)

        axes[0, j].bar(x, n_t, edgecolor="none")
        axes[1, j].plot(x, W1)
        axes[2, j].plot(x, d_t)
        axes[3, j].plot(x, D_t)

        axes[1, j].set_ylim(W1_ylim)
        axes[2, j].set_ylim(d_ylim)
        axes[3, j].set_ylim(D_ylim)

        axes[2, j].axhline(0.0, color="black", lw=0.9, alpha=0.65)
        axes[3, j].axhline(0.0, color="black", lw=0.9, alpha=0.65)

        for i in range(4):
            nice_axes(axes[i, j])

        title = short_title(asin2title.get(asin, f"ASIN {asin}"), maxlen=60)
        axes[0, j].set_title(title)

    for j in range(ncols):
        axes[0, j].set_ylabel(r"$n_t$")
        axes[1, j].set_ylabel(r"$W_1$")
        axes[2, j].set_ylabel(r"$d_t$")
        axes[3, j].set_ylabel(r"$D_t$")
        axes[3, j].set_xlabel("valid-week index")

    base = os.path.join(outdir, "fig5_Amazon_exemplars")
    save_png_and_pdf(fig, base, dpi=dpi)
    plt.close(fig)
    print(f"[OUTPUT] {base}.png")
    print(f"[OUTPUT] {base}.pdf")


def plot_fig6_cdf(outdir: str, qh_list: List[float], Q_list: List[float], dpi: int = 300):
    def cdf_xy(vals: List[float]):
        a = np.array(sorted(vals), dtype=np.float64)
        y = np.linspace(0, 1, len(a), endpoint=True)
        return a, y

    x1, y1 = cdf_xy(qh_list)
    x2, y2 = cdf_xy(Q_list)

    fig = plt.figure(figsize=(8.8, 4.8))
    ax = plt.gca()
    ax.plot(x1, y1, lw=3, label=r"$q_{0.9}(d_t)$ across items")
    ax.plot(x2, y2, lw=3, label=r"$Q=q_{0.9}(d_t)-q_{0.1}(d_t)$ across items")
    ax.set_title("Amazon Books cohort CDF (eligible items)")
    ax.set_xlabel("value")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    nice_axes(ax)

    base = os.path.join(outdir, "fig6_Amazon_cdf")
    save_png_and_pdf(fig, base, dpi=dpi)
    plt.close(fig)
    print(f"[OUTPUT] {base}.png")
    print(f"[OUTPUT] {base}.pdf")


def load_for_plot_only(outdir: str):
    weekly_scores_csv = os.path.join(outdir, "amazon_weekly_scores.csv")
    metrics_csv = os.path.join(outdir, "amazon_item_metrics.csv")

    if not os.path.exists(weekly_scores_csv):
        raise RuntimeError(f"[PLOT-ONLY] Missing required file: {weekly_scores_csv}")
    if not os.path.exists(metrics_csv):
        raise RuntimeError(f"[PLOT-ONLY] Missing required file: {metrics_csv}")

    per_asin_rows = defaultdict(list)
    with open(weekly_scores_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            asin = row["asin"]
            per_asin_rows[asin].append(
                (
                    int(row["week"]),
                    int(float(row["n_t"])),
                    float(row["W1"]),
                    float(row["d_t"]),
                    float(row["D_t"]),
                )
            )
    for asin in per_asin_rows:
        per_asin_rows[asin].sort(key=lambda x: x[0])

    metrics_rows = []
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            metrics_rows.append(
                (
                    row["asin"],
                    float(row["Q"]),
                    float(row["qh"]),
                    int(row["total_actions"]),
                    float(row["mean_star"]),
                    float(row["drift_abs_sum_over_T"]),
                    float(row["energy_q0.9_abs_d"]),
                    row["canon_title"],
                )
            )

    return per_asin_rows, metrics_rows


# ============================================================
# Optional oracle export helpers
# ============================================================
def emit_oracle_run_for_asin(
    out_root: str,
    asin: str,
    title: str,
    per_week: Dict[int, "WeekCell"],
    *,
    bin_edges: Optional[List[float]] = None,
    schema_version: str = "amazon_books_oracle_v1",
) -> str:
    os.makedirs(out_root, exist_ok=True)
    run_dir = os.path.join(out_root, asin)
    os.makedirs(run_dir, exist_ok=True)

    if bin_edges is None:
        bin_edges = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]

    week_ids = sorted(per_week.keys())
    if not week_ids:
        raise ValueError(f"ASIN {asin} has no valid weeks to export.")

    intervals_path = os.path.join(run_dir, "intervals.csv")
    with open(intervals_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "h_bins", "counts_json"])
        for j, wid in enumerate(week_ids, start=1):
            cell = per_week[wid]
            counts = cell.counts.astype(int).tolist()
            w.writerow([j, 5, json.dumps(counts)])

    meta = {
        "schema_version": schema_version,
        "config_id": f"amazon_books::{asin}",
        "dataset": "amazon_books",
        "asin": asin,
        "title": title,
        "bin_edges": bin_edges,
        "week_ids": week_ids,
        "num_intervals": len(week_ids),
    }
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return run_dir


def emit_oracle_runs(
    out_root: str,
    asins: List[str],
    asin2title: Dict[str, str],
    per_asin_week: Dict[str, Dict[int, "WeekCell"]],
) -> List[str]:
    run_dirs: List[str] = []
    for asin in asins:
        if asin not in per_asin_week:
            print(f"[WARN] ASIN {asin} not found in per_asin_week; skipping oracle export.")
            continue
        title = asin2title.get(asin, "")
        run_dir = emit_oracle_run_for_asin(out_root, asin, title, per_asin_week[asin])
        run_dirs.append(run_dir)
    return run_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", help="Books.json.gz (required unless --plot-only)")
    ap.add_argument("--eligible", required=True, help="eligible_asins.txt")
    ap.add_argument("--meta", required=True, help="usable_meta_books.csv (asin,title)")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--min-weekly-n", type=int, default=30)
    ap.add_argument("--baseline-weeks", type=int, default=8, help="first K valid weeks per ASIN contribute to global H0")
    ap.add_argument("--baseline-mc", type=int, default=200)
    ap.add_argument("--baseline-seed", type=int, default=999)
    ap.add_argument("--qhi", type=float, default=0.90)
    ap.add_argument("--qlo", type=float, default=0.10)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--controls", type=int, default=2)
    ap.add_argument("--control-drift-max", type=float, default=0.1)
    ap.add_argument("--progress-every", type=int, default=1_000_000)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-tex", action="store_true", help="Disable LaTeX text rendering (if TeX not installed).")
    ap.add_argument("--plot-only", action="store_true", help="Only plot from existing CSVs in --outdir.")

    ap.add_argument(
        "--mode",
        default="oracle",
        choices=["legacy", "oracle", "oracle+evidence"],
        help="legacy: original appendix pipeline only; oracle: also emit oracle meta.json/intervals.csv per exemplar; "
             "oracle+evidence: additionally run evidence_and_baselines.py on each emitted run directory.",
    )
    ap.add_argument(
        "--oracle-outdir",
        default=None,
        help="Root directory to write oracle runs (default: <outdir>/oracle_runs).",
    )
    ap.add_argument(
        "--evidence-script",
        default="evidence_and_baselines.py",
        help="Path to evidence_and_baselines.py (used when --mode oracle+evidence).",
    )

    ap.add_argument(
        "--exclude",
        default="harry potter,rowling,j.k. rowling,game of thrones,george r r martin,lotr,lord of the rings,tolkien,hobbit,chronicles of narnia,c.s. lewis,hunger games,twilight,stephen king,dan brown,da vinci code",
        help="comma-separated keywords to exclude from top exemplars by title",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    setup_pub_rcparams(use_tex=(not args.no_tex))
    exclude_keywords = [x.strip() for x in (args.exclude or "").split(",") if x.strip()]

    eligible = read_asin_set(args.eligible)
    asin2title = read_meta_titles(args.meta)

    # -----------------------------
    # PLOT-ONLY MODE (fast)
    # -----------------------------
    if args.plot_only:
        if args.mode != "legacy":
            print("[WARN] --plot-only implies no oracle export; forcing --mode legacy for this run.")
            args.mode = "legacy"

        print("[MODE] Plot-only mode enabled (skipping scan/MC/metrics computation)")
        per_asin_rows, metrics_rows = load_for_plot_only(args.outdir)

        exemplars = pick_exemplars(
            item_rows=metrics_rows,
            asin2title=asin2title,
            top_k=args.top_k,
            control_k=args.controls,
            exclude_keywords=exclude_keywords,
            control_drift_max=args.control_drift_max,
        )

        print("[INFO] Selected exemplars for Fig 5 (plot-only):")
        for a in exemplars:
            print(" ", a, "|", asin2title.get(a, ""))

        plot_fig5_exemplars(
            outdir=args.outdir,
            exemplars=exemplars,
            per_asin_rows=per_asin_rows,
            asin2title=asin2title,
            dpi=args.dpi,
        )

        qh_list = [row[2] for row in metrics_rows if not math.isnan(row[2])]
        Q_list = [row[1] for row in metrics_rows if not math.isnan(row[1])]
        plot_fig6_cdf(args.outdir, qh_list=qh_list, Q_list=Q_list, dpi=args.dpi)

        print("[DONE] Plot-only pipeline complete.")
        if not args.no_tex:
            print("[NOTE] If LaTeX errors, rerun with --no-tex (or install TeX).")
        return

    # -----------------------------
    # FULL MODE (end-to-end)
    # -----------------------------
    if not args.reviews:
        raise RuntimeError("Missing --reviews. Provide Books.json.gz, or run with --plot-only.")

    per_asin_week = scan_reviews_build_weekly_hist(
        args.reviews,
        eligible_asins=eligible,
        progress_every=args.progress_every,
        min_weekly_n=args.min_weekly_n,
    )

    per_valid = extract_valid_week_rows(per_asin_week, min_weekly_n=args.min_weekly_n)
    valid_asins = sorted(per_valid.keys())
    print(f"[INFO] ASINs with >=1 valid week: {len(valid_asins)}")

    H0 = np.zeros(5, dtype=np.int64)
    n_values: List[int] = []
    kept_week_cells = 0

    for asin in valid_asins:
        rows = per_valid[asin]
        take = rows[: args.baseline_weeks]
        for wk, n_t, counts in take:
            H0 += counts
            n_values.append(int(n_t))
            kept_week_cells += 1

    if H0.sum() <= 0:
        raise RuntimeError("Global H0 has zero mass. Check inputs / filters.")

    print("[INFO] Built global H0 from baseline weeks")
    print(f"[INFO] baseline weeks per ASIN: first {args.baseline_weeks} valid weeks")
    print(f"[INFO] baseline week-cells used: {kept_week_cells:,}")
    print(f"[INFO] H0 total actions: {int(H0.sum()):,}")
    print(f"[INFO] Unique n_t values (baseline): {len(set(n_values))}")

    print("[INFO] Estimating baseline correction table b(n) using multinomial MC")
    b_table = estimate_baseline_table_multinomial(
        ref_counts=H0,
        n_values=set(n_values),
        mc=args.baseline_mc,
        seed=args.baseline_seed,
    )
    print(f"[INFO] Built b(n) for {len(b_table)} n-values")

    weekly_hist_csv = os.path.join(args.outdir, "amazon_weekly_hist.csv")
    weekly_scores_csv = os.path.join(args.outdir, "amazon_weekly_scores.csv")

    per_asin_rows: Dict[str, List[Tuple[int, int, float, float, float]]] = {}
    item_d_values: Dict[str, List[float]] = defaultdict(list)
    item_totals: Dict[str, int] = {}
    item_mean_star: Dict[str, float] = {}

    stars = np.array([1, 2, 3, 4, 5], dtype=np.float64)

    with open(weekly_hist_csv, "w", newline="", encoding="utf-8") as f_hist, \
         open(weekly_scores_csv, "w", newline="", encoding="utf-8") as f_sc:

        wh = csv.writer(f_hist)
        ws = csv.writer(f_sc)

        wh.writerow(["asin", "week", "n_t", "c1", "c2", "c3", "c4", "c5"])
        ws.writerow(["asin", "week", "n_t", "W1", "d_t", "D_t"])

        for asin in valid_asins:
            rows = per_valid[asin]
            D = 0.0
            out_rows: List[Tuple[int, int, float, float, float]] = []
            tot_actions = 0
            sum_star = 0.0

            for wk, n_t, counts in rows:
                n_t = int(n_t)
                tot_actions += n_t
                sum_star += float(np.dot(counts.astype(np.float64), stars))

                W1 = w1_distance_from_counts(counts, H0)
                bn = baseline_lookup_nearest(b_table, n_t)

                d = float(W1 - bn)
                D += d

                wh.writerow([asin, wk, n_t, int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3]), int(counts[4])])
                ws.writerow([asin, wk, n_t, f"{W1:.8f}", f"{d:.8f}", f"{D:.8f}"])

                out_rows.append((wk, n_t, float(W1), float(d), float(D)))
                item_d_values[asin].append(float(d))

            per_asin_rows[asin] = out_rows
            item_totals[asin] = int(tot_actions)
            item_mean_star[asin] = (sum_star / tot_actions) if tot_actions > 0 else float("nan")

    print(f"[OUTPUT] {weekly_hist_csv}")
    print(f"[OUTPUT] {weekly_scores_csv}")

    metrics_csv = os.path.join(args.outdir, "amazon_item_metrics.csv")
    metrics_rows = []

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "asin", "title", "canon_title",
            "qh", "ql", "Q",
            "total_actions", "mean_star",
            "drift_abs_sum_over_T", "energy_q0.9_abs_d"
        ])

        for asin in valid_asins:
            dlist = item_d_values.get(asin, [])
            qh = quantile(dlist, args.qhi)
            ql = quantile(dlist, args.qlo)
            Qv = float(qh - ql)

            drift, energy = control_score(dlist)
            tot = int(item_totals.get(asin, 0))
            mean_star = float(item_mean_star.get(asin, float("nan")))

            title = (asin2title.get(asin, "") or "").strip()
            canon = canonicalize_title(title) or f"asin:{asin}"

            w.writerow([
                asin,
                title,
                canon,
                f"{qh:.8f}", f"{ql:.8f}", f"{Qv:.8f}",
                tot,
                f"{mean_star:.6f}",
                f"{drift:.8f}",
                f"{energy:.8f}",
            ])

            metrics_rows.append((asin, Qv, qh, tot, mean_star, drift, energy, canon))

    print(f"[OUTPUT] {metrics_csv}")

    exemplars = pick_exemplars(
        item_rows=metrics_rows,
        asin2title=asin2title,
        top_k=args.top_k,
        control_k=args.controls,
        exclude_keywords=exclude_keywords,
        control_drift_max=args.control_drift_max,
    )
    print("[INFO] Selected exemplars for Fig 5:")
    for a in exemplars:
        print(" ", a, "|", asin2title.get(a, ""))

    # ------------------------------------------------------------
    # Oracle export (new oracle pipeline input): meta.json + intervals.csv per exemplar
    # ------------------------------------------------------------
    if args.mode != "legacy":
        if args.oracle_outdir is None:
            args.oracle_outdir = os.path.join(args.outdir, "oracle_runs")

        if args.mode in ("oracle", "oracle+evidence"):
            print(f"[INFO] Emitting oracle runs to: {args.oracle_outdir}")
            run_dirs = emit_oracle_runs(
                out_root=args.oracle_outdir,
                asins=exemplars,
                asin2title=asin2title,
                per_asin_week=per_asin_week,
            )
            print(f"[INFO] Oracle runs created: {len(run_dirs)}")

            if args.mode == "oracle+evidence":
                for rd in run_dirs:
                    cmd = f'python "{args.evidence_script}" --input_dir "{rd}"'
                    print("[INFO] Running evidence:", cmd)
                    rc = os.system(cmd)
                    if rc != 0:
                        print(f"[WARN] evidence script failed for {rd} (rc={rc})")

    plot_fig5_exemplars(
        outdir=args.outdir,
        exemplars=exemplars,
        per_asin_rows=per_asin_rows,
        asin2title=asin2title,
        dpi=args.dpi,
    )

    qh_list = [row[2] for row in metrics_rows if not math.isnan(row[2])]
    Q_list = [row[1] for row in metrics_rows if not math.isnan(row[1])]
    plot_fig6_cdf(args.outdir, qh_list=qh_list, Q_list=Q_list, dpi=args.dpi)

    print("[DONE] Amazon Books appendix pipeline complete.")
    if not args.no_tex:
        print("[NOTE] If LaTeX errors, rerun with --no-tex (or install TeX).")


if __name__ == "__main__":
    main()
