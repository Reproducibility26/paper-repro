#!/usr/bin/env python3
"""
fig16.py — Figure 16: Diagnostic crossover between raw geometry and signed increments.

Purpose:
  This diagnostic illustrates how raw geometric distances and baseline-corrected
  signed increments behave differently under platform variance mismatch.
  It provides intuition for the effect of baseline correction rather than serving
  as a full attribution experiment.

Curves shown (as functions of the platform variance factor γ_plat):
  Raw geometry:
    - W_n(γ):  W1(normal, platform)
    - W_c(γ):  W1(coalition, platform)
    - Δ_W(γ) = W_n(γ) − W_c(γ)
  Signed increments (fixed action count n_diag):
    - E_n(γ):  E[d_t | normal]
    - E_c(γ):  E[d_t | coalition]
    - Δ_d(γ) = E_c(γ) − E_n(γ)

Here d_t = W1(H_t, H_ref) − b_W1(n_diag), where the baseline b_W1 is calibrated
under the platform reference distribution.

Experimental setup:
  - Means are matched (μ = 0.5); only variances differ.
  - Platform variance is swept as Var_plat = γ_plat · Var_coal.
  - Raw curves use large-n Monte Carlo histograms.
  - Signed-increment curves use a fixed diagnostic sample size n_diag.

Modes:
  - Full run (default): recompute curves, write cache CSV, then plot.
  - Plot-only (--plot_only): load cached CSV and regenerate the figure.

Outputs (unchanged):
  - data/fig16_curves.csv
  - figure/fig16_diag_merged_w1_dt.(png|pdf)

Usage:
  python fig16.py
  python fig16.py --plot_only

"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 13,
})

# ------------------------------------------------------------
# Path setup: script in Figure16, modules in parent dir (same as fig5.py)
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if A_DIR not in sys.path:
    sys.path.insert(0, A_DIR)

from evidence_and_baselines import (
    EvidenceContext,
    build_ref_probs_from_spec,
    mean_var_from_hist_probs,
    w1_hist,
    Ours_W1,
)

# ------------------------------------------------------------
# Logging (fig5.py style)
# ------------------------------------------------------------
def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)

def step(title: str) -> None:
    bar = "-" * 78
    print(f"\n{bar}\n[{_ts()}] {title}\n{bar}", flush=True)

def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


# ------------------------------------------------------------
# Config (override via CLI)
# ------------------------------------------------------------
CFG: Dict[str, Any] = dict(
    figure="fig16_diag_merged_w1_dt",

    # output layout (like fig5.py)
    data_subdir="data",
    fig_subdir="figure",
    cache_csv_name="fig16_curves.csv",
    out_png_name="fig16_diag_merged_w1_dt.png",
    out_pdf_name="fig16_diag_merged_w1_dt.pdf",

    # histogram support
    bins=80,
    x_min=0.0,
    x_max=1.0,

    # mean matched
    mu=0.50,

    # population variances in the diagnostic
    var_norm=0.05,
    var_coal=0.02,

    # platform variance sweep: Var_plat = gamma_plat * Var_coal
    gamma_plat_min=1.50,
    gamma_plat_max=2.50,
    gamma_plat_step=0.05,

    # raw W1 curves: MC over large-n histograms
    raw_n=200_000,
    raw_mc=200,

    # signed increments: fixed action count n_diag
    n_diag=800,
    dt_mc=400,

    # baseline b(n_diag): MC reps for baseline_table_mc
    baseline_mc=400,

    # reference histogram (platform) MC sample size
    ref_sample_size=300_000,

    # RNG
    seed=20260101,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _bin_edges() -> np.ndarray:
    return np.linspace(float(CFG["x_min"]), float(CFG["x_max"]), int(CFG["bins"]) + 1, dtype=np.float64)

def _sample_clipped_normal(
    mu: float, sigma: float, n: int, rng: np.random.Generator, x_min: float, x_max: float
) -> np.ndarray:
    x = rng.normal(loc=float(mu), scale=float(sigma), size=int(n))
    return np.clip(x, float(x_min), float(x_max))

def _counts_probs_from_samples(x: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    counts, _ = np.histogram(x, bins=edges)
    counts = counts.astype(np.int64)
    n = int(np.sum(counts))
    probs = counts.astype(np.float64) / float(n) if n > 0 else np.zeros_like(counts, dtype=np.float64)
    return counts, probs, n

def _w1_mc(
    mu: float, sigma: float, ref_probs: np.ndarray, edges: np.ndarray,
    n: int, mc: int, seed: int
) -> float:
    rng = np.random.default_rng(int(seed))
    vals = np.zeros(int(mc), dtype=np.float64)
    for i in range(int(mc)):
        x = _sample_clipped_normal(mu, sigma, int(n), rng, CFG["x_min"], CFG["x_max"])
        _, probs, _ = _counts_probs_from_samples(x, edges)
        vals[i] = w1_hist(probs, ref_probs, edges)
    return float(np.mean(vals))

def _dt_expectation_mc(
    mu: float, sigma: float, ref_probs: np.ndarray, edges: np.ndarray,
    b_n: float, n_diag: int, mc: int, seed: int
) -> float:
    rng = np.random.default_rng(int(seed))
    vals = np.zeros(int(mc), dtype=np.float64)
    for i in range(int(mc)):
        x = _sample_clipped_normal(mu, sigma, int(n_diag), rng, CFG["x_min"], CFG["x_max"])
        _, probs, _ = _counts_probs_from_samples(x, edges)
        vals[i] = w1_hist(probs, ref_probs, edges) - float(b_n)
    return float(np.mean(vals))

def _zero_cross(x: np.ndarray, y: np.ndarray) -> float | None:
    s = np.sign(y)
    idx = np.where(s[:-1] * s[1:] <= 0)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    x0, x1 = float(x[i]), float(x[i + 1])
    y0, y1 = float(y[i]), float(y[i + 1])
    if abs(y1 - y0) < 1e-12:
        return x0
    return x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)


# ------------------------------------------------------------
# I/O: cache CSV
# ------------------------------------------------------------
def write_curves_csv(path: str, gammas: np.ndarray,
                     Wn: np.ndarray, Wc: np.ndarray, dW: np.ndarray,
                     En: np.ndarray, Ec: np.ndarray, dd: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gamma", "Wn", "Wc", "dW", "En", "Ec", "dd"])
        for i in range(len(gammas)):
            w.writerow([
                f"{float(gammas[i]):.10g}",
                f"{float(Wn[i]):.10g}",
                f"{float(Wc[i]):.10g}",
                f"{float(dW[i]):.10g}",
                f"{float(En[i]):.10g}",
                f"{float(Ec[i]):.10g}",
                f"{float(dd[i]):.10g}",
            ])
    log(f"[ok] wrote cache CSV -> {path}")

def read_curves_csv(path: str) -> Dict[str, np.ndarray]:
    _require(path)
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        raise ValueError(f"Cache CSV is empty: {path}")

    def col(name: str) -> np.ndarray:
        return np.asarray([float(r0[name]) for r0 in rows], dtype=np.float64)

    return {
        "gamma": col("gamma"),
        "Wn": col("Wn"),
        "Wc": col("Wc"),
        "dW": col("dW"),
        "En": col("En"),
        "Ec": col("Ec"),
        "dd": col("dd"),
    }


# ------------------------------------------------------------
# Step 1: compute curves (full run)
# ------------------------------------------------------------
def compute_curves() -> Dict[str, np.ndarray]:
    step("FIG16 — STEP 1/3: COMPUTE CURVES (MC)")

    edges = _bin_edges()
    mu = float(CFG["mu"])
    sigma_norm = math.sqrt(float(CFG["var_norm"]))
    sigma_coal = math.sqrt(float(CFG["var_coal"]))

    gmin = float(CFG["gamma_plat_min"])
    gmax = float(CFG["gamma_plat_max"])
    gstep = float(CFG["gamma_plat_step"])
    gammas = np.arange(gmin, gmax + 1e-12, gstep, dtype=np.float64)

    Wn = np.zeros_like(gammas)
    Wc = np.zeros_like(gammas)
    dW = np.zeros_like(gammas)
    En = np.zeros_like(gammas)
    Ec = np.zeros_like(gammas)
    dd = np.zeros_like(gammas)

    raw_n = int(CFG["raw_n"])
    raw_mc = int(CFG["raw_mc"])
    n_diag = int(CFG["n_diag"])
    dt_mc = int(CFG["dt_mc"])
    baseline_mc = int(CFG["baseline_mc"])
    ref_sample_size = int(CFG["ref_sample_size"])
    seed0 = int(CFG["seed"])

    bl = Ours_W1()

    total = len(gammas)
    for j, g in enumerate(gammas, start=1):
        log(f"Progress: {j}/{total}  gamma_plat={float(g):.3f}")

        # Platform variance mismatch: Var_plat = g * Var_coal
        var_plat = float(g) * float(CFG["var_coal"])
        sigma_plat = math.sqrt(max(1e-12, var_plat))

        # (a) Build platform reference histogram at this binning
        H_plat_spec = {"type": "normal", "mu": mu, "sigma": sigma_plat}
        ref_probs = build_ref_probs_from_spec(
            spec=H_plat_spec,
            edges=edges,
            ref_sample_size=ref_sample_size,
            seed=seed0 + 777 + j,
        )

        # ctx for baseline table MC
        ref_mu, ref_var = mean_var_from_hist_probs(ref_probs, edges)
        ctx = EvidenceContext(
            edges=edges,
            ref_probs=ref_probs,
            ref_mu=ref_mu,
            ref_var=ref_var,
            ref_sigma=float(math.sqrt(max(ref_var, 0.0))),
            energy_D=None,
        )

        # (b) Raw W1 curves (large-n histograms)
        Wn[j - 1] = _w1_mc(mu, sigma_norm, ref_probs, edges, n=raw_n, mc=raw_mc, seed=seed0 + 1000 + j)
        Wc[j - 1] = _w1_mc(mu, sigma_coal, ref_probs, edges, n=raw_n, mc=raw_mc, seed=seed0 + 2000 + j)
        dW[j - 1] = Wn[j - 1] - Wc[j - 1]

        # (c) Baseline b(n_diag) under platform null
        # NOTE: baseline_table_mc computes all n=1..n_diag internally (dense prefix reuse).
        btab = bl.baseline_table_mc(
            n_grid=[n_diag],
            ctx=ctx,
            mc_reps=baseline_mc,
            mc_seed=seed0 + 3000 + j,
        )
        b_n = float(btab.get(n_diag, 0.0))

        # (d) Expected signed increments at fixed n_diag
        En[j - 1] = _dt_expectation_mc(mu, sigma_norm, ref_probs, edges, b_n=b_n, n_diag=n_diag,
                                       mc=dt_mc, seed=seed0 + 4000 + j)
        Ec[j - 1] = _dt_expectation_mc(mu, sigma_coal, ref_probs, edges, b_n=b_n, n_diag=n_diag,
                                       mc=dt_mc, seed=seed0 + 5000 + j)
        dd[j - 1] = Ec[j - 1] - En[j - 1]

        log(
            f"  Wn={Wn[j-1]:.6g}  Wc={Wc[j-1]:.6g}  dW={dW[j-1]:+.6g} | "
            f"En={En[j-1]:+.6g}  Ec={Ec[j-1]:+.6g}  dd={dd[j-1]:+.6g}"
        )

    return {"gamma": gammas, "Wn": Wn, "Wc": Wc, "dW": dW, "En": En, "Ec": Ec, "dd": dd}


# ------------------------------------------------------------
# Step 2: plot
# ------------------------------------------------------------
def plot_curves(curves: Dict[str, np.ndarray], out_png: str, out_pdf: str) -> None:
    step("FIG16 — STEP 3/3: PLOT")

    gammas = curves["gamma"]
    Wn = curves["Wn"]
    Wc = curves["Wc"]
    dW = curves["dW"]
    En = curves["En"]
    Ec = curves["Ec"]
    dd = curves["dd"]

    n_diag = int(CFG["n_diag"])
    cross_dW = _zero_cross(gammas, dW)
    cross_dd = _zero_cross(gammas, dd)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    ax.plot(gammas, Wn, marker="o", lw=2.6, label=r"$W_1(\mathrm{norm},\mathrm{plat})$")
    ax.plot(gammas, Wc, marker="o", lw=2.6, label=r"$W_1(\mathrm{coal},\mathrm{plat})$")
    ax.plot(gammas, dW, marker="o", lw=2.6,
            label=r"$\Delta_W(\gamma)=W_1(\mathrm{norm},\mathrm{plat})-W_1(\mathrm{coal},\mathrm{plat})$")

    ax.plot(gammas, En, marker="s", lw=2.6, linestyle="--",
            label=rf"$\mathrm{{E}}[d_t\mid \mathrm{{norm}}]$ (n={n_diag})")
    ax.plot(gammas, Ec, marker="s", lw=2.6, linestyle="--",
            label=rf"$\mathrm{{E}}[d_t\mid \mathrm{{coal}}]$ (n={n_diag})")
    ax.plot(gammas, dd, marker="s", lw=2.6, linestyle="--",
            label=r"$\Delta_d(\gamma)=\mathrm{{E}}[d_t\mid \mathrm{coal}]-\mathrm{{E}}[d_t\mid \mathrm{norm}]$")

    ax.axhline(0.0, lw=1.3, linestyle=":", alpha=0.9)

    y0 = ax.get_ylim()[0]
    if cross_dW is not None:
        ax.axvline(cross_dW, lw=1.3, linestyle=":", alpha=0.9)
        ax.text(cross_dW, y0, f"  $\\Delta W$ flip ~ {cross_dW:.2f}", rotation=90, va="bottom", fontsize=10)
    if cross_dd is not None:
        ax.axvline(cross_dd, lw=1.3, linestyle=":", alpha=0.9)
        ax.text(cross_dd, y0, f"  $\\Delta d$ flip ~ {cross_dd:.2f}", rotation=90, va="bottom", fontsize=10)

    ax.set_xlabel(r"Platform variance factor $\gamma_{\mathrm{plat}}$")
    ax.set_ylabel(r"Distance / signed increment (Wasserstein units)")
    ax.set_xlim(1.48, 2.52)
    ax.grid(alpha=0.15)
    ax.legend(loc="lower right", frameon=True, fontsize=11, ncol=1)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log(f"[plot] saved PNG -> {out_png}")
    log(f"[plot] saved PDF -> {out_pdf}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Fig16 diagnostic (new oracle): full run by default, or --plot_only")
    ap.add_argument("--plot_only", action="store_true", default=False)

    # Overrides (keep minimal; add more if you want)
    ap.add_argument("--bins", type=int, default=int(CFG["bins"]))
    ap.add_argument("--raw_mc", type=int, default=int(CFG["raw_mc"]))
    ap.add_argument("--dt_mc", type=int, default=int(CFG["dt_mc"]))
    ap.add_argument("--baseline_mc", type=int, default=int(CFG["baseline_mc"]))
    ap.add_argument("--ref_sample_size", type=int, default=int(CFG["ref_sample_size"]))
    ap.add_argument("--n_diag", type=int, default=int(CFG["n_diag"]))
    ap.add_argument("--gamma_plat_min", type=float, default=float(CFG["gamma_plat_min"]))
    ap.add_argument("--gamma_plat_max", type=float, default=float(CFG["gamma_plat_max"]))
    ap.add_argument("--gamma_plat_step", type=float, default=float(CFG["gamma_plat_step"]))
    ap.add_argument("--seed", type=int, default=int(CFG["seed"]))

    args = ap.parse_args(argv)

    CFG["bins"] = int(args.bins)
    CFG["raw_mc"] = int(args.raw_mc)
    CFG["dt_mc"] = int(args.dt_mc)
    CFG["baseline_mc"] = int(args.baseline_mc)
    CFG["ref_sample_size"] = int(args.ref_sample_size)
    CFG["n_diag"] = int(args.n_diag)
    CFG["gamma_plat_min"] = float(args.gamma_plat_min)
    CFG["gamma_plat_max"] = float(args.gamma_plat_max)
    CFG["gamma_plat_step"] = float(args.gamma_plat_step)
    CFG["seed"] = int(args.seed)

    data_dir = os.path.join(THIS_DIR, CFG["data_subdir"])
    fig_dir = os.path.join(THIS_DIR, CFG["fig_subdir"])
    _ensure_dir(data_dir)
    _ensure_dir(fig_dir)

    cache_csv = os.path.join(data_dir, CFG["cache_csv_name"])
    out_png = os.path.join(fig_dir, CFG["out_png_name"])
    out_pdf = os.path.join(fig_dir, CFG["out_pdf_name"])

    step("FIG16 START")
    log(f"Mode: {'PLOT-ONLY' if args.plot_only else 'FULL'}")
    log(f"bins={CFG['bins']}  gamma_plat∈[{CFG['gamma_plat_min']},{CFG['gamma_plat_max']}] step={CFG['gamma_plat_step']}")
    log(f"raw_n={CFG['raw_n']} raw_mc={CFG['raw_mc']} | n_diag={CFG['n_diag']} dt_mc={CFG['dt_mc']}")
    log(f"baseline_mc={CFG['baseline_mc']}  ref_sample_size={CFG['ref_sample_size']}")
    log(f"cache_csv={os.path.abspath(cache_csv)}")

    if args.plot_only:
        step("FIG16 — STEP 1/3: LOAD CACHE")
        curves = read_curves_csv(cache_csv)
        plot_curves(curves, out_png, out_pdf)
        step("FIG16 DONE")
        return 0

    # full run
    curves = compute_curves()

    step("FIG16 — STEP 2/3: WRITE CACHE")
    write_curves_csv(cache_csv,
                     curves["gamma"], curves["Wn"], curves["Wc"], curves["dW"],
                     curves["En"], curves["Ec"], curves["dd"])

    plot_curves(curves, out_png, out_pdf)

    step("FIG16 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
