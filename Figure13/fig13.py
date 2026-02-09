#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig13.py — Figure 13: Macro merging and interval resolution effects (Geo-only).

Purpose:
  This figure studies how aggregating micro-intervals into coarser macro intervals
  affects detection performance when using the geometric Wasserstein-1 score.
  No baseline comparisons are shown; the focus is on resolution and aggregation effects.

Design:
  - Generate micro-interval data once per (R_exp, seed).
  - For each macro size m, merge consecutive blocks of m micro intervals:
      * Histogram counts are summed across the block.
      * Action counts are summed across the block.
      * Participants are merged via set union using exposures_by_t.npz.
  - At each macro step j, compute the Geo-W1 score with baseline correction and
    accumulate per-user scores.
  - Evaluate ROC–AUC as a function of global time for each macro size.

Notes:
  - Uses the compact CSR-by-interval exposure format (exposures_by_t.npz).
  - Mean shift is fixed; only temporal aggregation and resolution are varied.
  - Intended to isolate the effect of reporting granularity on attribution accuracy.

Usage:
  python fig13.py
  python fig13.py --plot_only

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple, Set

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
# Path setup: script in Figure13, modules in parent dir
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if A_DIR not in sys.path:
    sys.path.insert(0, A_DIR)


# ------------------------------------------------------------
# Logging (match fig9)
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


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


# ------------------------------------------------------------
# Oracles / utilities
# ------------------------------------------------------------
from rotation_generator import RotationParams, generate_rotation_dataset

from evidence_and_baselines import (
    Ours_W1,
    EvidenceContext,
    build_ref_probs_from_spec,
    mean_var_from_hist_probs,
)


# ------------------------------------------------------------
# Global config
# ------------------------------------------------------------
CFG: Dict[str, Any] = dict(
    figure="fig13_macro_merging",

    # base timeline
    T_base=2000,

    # populations
    N_norm=20000,
    N_coal=2000,

    # behavior
    p_norm=0.04,
    p_on=0.50,
    Rexp_list=[1.0, 2.0],

    # fixed mean shift
    delta_mu=0.06,

    # macro sizes
    merge_factors=[1, 2, 5, 10, 20],

    # histogram domain
    bins=20,
    x_min=0.0,
    x_max=1.0,

    # platform normal reference
    mu_norm=0.50,
    var_norm=0.05,  # sigma^2

    # baseline table b_W1(n) settings
    baseline_ref_sample_size=300_000,
    baseline_mc_reps=500,
    baseline_mc_seed=777,

    # repeats / bands
    seeds=[0, 1, 2, 3, 4],
    qlo=0.05,
    qhi=0.95,
)


def _sigma_norm() -> float:
    return float(np.sqrt(float(CFG["var_norm"])))


def compute_k_on(R_exp: float) -> int:
    """
    Rexp = (p_on * k_on / N_coal) / p_norm  =>  k_on = Rexp * p_norm * N_coal / p_on
    """
    k = int(round(float(R_exp) * float(CFG["p_norm"]) * int(CFG["N_coal"]) / max(float(CFG["p_on"]), 1e-12)))
    return int(np.clip(k, 0, int(CFG["N_coal"])))


# ------------------------------------------------------------
# IO helpers for micro-run files
# ------------------------------------------------------------
def read_intervals_counts(intervals_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    intervals.csv rows contain:
      n_actions, counts_json
    Returns:
      counts_mat: (T,B) int64
      n_actions:  (T,)  int64
    """
    _require(intervals_csv)
    counts_list = []
    n_list = []
    with open(intervals_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            n_list.append(int(row["n_actions"]))
            counts = np.asarray(json.loads(row["counts_json"]), dtype=np.int64)
            counts_list.append(counts)
    counts_mat = np.stack(counts_list, axis=0)
    n_actions = np.asarray(n_list, dtype=np.int64)
    return counts_mat, n_actions


def _load_exposures_npz(exposures_npz: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    exposures_by_t.npz must contain:
      t_ptr: int64 (T+1,)
      u_ids: int32 (nnz,)
      a_ut : float32/float64 (nnz,)
    """
    _require(exposures_npz)
    z = np.load(exposures_npz, allow_pickle=False)
    if "t_ptr" not in z or "u_ids" not in z or "a_ut" not in z:
        raise ValueError("exposures_by_t.npz missing required arrays: t_ptr, u_ids, a_ut")
    t_ptr = np.asarray(z["t_ptr"], dtype=np.int64)
    u_ids = np.asarray(z["u_ids"], dtype=np.int64)
    a_ut = np.asarray(z["a_ut"], dtype=np.float64)
    if t_ptr.ndim != 1 or u_ids.ndim != 1 or a_ut.ndim != 1:
        raise ValueError("exposures_by_t.npz arrays must be 1D")
    if u_ids.shape[0] != a_ut.shape[0]:
        raise ValueError("exposures_by_t.npz: u_ids and a_ut must have same length")
    if t_ptr.shape[0] < 2 or t_ptr[0] != 0 or t_ptr[-1] != u_ids.shape[0]:
        raise ValueError("exposures_by_t.npz: invalid t_ptr offsets")
    if np.any(t_ptr[1:] < t_ptr[:-1]):
        raise ValueError("exposures_by_t.npz: t_ptr must be nondecreasing")
    return t_ptr, u_ids, a_ut


def read_participants_from_exposures(exposures_npz: str, T: int) -> List[np.ndarray]:
    """
    NEW pipeline replacement for read_participants(participation.csv).

    Returns parts[t-1] = sorted unique ids in interval t (ignores weights).
    """
    t_ptr, u_ids, _ = _load_exposures_npz(exposures_npz)
    T_exp = int(t_ptr.shape[0] - 1)
    T_use = min(int(T), T_exp)

    out: List[np.ndarray] = []
    for t in range(1, T_use + 1):
        start = int(t_ptr[t - 1])
        end = int(t_ptr[t])
        if end <= start:
            out.append(np.zeros((0,), dtype=np.int64))
        else:
            uu = u_ids[start:end]
            out.append(np.asarray(sorted(set(int(x) for x in uu.tolist())), dtype=np.int64))

    # If requested T exceeds exposures length, pad with empty.
    for _ in range(T_use + 1, int(T) + 1):
        out.append(np.zeros((0,), dtype=np.int64))
    return out


# ------------------------------------------------------------
# Macro merging
# ------------------------------------------------------------
def merge_macro(
    counts_mat: np.ndarray,     # (T,B)
    n_actions: np.ndarray,      # (T,)
    parts: List[np.ndarray],    # len T
    m: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Merge consecutive blocks of size m:
      counts_macro[j] = sum counts in block
      n_macro[j]      = sum n_actions in block
      parts_macro[j]  = union participants in block
    """
    T, B = counts_mat.shape
    J = T // m
    if J <= 0:
        raise ValueError(f"m={m} too large for T={T}")

    counts_macro = np.zeros((J, B), dtype=np.int64)
    n_macro = np.zeros((J,), dtype=np.int64)
    parts_macro: List[np.ndarray] = []

    for j in range(J):
        a = j * m
        b = (j + 1) * m
        counts_macro[j] = np.sum(counts_mat[a:b], axis=0)
        n_macro[j] = int(np.sum(n_actions[a:b]))

        s: Set[int] = set()
        for t in range(a, b):
            for u in parts[t]:
                s.add(int(u))
        parts_macro.append(np.asarray(sorted(s), dtype=np.int64) if s else np.zeros((0,), dtype=np.int64))

    return counts_macro, n_macro, parts_macro


# ------------------------------------------------------------
# Metric: ROC-AUC (no sklearn)
# ------------------------------------------------------------
def roc_auc(scores: np.ndarray, y01: np.ndarray) -> float:
    """
    Deterministic ROC-AUC with stable tie-breaking by u id.
    """
    u = np.arange(scores.shape[0], dtype=np.int64)
    order = np.lexsort((u, -scores))
    y = y01[order]
    s = scores[order]

    P = float(np.sum(y == 1))
    N = float(np.sum(y == 0))
    if P == 0 or N == 0:
        return 0.5

    tp = np.cumsum(y == 1).astype(np.float64)
    fp = np.cumsum(y == 0).astype(np.float64)

    distinct = np.where(np.diff(s) != 0)[0]
    idx = np.r_[distinct, len(s) - 1]

    tpr = tp[idx] / P
    fpr = fp[idx] / N

    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]

    return float(np.trapz(tpr, fpr))


def probs_from_counts(counts: np.ndarray) -> Tuple[np.ndarray, int]:
    n = int(np.sum(counts))
    if n <= 0:
        return np.zeros_like(counts, dtype=np.float64), 0
    return counts.astype(np.float64) / float(n), n


# ------------------------------------------------------------
# Context + baseline caching
# ------------------------------------------------------------
def build_ctx(bin_edges: np.ndarray) -> EvidenceContext:
    H_plat_spec = {"type": "normal", "mu": float(CFG["mu_norm"]), "sigma": float(_sigma_norm())}

    ref_probs = build_ref_probs_from_spec(
        H_plat_spec,
        bin_edges,
        ref_sample_size=int(CFG["baseline_ref_sample_size"]),
        seed=int(CFG["baseline_mc_seed"]) + 999,
    )
    ref_mu, ref_var = mean_var_from_hist_probs(ref_probs, bin_edges)
    ref_sigma = float(np.sqrt(max(ref_var, 0.0)))

    return EvidenceContext(
        edges=bin_edges,
        ref_probs=ref_probs,
        ref_mu=float(ref_mu),
        ref_var=float(ref_var),
        ref_sigma=float(ref_sigma),
        energy_D=None,
    )


def baseline_dense_cached(
    bl: Ours_W1,
    ctx: EvidenceContext,
    Nmax: int,
    cache_dir: str,
) -> Dict[int, float]:
    """
    Cache dense baseline table b_W1(n) for n=1..Nmax as JSON.
    Keyed only by Nmax because ctx is fixed by CFG + bin_edges for this figure.
    """
    _ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, f"baseline_w1_dense__Nmax_{Nmax}.json")
    if os.path.exists(cache_path):
        obj = _read_json(cache_path)
        return {int(k): float(v) for k, v in obj.items()}

    step(f"Baseline MC (dense) — Nmax={Nmax}")
    t0 = time.time()
    n_grid = list(range(1, Nmax + 1))
    tab = bl.baseline_table_mc(
        n_grid=n_grid,
        ctx=ctx,
        mc_reps=int(CFG["baseline_mc_reps"]),
        mc_seed=int(CFG["baseline_mc_seed"]),
    )
    _write_json(cache_path, {str(k): float(v) for k, v in tab.items()})
    log(f"[baseline] wrote cache -> {cache_path}  ({time.time() - t0:.1f}s)")
    return tab


# ------------------------------------------------------------
# Experiment runner: one micro run, compute macro AUC trajectories
# ------------------------------------------------------------
def run_one_seed(
    seed: int,
    R_exp: float,
    ctx: EvidenceContext,
    bl: Ours_W1,
    run_dir: str,
    baseline_cache_dir: str,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    For a single (R_exp, seed):
      - generate micro dataset in run_dir
      - for each m in merge_factors:
          merge, compute d_j, accumulate S, compute AUC per macro step

    Returns:
      dict m -> (t_global, auc)
    """
    _ensure_dir(run_dir)

    k_on = compute_k_on(R_exp)
    T = int(CFG["T_base"])

    # --- data generation (micro) ---
    params = RotationParams(
        N_norm=int(CFG["N_norm"]),
        N_coal=int(CFG["N_coal"]),
        T=T,
        p_norm=float(CFG["p_norm"]),
        p_on=float(CFG["p_on"]),
        k_on=int(k_on),
        bins=int(CFG["bins"]),
        x_min=float(CFG["x_min"]),
        x_max=float(CFG["x_max"]),
        mu_norm=float(CFG["mu_norm"]),
        sigma_norm=float(_sigma_norm()),
        D_norm={"type": "normal", "mu": float(CFG["mu_norm"]), "sigma": float(_sigma_norm())},
        D_coal={"type": "normal_shift", "delta": float(CFG["delta_mu"])},
        actions_per_participant_norm=1,
        actions_per_participant_coal=1,
        seed=int(seed),
    )
    generate_rotation_dataset(params, out_dir=run_dir)

    meta = _read_json(os.path.join(run_dir, "meta.json"))
    y01 = np.asarray(meta["arrays"]["is_coal_member"], dtype=np.int8)
    N_total = int(len(y01))

    counts_micro, n_micro = read_intervals_counts(os.path.join(run_dir, "intervals.csv"))

    # NEW pipeline: participants from exposures_by_t.npz
    exp_npz = os.path.join(run_dir, "exposures_by_t.npz")
    _require(exp_npz)
    parts_micro = read_participants_from_exposures(exp_npz, T=T)

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    merge_factors: List[int] = list(CFG["merge_factors"])

    for m in merge_factors:
        log(f"[macro] m={m} merge+score")
        counts_macro, n_macro, parts_macro = merge_macro(counts_micro, n_micro, parts_micro, m=m)
        J = counts_macro.shape[0]
        t_global = np.asarray([(j + 1) * m for j in range(J)], dtype=np.int64)

        Nmax = int(np.max(n_macro)) if int(np.max(n_macro)) > 0 else 1
        btab = baseline_dense_cached(bl, ctx, Nmax=Nmax, cache_dir=baseline_cache_dir)

        S = np.zeros((N_total,), dtype=np.float64)
        auc = np.zeros((J,), dtype=np.float64)

        for j in range(J):
            probs, n_j = probs_from_counts(counts_macro[j])
            if n_j <= 0:
                d_j = 0.0
            else:
                X_j = float(bl.score_from_interval(counts_macro[j], probs, n_j, ctx))
                d_j = float(X_j - btab.get(n_j, 0.0))

            ids = parts_macro[j]
            if ids.size > 0:
                S[ids] += d_j

            auc[j] = roc_auc(S, y01)

        out[m] = (t_global, auc)

    return out


# ------------------------------------------------------------
# CSV + plotting
# ------------------------------------------------------------
def save_results_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to save.")
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_results_csv(path: str) -> List[Dict[str, str]]:
    _require(path)
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_from_rows(rows: List[Dict[str, str]], out_png: str, out_pdf: str) -> None:
    merge_factors = list(CFG["merge_factors"])
    qlo = float(CFG["qlo"])
    qhi = float(CFG["qhi"])
    T_base = int(CFG["T_base"])

    Rvals = sorted({float(r["R_exp"]) for r in rows})
    if len(Rvals) != 2:
        raise ValueError("Expected exactly 2 panels (two R_exp values).")

    step("PLOT — Fig13 macro merging")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)

    for ax, R_exp in zip(axes, Rvals):
        for m in merge_factors:
            t_list = sorted({
                int(float(r["t_global"]))
                for r in rows
                if float(r["R_exp"]) == R_exp and int(r["merge_m"]) == m
            })
            if not t_list:
                continue

            med, lo, hi = [], [], []
            for t in t_list:
                aucs = [
                    float(r["auc"])
                    for r in rows
                    if float(r["R_exp"]) == R_exp
                    and int(r["merge_m"]) == m
                    and int(float(r["t_global"])) == t
                ]
                a = np.asarray(aucs, dtype=np.float64)
                med.append(float(np.median(a)))
                lo.append(float(np.quantile(a, qlo)))
                hi.append(float(np.quantile(a, qhi)))

            t_arr = np.asarray(t_list, dtype=np.float64)
            med = np.asarray(med, dtype=np.float64)
            lo = np.asarray(lo, dtype=np.float64)
            hi = np.asarray(hi, dtype=np.float64)

            ax.plot(t_arr, med, lw=2.2, label=rf"$m={m}$")
            ax.fill_between(t_arr, lo, hi, alpha=0.12)

        ax.axhline(0.8, lw=1.2, linestyle="--", alpha=0.9)
        ax.axhline(0.9, lw=1.2, linestyle="--", alpha=0.9)
        ax.grid(alpha=0.15)
        ax.set_xlim(0, T_base)
        ax.set_ylim(0.48, 1.01)
        ax.set_xlabel(r"Global time $t$ (base intervals)")
        ax.set_title(rf"$R_{{\mathrm{{exp}}}}={R_exp:.1f}$, fixed $\Delta\mu={float(CFG['delta_mu']):.2f}$")

    axes[0].set_ylabel(r"ROC--AUC of divergence-based attribution")
    axes[1].legend(loc="lower right", frameon=True, ncol=1, title="Macro size")

    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    log(f"[plot] wrote: {out_png}")
    log(f"[plot] wrote: {out_pdf}")


# ------------------------------------------------------------
# Config + manifest
# ------------------------------------------------------------
def write_config_json(path: str, data_paths: Dict[str, Any]) -> None:
    cfg = dict(CFG)
    cfg["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cfg["script"] = os.path.basename(__file__)
    cfg["paths"] = data_paths
    cfg["notes"] = (
        "Fig13 macro merging. Geo-only: d_j = W1(H~_j, H_plat) - b_W1(n~_j), "
        "S[u]+=d_j for u in union participants; report ROC-AUC(S,y) vs global time. "
        "UPDATED: participants from exposures_by_t.npz."
    )
    _write_json(path, cfg)
    log(f"[config] wrote -> {path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Fig13: macro merging / interval resolution (Geo-only)")
    ap.add_argument("--plot_only", action="store_true", default=False)

    ap.add_argument("--T_base", type=int, default=int(CFG["T_base"]))
    ap.add_argument("--baseline_mc_reps", type=int, default=int(CFG["baseline_mc_reps"]))
    args = ap.parse_args(argv)

    CFG["T_base"] = int(args.T_base)
    CFG["baseline_mc_reps"] = int(args.baseline_mc_reps)

    data_dir = os.path.join(THIS_DIR, "data")
    fig_dir = os.path.join(THIS_DIR, "figure")
    runs_root = os.path.join(data_dir, "runs")
    baseline_cache_dir = os.path.join(data_dir, "baseline_cache", "fig13_cache")

    _ensure_dir(data_dir)
    _ensure_dir(fig_dir)
    _ensure_dir(runs_root)
    _ensure_dir(baseline_cache_dir)

    results_csv = os.path.join(data_dir, "fig13_results_macro_merging.csv")
    out_png = os.path.join(fig_dir, "fig13_macro_merging.png")
    out_pdf = os.path.join(fig_dir, "fig13_macro_merging.pdf")

    step("FIG13 START")
    log(f"Mode: {'PLOT-ONLY' if args.plot_only else 'FULL'}")
    log(f"T_base={CFG['T_base']}  merge_factors={CFG['merge_factors']}")
    log(f"Rexp_list={CFG['Rexp_list']}  seeds={CFG['seeds']}")
    log(f"baseline_mc_reps={CFG['baseline_mc_reps']}  baseline_ref_sample_size={CFG['baseline_ref_sample_size']}")
    log(f"runs_root={os.path.abspath(runs_root)}")
    log(f"baseline_cache_dir={os.path.abspath(baseline_cache_dir)}")

    if not args.plot_only:
        bin_edges = np.linspace(float(CFG["x_min"]), float(CFG["x_max"]), int(CFG["bins"]) + 1, dtype=np.float64)
        ctx = build_ctx(bin_edges)
        bl = Ours_W1()

        rows_out: List[Dict[str, Any]] = []
        Rexp_list: List[float] = list(CFG["Rexp_list"])
        seeds: List[int] = list(CFG["seeds"])

        for R_exp in Rexp_list:
            step(f"R_exp = {R_exp:g}")
            for seed in seeds:
                t0 = time.time()
                run_dir = os.path.join(runs_root, f"R{R_exp:g}".replace(".", "p"), f"seed_{seed}")
                log(f"[run] R_exp={R_exp:g} seed={seed} -> {run_dir}")

                res = run_one_seed(
                    seed=int(seed),
                    R_exp=float(R_exp),
                    ctx=ctx,
                    bl=bl,
                    run_dir=run_dir,
                    baseline_cache_dir=baseline_cache_dir,
                )

                for m, (t_global, auc) in res.items():
                    for tg, a in zip(t_global.tolist(), auc.tolist()):
                        rows_out.append(
                            dict(
                                R_exp=float(R_exp),
                                seed=int(seed),
                                merge_m=int(m),
                                t_global=int(tg),
                                auc=float(a),
                            )
                        )

                log(f"[ok] R_exp={R_exp:g} seed={seed} done in {time.time() - t0:.1f}s")

        save_results_csv(results_csv, rows_out)
        log(f"[cache] wrote results -> {results_csv}")

        cfg_json = os.path.join(data_dir, "fig13_config.json")
        write_config_json(
            cfg_json,
            data_paths=dict(
                results_csv=os.path.abspath(results_csv),
                runs_root=os.path.abspath(runs_root),
                baseline_cache_dir=os.path.abspath(baseline_cache_dir),
                out_png=os.path.abspath(out_png),
                out_pdf=os.path.abspath(out_pdf),
            ),
        )

    rows = load_results_csv(results_csv)
    plot_from_rows(rows, out_png=out_png, out_pdf=out_pdf)

    manifest = dict(
        CFG=dict(CFG),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        outputs=dict(
            results_csv=os.path.abspath(results_csv),
            out_png=os.path.abspath(out_png),
            out_pdf=os.path.abspath(out_pdf),
        ),
        runs_root=os.path.abspath(runs_root),
        baseline_cache_dir=os.path.abspath(baseline_cache_dir),
    )
    _write_json(os.path.join(fig_dir, "fig13_manifest.json"), manifest)

    step("FIG13 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
