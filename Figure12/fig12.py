#!/usr/bin/env python3
"""
fig12.py — Figure 12: Detection delay heatmaps $t_{\tau}$ over $(\Delta\mu, p_{\mathrm{on}})$ (Geo-only).

Purpose:
  This figure visualizes detection delay as a function of coalition strength
  (mean shift $\Delta\mu$) and intermittency $p_{\mathrm{on}}$, using the
  geometric Wasserstein-1 score only.

Definition:
  $t_{\tau}$ is the earliest evaluation time $t$ such that
  $\mathrm{AUC}(S^{(t)}) \ge \tau$, where
  $S^{(t)}(u) = \sum_{s \le t} a_{u,s} \, d_s$
  and $d_s = w_{\text{ours\_w1}}(s)$ is the baseline-corrected increment.

Design:
  - Two panels corresponding to different exposure ratios $R_{\exp}$.
  - For each $(\Delta\mu, p_{\mathrm{on}})$ grid point, repeat over multiple seeds.
  - Baseline tables are warmed once per $R_{\exp}$ to ensure full $b(n,h)$ coverage.

Outputs (relative to this file):
  - Experiment CSV:   ./data/fig12_delay_heatmap.csv
  - Config / manifest: ./data/fig12_config.json, ./figure/fig12_manifest.json
  - Figures:          ./figure/fig12_delay_heatmap.{png,pdf}

Usage:
  # Full run
  python fig12.py

  # Plot-only mode
  python fig12.py --plot_only

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

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
# Path setup: script in Figure11, modules in parent dir
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if A_DIR not in sys.path:
    sys.path.insert(0, A_DIR)


# ------------------------------------------------------------
# Logging / IO
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
# Default config (override via CLI)
# ------------------------------------------------------------
CFG: Dict[str, Any] = dict(
    # horizon / evaluation
    T_MAX=2000,
    EVAL_STRIDE=10,
    TAU=0.90,

    # grid
    dmu_min=0.00,
    dmu_max=0.06,
    dmu_points=13,           # e.g., 0.00..0.06
    pon_min=0.05,
    pon_max=1.00,
    pon_points=20,           # e.g., 0.05..1.00

    # panels
    Rexp_values=[1.0, 2.0],

    # population / behavior
    N_norm=20000,
    N_coal=2000,
    p_norm=0.04,

    # signal model
    mu_norm=0.50,
    var_norm=0.05,

    # histogram support
    bins=50,
    x_min=0.0,
    x_max=1.0,

    # seeds / repeats
    runs=10,
    seed_base=20260201,

    # baseline calibration (evidence engine)
    mc_reps=250,
    ref_sample_size=200_000,
    cache_tables=True,
    baseline_mc_seed=999,
)


METHOD = "ours_w1"   # Geo only


# ------------------------------------------------------------
# AUC helper (tie-safe rank statistic; matches old script)
# ------------------------------------------------------------
def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC via Mann–Whitney U / rank statistic (tie-safe)."""
    labels = labels.astype(int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

    s_sorted = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = 0.5 * ((i + 1) + (j + 1))
            ranks[order[i:j + 1]] = avg
        i = j + 1

    sum_ranks_pos = ranks[labels == 1].sum()
    u_pos = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


# ------------------------------------------------------------
# Load series / stream exposures NPZ to compute S^(t)
# ------------------------------------------------------------
def load_labels(meta_json: str) -> np.ndarray:
    meta = _read_json(meta_json)
    arr = meta.get("arrays", {}).get("is_coal_member", None)
    if arr is None:
        raise ValueError("meta.json missing arrays.is_coal_member")
    return np.asarray(arr, dtype=np.int8)


def load_w_series(evidence_csv: str, col: str, T_max: int) -> np.ndarray:
    """Return dense w_t for t=1..T_max (missing t filled with 0)."""
    t_list: List[int] = []
    w_list: List[float] = []
    with open(evidence_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or "t" not in r.fieldnames:
            raise ValueError("evidence.csv missing t")
        if col not in r.fieldnames:
            raise ValueError(f"evidence.csv missing {col}")
        for row in r:
            t = int(row["t"])
            if 1 <= t <= T_max:
                t_list.append(t)
                w_list.append(float(row[col]) if row[col] else 0.0)

    w_t = np.zeros((T_max,), dtype=np.float64)
    if t_list:
        t_arr = np.asarray(t_list, dtype=np.int64)
        w_arr = np.asarray(w_list, dtype=np.float64)
        w_t[t_arr - 1] = w_arr
    return w_t


def _load_exposures_npz(exposures_npz: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    exposures_by_t.npz must contain:
      t_ptr: int64 (T+1,)
      u_ids: int32 (nnz,)
      a_ut : float32/float64 (nnz,)
    """
    z = np.load(exposures_npz, allow_pickle=False)
    if "t_ptr" not in z or "u_ids" not in z or "a_ut" not in z:
        raise ValueError("exposures_by_t.npz missing required arrays: t_ptr, u_ids, a_ut")
    t_ptr = np.asarray(z["t_ptr"], dtype=np.int64)
    u_ids = np.asarray(z["u_ids"], dtype=np.int32)
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


def detection_delay_t_tau_npz(
    exposures_npz: str,
    labels: np.ndarray,
    w_t: np.ndarray,
    tau: float,
    eval_stride: int,
) -> int:
    """
    Compute t_tau := earliest t with AUC(S^(t)) >= tau, where
      S^(t)(u) = sum_{s<=t} a_{u,s} * w_s.

    Uses exposures_by_t.npz sparse-by-interval form (CSR-by-time):
      for each t: events are u_ids[t_ptr[t-1]:t_ptr[t]], weights a_ut[...] .
    """
    T_max = int(len(w_t))
    N = int(len(labels))
    S = np.zeros((N,), dtype=np.float64)

    eval_stride = max(1, int(eval_stride))
    t_ptr, u_ids, a_ut = _load_exposures_npz(exposures_npz)
    T_exp = int(t_ptr.shape[0] - 1)
    T_use = min(T_max, T_exp)

    next_eval_t = eval_stride

    for t in range(1, T_use + 1):
        start = int(t_ptr[t - 1])
        end = int(t_ptr[t])
        if end > start:
            wt = float(w_t[t - 1])
            users = u_ids[start:end]
            weights = a_ut[start:end]
            for u, a in zip(users, weights):
                uu = int(u)
                if 0 <= uu < N:
                    S[uu] += float(a) * wt

        while next_eval_t <= t and next_eval_t <= T_use:
            if roc_auc(labels, S) >= tau:
                return int(next_eval_t)
            next_eval_t += eval_stride

    while next_eval_t <= T_use:
        if roc_auc(labels, S) >= tau:
            return int(next_eval_t)
        next_eval_t += eval_stride

    return int(T_max + 1)


# ------------------------------------------------------------
# Exposure matching
# ------------------------------------------------------------
def k_on_from_Rexp(Rexp: float, p_norm: float, N_coal: int, p_on: float) -> int:
    """
    R_exp = (p_on * k_on / N_coal) / p_norm  =>  k_on = R_exp * p_norm * N_coal / p_on
    Clamp to [1, N_coal] if p_on>0 else 0.
    """
    if p_on <= 0:
        return 0
    k = int(round((float(Rexp) * float(p_norm) * float(N_coal)) / max(float(p_on), 1e-12)))
    return max(1, min(int(N_coal), k))


# ------------------------------------------------------------
# Warm-up baseline cache (important!)
# ------------------------------------------------------------
def warmup_baseline_cache(
    Rexp: float,
    p_on_min: float,
    baseline_cache_dir: str,
    runs_root: str,
) -> None:
    """
    Ensure baseline cache contains b(n,h) up to the MAX n that can occur in the sweep.
    We warm up at smallest p_on -> largest k_on -> largest Nmax.
    We use Δμ=0 (null) so it's a pure calibration run.
    """
    from rotation_generator import RotationParams, generate_rotation_dataset
    from evidence_and_baselines import EngineParams, run_evidence_engine

    _ensure_dir(baseline_cache_dir)

    seed = int(CFG["seed_base"]) + 777 + int(round(1000 * Rexp))
    T = int(CFG["T_MAX"])
    N_norm = int(CFG["N_norm"])
    N_coal = int(CFG["N_coal"])
    p_norm = float(CFG["p_norm"])
    p_on = float(p_on_min)
    k_on = k_on_from_Rexp(Rexp, p_norm, N_coal, p_on)

    mu = float(CFG["mu_norm"])
    sigma = float(math.sqrt(float(CFG["var_norm"])))
    bins = int(CFG["bins"])

    D_norm = {"type": "normal", "mu": mu, "sigma": sigma}

    params = RotationParams(
        N_norm=N_norm,
        N_coal=N_coal,
        T=T,
        p_norm=p_norm,
        p_on=p_on,
        k_on=int(k_on),
        bins=bins,
        x_min=float(CFG["x_min"]),
        x_max=float(CFG["x_max"]),
        seed=int(seed),
        D_norm=D_norm,
        D_coal=D_norm,  # null
        actions_per_participant_norm=1,
        actions_per_participant_coal=1,
    )

    warm_dir = os.path.join(runs_root, f"_warmup_Rexp_{Rexp:.1f}")
    _ensure_dir(warm_dir)

    step(f"WARMUP baseline cache for Rexp={Rexp} at p_on={p_on_min} (k_on={k_on})")
    generate_rotation_dataset(params, out_dir=warm_dir)
    _require(os.path.join(warm_dir, "exposures_by_t.npz"))

    eng = EngineParams(
        H_plat_spec=D_norm,
        ref_sample_size=int(CFG["ref_sample_size"]),
        mc_reps=int(CFG["mc_reps"]),
        mc_seed=int(CFG["baseline_mc_seed"]),
        cache_tables=bool(CFG["cache_tables"]),
        out_dir=warm_dir,
        baseline_cache_dir=baseline_cache_dir,
    )
    run_evidence_engine(warm_dir, eng)
    log(f"[warmup] baseline cache ready: {baseline_cache_dir}")


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------
def run_experiment_to_csv(exp_csv: str, runs_root: str, baseline_cache_root: str) -> None:
    from rotation_generator import RotationParams, generate_rotation_dataset
    from evidence_and_baselines import EngineParams, run_evidence_engine

    _ensure_dir(runs_root)
    _ensure_dir(baseline_cache_root)
    _ensure_dir(os.path.dirname(exp_csv) or ".")

    T_MAX = int(CFG["T_MAX"])
    stride = int(CFG["EVAL_STRIDE"])
    tau = float(CFG["TAU"])

    dmus = np.linspace(float(CFG["dmu_min"]), float(CFG["dmu_max"]), int(CFG["dmu_points"]), dtype=np.float64)
    pons = np.linspace(float(CFG["pon_min"]), float(CFG["pon_max"]), int(CFG["pon_points"]), dtype=np.float64)
    Rexp_vals = [float(x) for x in CFG["Rexp_values"]]

    # Warm up caches per Rexp
    for Rexp in Rexp_vals:
        cache_dir = os.path.join(baseline_cache_root, f"Rexp_{Rexp:.1f}")
        warmup_baseline_cache(Rexp, p_on_min=float(CFG["pon_min"]), baseline_cache_dir=cache_dir, runs_root=runs_root)

    rows: List[Dict[str, Any]] = []

    for Rexp in Rexp_vals:
        cache_dir = os.path.join(baseline_cache_root, f"Rexp_{Rexp:.1f}")
        _ensure_dir(cache_dir)

        step(f"SWEEP Rexp={Rexp:.1f}")
        for p_on in pons:
            for dmu in dmus:
                ttaus: List[int] = []

                for r in range(int(CFG["runs"])):
                    seed = (
                        int(CFG["seed_base"]) + r
                        + int(round(1_000_000 * p_on))
                        + int(round(1_000_000 * dmu))
                        + int(round(1000 * Rexp))
                    )

                    N_norm = int(CFG["N_norm"])
                    N_coal = int(CFG["N_coal"])
                    p_norm = float(CFG["p_norm"])
                    k_on = k_on_from_Rexp(Rexp, p_norm, N_coal, float(p_on))

                    mu = float(CFG["mu_norm"])
                    sigma = float(math.sqrt(float(CFG["var_norm"])))
                    D_norm = {"type": "normal", "mu": mu, "sigma": sigma}
                    D_coal = {"type": "normal_shift", "delta": float(dmu)}

                    run_dir = os.path.join(
                        runs_root,
                        f"Rexp_{Rexp:.1f}",
                        f"pon_{p_on:.3f}",
                        f"dmu_{dmu:.4f}",
                        f"run_{r:02d}",
                    )
                    _ensure_dir(run_dir)

                    # generate
                    params = RotationParams(
                        N_norm=N_norm,
                        N_coal=N_coal,
                        T=T_MAX,
                        p_norm=p_norm,
                        p_on=float(p_on),
                        k_on=int(k_on),
                        bins=int(CFG["bins"]),
                        x_min=float(CFG["x_min"]),
                        x_max=float(CFG["x_max"]),
                        seed=int(seed),
                        D_norm=D_norm,
                        D_coal=D_coal,
                        actions_per_participant_norm=1,
                        actions_per_participant_coal=1,
                    )
                    generate_rotation_dataset(params, out_dir=run_dir)
                    _require(os.path.join(run_dir, "exposures_by_t.npz"))

                    # evidence
                    eng = EngineParams(
                        H_plat_spec=D_norm,
                        ref_sample_size=int(CFG["ref_sample_size"]),
                        mc_reps=int(CFG["mc_reps"]),
                        mc_seed=int(CFG["baseline_mc_seed"]),
                        cache_tables=bool(CFG["cache_tables"]),
                        out_dir=run_dir,
                        baseline_cache_dir=cache_dir,
                    )
                    run_evidence_engine(run_dir, eng)

                    # compute delay using NPZ exposures
                    meta_json = os.path.join(run_dir, "meta.json")
                    exp_npz = os.path.join(run_dir, "exposures_by_t.npz")
                    ev_csv = os.path.join(run_dir, "evidence.csv")
                    _require(meta_json); _require(exp_npz); _require(ev_csv)

                    labels = load_labels(meta_json)
                    w_t = load_w_series(ev_csv, f"w__{METHOD}", T_max=T_MAX)
                    t_tau = detection_delay_t_tau_npz(exp_npz, labels, w_t, tau=tau, eval_stride=stride)
                    ttaus.append(int(t_tau))

                mean_tau = float(np.mean(np.asarray(ttaus, dtype=np.float64)))
                rows.append({
                    "Rexp": float(Rexp),
                    "p_on": float(p_on),
                    "delta_mu": float(dmu),
                    "t_tau_mean": mean_tau,
                    "t_tau_q05": float(np.quantile(ttaus, 0.05)),
                    "t_tau_q95": float(np.quantile(ttaus, 0.95)),
                })

                log(f"Rexp={Rexp:.1f} p_on={p_on:.3f} dmu={dmu:.4f}  t_tau_mean={mean_tau:.1f}")

    with open(exp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Rexp", "p_on", "delta_mu", "t_tau_mean", "t_tau_q05", "t_tau_q95"])
        w.writeheader()
        w.writerows(rows)

    log(f"[ok] wrote -> {exp_csv}")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def plot_from_csv(exp_csv: str, out_png: str, out_pdf: str) -> None:
    _require(exp_csv)

    dmus = np.linspace(float(CFG["dmu_min"]), float(CFG["dmu_max"]), int(CFG["dmu_points"]), dtype=np.float64)
    pons = np.linspace(float(CFG["pon_min"]), float(CFG["pon_max"]), int(CFG["pon_points"]), dtype=np.float64)
    Rexp_vals = [float(x) for x in CFG["Rexp_values"]]
    T_MAX = int(CFG["T_MAX"])

    grid: Dict[float, np.ndarray] = {R: np.full((len(pons), len(dmus)), np.nan, dtype=np.float64) for R in Rexp_vals}

    with open(exp_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            R = float(row["Rexp"])
            p_on = float(row["p_on"])
            dmu = float(row["delta_mu"])
            val = float(row["t_tau_mean"])
            ip = int(np.argmin(np.abs(pons - p_on)))
            idm = int(np.argmin(np.abs(dmus - dmu)))
            grid[R][ip, idm] = val

    step("PLOT — Fig.11")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)

    vmin = 0.0
    vmax = float(T_MAX + 1)

    for j, R in enumerate(Rexp_vals):
        ax = axes[j]
        M = grid[R]

        im = ax.imshow(
            M,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            extent=[dmus.min(), dmus.max(), pons.min(), pons.max()],
        )

        ax.set_title(rf"$R_{{\exp}}={R:.0f}$")
        ax.set_xlabel(r"mean shift $\Delta\mu$")
        if j == 0:
            ax.set_ylabel(r"intermittency $p_{\mathrm{on}}$")

        ax.axvline(0.0, lw=1.0, linestyle=":", alpha=0.7)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label(r"$t_{\tau}$ (mean across seeds; $T_{\max}+1$ = not reached)")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log(f"[plot] saved PNG -> {out_png}")
    log(f"[plot] saved PDF -> {out_pdf}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Fig12: detection delay heatmaps (t_tau over Δμ×p_on)")
    ap.add_argument("--plot_only", action="store_true", default=False)

    # overrides
    ap.add_argument("--runs", type=int, default=int(CFG["runs"]))
    ap.add_argument("--T_MAX", type=int, default=int(CFG["T_MAX"]))
    ap.add_argument("--EVAL_STRIDE", type=int, default=int(CFG["EVAL_STRIDE"]))
    ap.add_argument("--TAU", type=float, default=float(CFG["TAU"]))

    ap.add_argument("--dmu_min", type=float, default=float(CFG["dmu_min"]))
    ap.add_argument("--dmu_max", type=float, default=float(CFG["dmu_max"]))
    ap.add_argument("--dmu_points", type=int, default=int(CFG["dmu_points"]))
    ap.add_argument("--pon_min", type=float, default=float(CFG["pon_min"]))
    ap.add_argument("--pon_max", type=float, default=float(CFG["pon_max"]))
    ap.add_argument("--pon_points", type=int, default=int(CFG["pon_points"]))

    ap.add_argument("--mc_reps", type=int, default=int(CFG["mc_reps"]))
    ap.add_argument("--ref_sample_size", type=int, default=int(CFG["ref_sample_size"]))
    ap.add_argument("--baseline_mc_seed", type=int, default=int(CFG["baseline_mc_seed"]))
    ap.add_argument("--no_cache", action="store_true", default=False)

    args = ap.parse_args(argv)

    CFG["runs"] = int(args.runs)
    CFG["T_MAX"] = int(args.T_MAX)
    CFG["EVAL_STRIDE"] = int(args.EVAL_STRIDE)
    CFG["TAU"] = float(args.TAU)

    CFG["dmu_min"] = float(args.dmu_min)
    CFG["dmu_max"] = float(args.dmu_max)
    CFG["dmu_points"] = int(args.dmu_points)
    CFG["pon_min"] = float(args.pon_min)
    CFG["pon_max"] = float(args.pon_max)
    CFG["pon_points"] = int(args.pon_points)

    CFG["mc_reps"] = int(args.mc_reps)
    CFG["ref_sample_size"] = int(args.ref_sample_size)
    CFG["baseline_mc_seed"] = int(args.baseline_mc_seed)
    CFG["cache_tables"] = bool(not args.no_cache)

    data_dir = os.path.join(THIS_DIR, "data")
    fig_dir = os.path.join(THIS_DIR, "figure")
    runs_root = os.path.join(data_dir, "runs")
    baseline_cache_root = os.path.join(data_dir, "baseline_cache")

    _ensure_dir(data_dir)
    _ensure_dir(fig_dir)
    _ensure_dir(runs_root)
    _ensure_dir(baseline_cache_root)

    exp_csv = os.path.join(data_dir, "fig12_delay_heatmap.csv")
    cfg_json = os.path.join(data_dir, "fig12_config.json")
    out_png = os.path.join(fig_dir, "fig12_delay_heatmap.png")
    out_pdf = os.path.join(fig_dir, "fig12_delay_heatmap.pdf")

    step("FIG12 START")
    log(f"Mode: {'PLOT-ONLY' if args.plot_only else 'FULL'}")
    log(f"TAU={CFG['TAU']}  stride={CFG['EVAL_STRIDE']}  T_MAX={CFG['T_MAX']}  runs={CFG['runs']}")
    log(f"Δμ grid: [{CFG['dmu_min']},{CFG['dmu_max']}] x {CFG['dmu_points']}")
    log(f"p_on grid: [{CFG['pon_min']},{CFG['pon_max']}] x {CFG['pon_points']}")
    log(f"mc_reps={CFG['mc_reps']}  ref_sample_size={CFG['ref_sample_size']}  cache_tables={CFG['cache_tables']}")
    log(f"baseline_cache_root={os.path.abspath(baseline_cache_root)}")

    if not args.plot_only:
        _write_json(cfg_json, {**CFG, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        run_experiment_to_csv(exp_csv, runs_root=runs_root, baseline_cache_root=baseline_cache_root)
        _write_json(os.path.join(fig_dir, "fig12_manifest.json"), {
            "CFG": dict(CFG),
            "outputs": {"png": os.path.abspath(out_png), "pdf": os.path.abspath(out_pdf), "experiment_csv": os.path.abspath(exp_csv)},
            "baseline_cache_root": os.path.abspath(baseline_cache_root),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

    plot_from_csv(exp_csv, out_png, out_pdf)

    step("FIG12 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
