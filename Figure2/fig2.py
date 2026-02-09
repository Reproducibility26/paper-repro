#!/usr/bin/env python3
"""
fig2.py — Figure 2: Baseline-correction ablation under intermittency.

What this script does:
  1) Generate two synthetic runs under the rotation model:
       - WITH coalition: intermittent activity (p_on > 0) with enforced exposure ratio.
       - NO-coal control: identical setup but with p_on = 0 (no coalition effect).
  2) Compute interval-level evidence using Script #2, including:
       - Raw transport score X_t  = score__ours_w1
       - Baseline-corrected increment d_t = w__ours_w1
  3) Accumulate per-account scores S(u) = sum_t a_{u,t} * X_t or d_t using
     exposures_by_t.npz, and compute median + 5–95% bands for coalition vs normal users.
  4) Plot cumulative trajectories comparing uncorrected vs corrected accumulation.

Inputs / outputs (relative to this file):
  - Data directories:        ./data/with_coalition/, ./data/no_coalition/
  - Experiment cache CSV:   ./data/fig2_experiment.csv
  - Figures:                ./figure/fig2_baseline_correction_ablation.{png,pdf}
  - Config / manifest:      ./figure/fig2_config.json, ./figure/fig2_manifest.json

Dependencies (repo root on PYTHONPATH):
  - rotation_generator.py       (RotationParams, generate_rotation_dataset)
  - evidence_and_baselines.py   (EngineParams, run_evidence_engine)
  - exposures_by_t.npz format   (compact CSR-by-interval exposure representation)

Usage:
  # Full pipeline: generate data, compute evidence, accumulate scores, plot
  python fig2.py

  # Plot-only mode (reuse existing fig2_experiment.csv)
  python fig2.py --plot_only

  # Common parameter overrides
  python fig2.py --delta_mu 0.05 --p_on 0.3 --R_exp 1.5
  python fig2.py --mc_reps 300 --ref_sample_size 200000 --baseline_mc_seed 999

Notes:
  - Baseline Monte Carlo tables are cached under ./data/baseline_cache/ for reuse.
  - The key diagnostic is that uncorrected X_t accumulation drifts under the null,
    while corrected d_t accumulation is approximately mean-zero for normal users.

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
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


# -------------------------
# Path setup: script in Figure2, modules in parent dir
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if A_DIR not in sys.path:
    sys.path.insert(0, A_DIR)


# -------------------------
# Logging
# -------------------------
def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def step(title: str) -> None:
    bar = "-" * 72
    print(f"\n{bar}\n[{_ts()}] {title}\n{bar}", flush=True)


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Fig2 config
# -------------------------
CFG: Dict[str, Any] = dict(
    figure="fig2_baseline_correction_ablation",

    # timeline / population
    T=4000,
    N_norm=20000,
    N_coal=2000,

    # behavior (intermittency)
    p_norm=0.04,
    p_on=0.20,         # intermittent coalition campaign
    R_exp_target=1.0,  # enforce exposure ratio

    # histogram support
    bins=50,
    x_min=0.0,
    x_max=1.0,

    # signal model
    mu_norm=0.50,
    var_norm=0.05,
    delta_mu=0.02,     # small shift ok; ablation is about correction

    # evidence baseline MC (new oracle)
    mc_reps=300,
    ref_sample_size=200_000,
    cache_tables=True,

    # IMPORTANT: fixed baseline MC seed for cache reuse (per-figure)
    baseline_mc_seed=999,

    # plotting / smoothing
    smooth_w=25,
    q_lo=0.05,
    q_hi=0.95,

    # seeds (generator)
    seed_with=10,
    seed_nocoal=11,
)


def write_config_json(path: str) -> None:
    cfg = dict(CFG)

    p_norm = float(cfg["p_norm"])
    p_on = float(cfg["p_on"])
    N_coal = int(cfg["N_coal"])
    R_exp = float(cfg["R_exp_target"])

    k_on = int(round((R_exp * p_norm * N_coal) / max(p_on, 1e-12)))
    k_on = max(1, min(N_coal, k_on))

    cfg["derived_k_on"] = k_on
    cfg["derived_R_exp_realized"] = (p_on * k_on / N_coal) / max(p_norm, 1e-12)
    cfg["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cfg["script"] = os.path.basename(__file__)

    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    log(f"[config] wrote -> {path}")


# -------------------------
# Predict N_max (for MC sizing)
# -------------------------
def estimate_Nmax(
    N_norm: int,
    p_norm: float,
    N_coal: int,
    p_on: float,
    R_exp_target: float,
    safety_sigma: float = 3.0,
) -> Dict[str, float]:
    mu_norm = float(N_norm) * float(p_norm)
    sigma_norm = float(np.sqrt(max(mu_norm, 1e-12)))

    if p_on <= 0:
        k_on = 0.0
    else:
        k_on = (float(R_exp_target) * float(p_norm) * float(N_coal)) / max(float(p_on), 1e-12)
        k_on = max(1.0, min(float(N_coal), float(k_on)))

    Nmax_est = mu_norm + k_on + float(safety_sigma) * sigma_norm
    return {"mu_norm": mu_norm, "sigma_norm": sigma_norm, "k_on": k_on, "Nmax_est": Nmax_est}


# -------------------------
# Utilities: smoothing + quantile bands
# -------------------------
def smooth_moving_average(x: np.ndarray, w: int) -> np.ndarray:
    w = int(max(1, w))
    if w <= 1:
        return x.astype(np.float64, copy=True)
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x.astype(np.float64), kernel, mode="same")


def quantile_bands(arr: np.ndarray, lo: float, hi: float) -> Tuple[float, float, float]:
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    med = float(np.quantile(arr, 0.5))
    ql = float(np.quantile(arr, lo))
    qh = float(np.quantile(arr, hi))
    return med, ql, qh


def load_labels(meta_json: str) -> np.ndarray:
    meta = _read_json(meta_json)
    arr = meta.get("arrays", {}).get("is_coal_member", None)
    if arr is None:
        raise ValueError("meta.json missing arrays.is_coal_member")
    return np.asarray(arr, dtype=np.int8)


def load_series_from_evidence(evidence_csv: str, col: str) -> Tuple[np.ndarray, int]:
    _require(evidence_csv)
    t_list: List[int] = []
    v_list: List[float] = []
    with open(evidence_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("evidence.csv has no header")
        if "t" not in r.fieldnames:
            raise ValueError("evidence.csv missing 't'")
        if col not in r.fieldnames:
            raise ValueError(f"evidence.csv missing '{col}'")
        for row in r:
            t_list.append(int(row["t"]))
            v_list.append(float(row[col]) if row[col] != "" else 0.0)
    if not t_list:
        raise ValueError("evidence.csv appears empty")

    t_arr = np.asarray(t_list, dtype=np.int64)
    v_arr = np.asarray(v_list, dtype=np.float64)
    T = int(np.max(t_arr))
    out = np.zeros((T,), dtype=np.float64)
    out[t_arr - 1] = v_arr
    return out, T


def compute_bands_from_exposures_npz(
    exposures_npz: str,
    y_is_coal: np.ndarray,
    w_t: np.ndarray,
    T: int,
    q_lo: float,
    q_hi: float,
    want_coal_and_norm: bool,
) -> Dict[str, np.ndarray]:
    """
    scores[u] += a_ut * w_t[t-1], then record quantile bands at each t.
    Exposures come from exposures_by_t.npz (new pipeline).
    """
    _require(exposures_npz)
    N_total = int(len(y_is_coal))
    is_coal = (y_is_coal == 1)
    coal_idx = np.where(is_coal)[0]
    norm_idx = np.where(~is_coal)[0]

    scores = np.zeros((N_total,), dtype=np.float64)

    if want_coal_and_norm:
        coal_med = np.zeros((T,), dtype=np.float64)
        coal_qlo = np.zeros((T,), dtype=np.float64)
        coal_qhi = np.zeros((T,), dtype=np.float64)
        norm_med = np.zeros((T,), dtype=np.float64)
        norm_qlo = np.zeros((T,), dtype=np.float64)
        norm_qhi = np.zeros((T,), dtype=np.float64)
    else:
        norm_med = np.zeros((T,), dtype=np.float64)
        norm_qlo = np.zeros((T,), dtype=np.float64)
        norm_qhi = np.zeros((T,), dtype=np.float64)

    z = np.load(exposures_npz, allow_pickle=False)
    if "t_ptr" not in z or "u_ids" not in z or "a_ut" not in z:
        raise ValueError("exposures_by_t.npz missing required arrays: t_ptr, u_ids, a_ut")

    t_ptr = np.asarray(z["t_ptr"], dtype=np.int64)
    u_ids = np.asarray(z["u_ids"], dtype=np.int32)
    a_ut = np.asarray(z["a_ut"], dtype=np.float64)

    T_exp = int(t_ptr.shape[0] - 1)
    T_use = min(int(T), int(T_exp), int(len(w_t)))

    def finalize_time(t_val: int) -> None:
        i = t_val - 1
        if i < 0 or i >= T_use:
            return
        if want_coal_and_norm:
            cm, cl, ch = quantile_bands(scores[coal_idx], lo=q_lo, hi=q_hi)
            nm, nl, nh = quantile_bands(scores[norm_idx], lo=q_lo, hi=q_hi)
            coal_med[i], coal_qlo[i], coal_qhi[i] = cm, cl, ch
            norm_med[i], norm_qlo[i], norm_qhi[i] = nm, nl, nh
        else:
            nm, nl, nh = quantile_bands(scores[norm_idx], lo=q_lo, hi=q_hi)
            norm_med[i], norm_qlo[i], norm_qhi[i] = nm, nl, nh

    for t in range(1, T_use + 1):
        start = int(t_ptr[t - 1])
        end = int(t_ptr[t])
        if end > start:
            wt = float(w_t[t - 1])
            users = u_ids[start:end]
            weights = a_ut[start:end]
            for u, a in zip(users, weights):
                uu = int(u)
                if 0 <= uu < N_total:
                    scores[uu] += float(a) * wt
        finalize_time(t)

    # pad to requested T (should normally match)
    if T_use < T:
        src = T_use - 1 if T_use > 0 else 0
        for t in range(T_use + 1, T + 1):
            i = t - 1
            if want_coal_and_norm:
                coal_med[i] = coal_med[src]
                coal_qlo[i] = coal_qlo[src]
                coal_qhi[i] = coal_qhi[src]
                norm_med[i] = norm_med[src]
                norm_qlo[i] = norm_qlo[src]
                norm_qhi[i] = norm_qhi[src]
            else:
                norm_med[i] = norm_med[src]
                norm_qlo[i] = norm_qlo[src]
                norm_qhi[i] = norm_qhi[src]

    if want_coal_and_norm:
        return dict(
            coal_med=coal_med, coal_qlo=coal_qlo, coal_qhi=coal_qhi,
            norm_med=norm_med, norm_qlo=norm_qlo, norm_qhi=norm_qhi,
        )
    else:
        return dict(norm_med=norm_med, norm_qlo=norm_qlo, norm_qhi=norm_qhi)


# ============================================================
# 1) Generate data (two run dirs)
# ============================================================
def generate_two_runs(run_with_dir: str, run_nocoal_dir: str) -> Dict[str, Any]:
    from rotation_generator import RotationParams, generate_rotation_dataset

    T = int(CFG["T"])
    N_norm = int(CFG["N_norm"])
    N_coal = int(CFG["N_coal"])
    p_norm = float(CFG["p_norm"])
    p_on = float(CFG["p_on"])
    Rexp = float(CFG["R_exp_target"])

    # Enforce exposure ratio: R_exp = (p_on*k_on/N_coal)/p_norm
    k_on = int(round((Rexp * p_norm * N_coal) / max(p_on, 1e-12)))
    k_on = max(1, min(N_coal, k_on))
    realized = (p_on * k_on / N_coal) / max(p_norm, 1e-12)

    mu_norm = float(CFG["mu_norm"])
    sigma_norm = float(np.sqrt(float(CFG["var_norm"])))
    delta_mu = float(CFG["delta_mu"])

    bins = int(CFG["bins"])
    x_min = float(CFG["x_min"])
    x_max = float(CFG["x_max"])

    D_norm = {"type": "normal", "mu": mu_norm, "sigma": sigma_norm}
    D_coal = {"type": "normal_shift", "delta": delta_mu}

    params_with = RotationParams(
        N_norm=N_norm,
        N_coal=N_coal,
        T=T,
        p_norm=p_norm,
        p_on=p_on,
        k_on=k_on,
        bins=bins,
        x_min=x_min,
        x_max=x_max,
        seed=int(CFG["seed_with"]),
        D_norm=D_norm,
        D_coal=D_coal,
        actions_per_participant_norm=1,
        actions_per_participant_coal=1,
    )

    # No-coal control: p_on=0, k_on=0 (matches old logic)
    params_nocoal = RotationParams(
        N_norm=N_norm,
        N_coal=N_coal,
        T=T,
        p_norm=p_norm,
        p_on=0.0,
        k_on=0,
        bins=bins,
        x_min=x_min,
        x_max=x_max,
        seed=int(CFG["seed_nocoal"]),
        D_norm=D_norm,
        D_coal=D_norm,
        actions_per_participant_norm=1,
        actions_per_participant_coal=1,
    )

    _ensure_dir(run_with_dir)
    _ensure_dir(run_nocoal_dir)

    step("GENERATE — WITH coalition (intermittent)")
    log(f"run_dir={run_with_dir}")
    log(f"k_on={k_on}  target_Rexp={Rexp:.3g}  realized_Rexp={realized:.3g}  p_on={p_on}")
    generate_rotation_dataset(params_with, out_dir=run_with_dir)

    step("GENERATE — NO-coal control (p_on=0, k_on=0)")
    log(f"run_dir={run_nocoal_dir}")
    generate_rotation_dataset(params_nocoal, out_dir=run_nocoal_dir)

    return {
        "k_on": k_on,
        "Rexp_realized": realized,
        "D_norm": D_norm,
        "D_coal": D_coal,
        "params_with": asdict(params_with),
        "params_nocoal": asdict(params_nocoal),
    }


# ============================================================
# 2) Experiment: compute X_t vs d_t and score bands
# ============================================================
def run_experiment(run_with_dir: str, run_nocoal_dir: str, out_experiment_csv: str, baseline_cache_dir: str) -> Dict[str, Any]:
    from evidence_and_baselines import EngineParams, run_evidence_engine

    # Predict N_max (WITH setting)
    step("PREDICT N_max (before evidence)")
    est = estimate_Nmax(
        N_norm=int(CFG["N_norm"]),
        p_norm=float(CFG["p_norm"]),
        N_coal=int(CFG["N_coal"]),
        p_on=float(CFG["p_on"]),
        R_exp_target=float(CFG["R_exp_target"]),
        safety_sigma=3.0,
    )
    log("Predicted interval sample size (from settings):")
    log(f"  E[n_norm]      = {est['mu_norm']:.1f}")
    log(f"  std[n_norm]    = {est['sigma_norm']:.1f}")
    log(f"  k_on (coal)    = {est['k_on']:.1f}")
    log(f"  ==> N_max_est  ≈ {int(np.ceil(est['Nmax_est']))}")
    est_cost = int(np.ceil(est["Nmax_est"])) * int(CFG["mc_reps"])
    log(f"Baseline MC workload ≈ mc_reps × N_max ≈ {est_cost:,}")
    log(f"baseline_cache_dir={os.path.abspath(baseline_cache_dir)}")

    H_plat_spec = {
        "type": "normal",
        "mu": float(CFG["mu_norm"]),
        "sigma": float(np.sqrt(float(CFG["var_norm"]))),
    }

    _ensure_dir(baseline_cache_dir)

    # Evidence for WITH (shared cache dir)
    step("EVIDENCE — WITH run")
    eng_with = EngineParams(
        H_plat_spec=H_plat_spec,
        ref_sample_size=int(CFG["ref_sample_size"]),
        mc_reps=int(CFG["mc_reps"]),
        mc_seed=int(CFG["baseline_mc_seed"]),
        cache_tables=bool(CFG["cache_tables"]),
        out_dir=run_with_dir,
        baseline_cache_dir=baseline_cache_dir,
    )
    log(f"mc_reps={eng_with.mc_reps}, ref_sample_size={eng_with.ref_sample_size}, mc_seed={eng_with.mc_seed}, cache_tables={eng_with.cache_tables}")
    run_evidence_engine(run_with_dir, eng_with)

    # Evidence for NO-coal (shared cache dir)
    step("EVIDENCE — NO-coal run")
    eng_nc = EngineParams(
        H_plat_spec=H_plat_spec,
        ref_sample_size=int(CFG["ref_sample_size"]),
        mc_reps=int(CFG["mc_reps"]),
        mc_seed=int(CFG["baseline_mc_seed"]),
        cache_tables=bool(CFG["cache_tables"]),
        out_dir=run_nocoal_dir,
        baseline_cache_dir=baseline_cache_dir,
    )
    log(f"mc_reps={eng_nc.mc_reps}, ref_sample_size={eng_nc.ref_sample_size}, mc_seed={eng_nc.mc_seed}, cache_tables={eng_nc.cache_tables}")
    run_evidence_engine(run_nocoal_dir, eng_nc)

    # Required files (NEW pipeline: exposures_by_t.npz)
    ev_with = os.path.join(run_with_dir, "evidence.csv")
    ev_nc = os.path.join(run_nocoal_dir, "evidence.csv")
    meta_with = os.path.join(run_with_dir, "meta.json")
    meta_nc = os.path.join(run_nocoal_dir, "meta.json")
    exp_with = os.path.join(run_with_dir, "exposures_by_t.npz")
    exp_nc = os.path.join(run_nocoal_dir, "exposures_by_t.npz")
    for p in [ev_with, ev_nc, meta_with, meta_nc, exp_with, exp_nc]:
        _require(p)

    y_with = load_labels(meta_with)
    y_nc = load_labels(meta_nc)
    if len(y_with) != len(y_nc):
        raise ValueError("WITH and NO-coal runs have different N_total in meta.json.")
    if not np.all(y_with == y_nc):
        log("[warn] coalition membership arrays differ; using WITH labels for grouping.")

    # Interval series:
    #   X_t: raw transport score (score__ours_w1)
    #   d_t: baseline-corrected increment (w__ours_w1)
    X_with, T1 = load_series_from_evidence(ev_with, "score__ours_w1")
    d_with, T2 = load_series_from_evidence(ev_with, "w__ours_w1")
    X_nc, T3 = load_series_from_evidence(ev_nc, "score__ours_w1")
    d_nc, T4 = load_series_from_evidence(ev_nc, "w__ours_w1")

    T = min(T1, T2, T3, T4)
    X_with = X_with[:T]
    d_with = d_with[:T]
    X_nc = X_nc[:T]
    d_nc = d_nc[:T]

    w_smooth = int(CFG["smooth_w"])
    X_with_s = smooth_moving_average(X_with, w_smooth)
    X_nc_s = smooth_moving_average(X_nc, w_smooth)
    d_with_s = smooth_moving_average(d_with, w_smooth)
    d_nc_s = smooth_moving_average(d_nc, w_smooth)

    qlo = float(CFG["q_lo"])
    qhi = float(CFG["q_hi"])

    # Cumulative score bands under uncorrected weights X_t
    step("BANDS — Uncorrected (X_t) accumulation")
    bands_X_with = compute_bands_from_exposures_npz(exp_with, y_with, X_with, T, qlo, qhi, want_coal_and_norm=True)
    bands_X_nc = compute_bands_from_exposures_npz(exp_nc, y_with, X_nc, T, qlo, qhi, want_coal_and_norm=False)

    # Cumulative score bands under corrected weights d_t
    step("BANDS — Corrected (d_t) accumulation")
    bands_d_with = compute_bands_from_exposures_npz(exp_with, y_with, d_with, T, qlo, qhi, want_coal_and_norm=True)
    bands_d_nc = compute_bands_from_exposures_npz(exp_nc, y_with, d_nc, T, qlo, qhi, want_coal_and_norm=False)

    # Write experiment CSV
    _ensure_dir(os.path.dirname(out_experiment_csv) or ".")
    step("WRITE — fig2_experiment.csv")
    with open(out_experiment_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "X_with", "X_with_smooth",
            "X_nocoal", "X_nocoal_smooth",
            "d_with", "d_with_smooth",
            "d_nocoal", "d_nocoal_smooth",

            "coal_med_X", "coal_q05_X", "coal_q95_X",
            "norm_med_X", "norm_q05_X", "norm_q95_X",
            "norm_nocoal_med_X", "norm_nocoal_q05_X", "norm_nocoal_q95_X",

            "coal_med_d", "coal_q05_d", "coal_q95_d",
            "norm_med_d", "norm_q05_d", "norm_q95_d",
            "norm_nocoal_med_d", "norm_nocoal_q05_d", "norm_nocoal_q95_d",
        ])

        for t in range(1, T + 1):
            i = t - 1
            w.writerow([
                t,
                float(X_with[i]), float(X_with_s[i]),
                float(X_nc[i]), float(X_nc_s[i]),
                float(d_with[i]), float(d_with_s[i]),
                float(d_nc[i]), float(d_nc_s[i]),

                float(bands_X_with["coal_med"][i]), float(bands_X_with["coal_qlo"][i]), float(bands_X_with["coal_qhi"][i]),
                float(bands_X_with["norm_med"][i]), float(bands_X_with["norm_qlo"][i]), float(bands_X_with["norm_qhi"][i]),
                float(bands_X_nc["norm_med"][i]), float(bands_X_nc["norm_qlo"][i]), float(bands_X_nc["norm_qhi"][i]),

                float(bands_d_with["coal_med"][i]), float(bands_d_with["coal_qlo"][i]), float(bands_d_with["coal_qhi"][i]),
                float(bands_d_with["norm_med"][i]), float(bands_d_with["norm_qlo"][i]), float(bands_d_with["norm_qhi"][i]),
                float(bands_d_nc["norm_med"][i]), float(bands_d_nc["norm_qlo"][i]), float(bands_d_nc["norm_qhi"][i]),
            ])

    log(f"[experiment] wrote -> {out_experiment_csv}")
    log(f"[experiment] mean(X_t) no coalition: {float(np.mean(X_nc)):.6g}   (should be > 0)")
    log(f"[experiment] mean(d_t) no coalition: {float(np.mean(d_nc)):.6g}   (should be ~ 0)")
    return {"T": T, "experiment_csv": os.path.abspath(out_experiment_csv)}


# ============================================================
# 3) Plot
# ============================================================
def plot_figure_from_experiment_csv(experiment_csv: str, out_png: str, out_pdf: str) -> None:
    _require(experiment_csv)

    need = [
        "coal_med_X", "coal_q05_X", "coal_q95_X",
        "norm_med_X", "norm_q05_X", "norm_q95_X",
        "coal_med_d", "coal_q05_d", "coal_q95_d",
        "norm_med_d", "norm_q05_d", "norm_q95_d",
    ]
    cols: Dict[str, List[float]] = {k: [] for k in need}
    t: List[int] = []

    with open(experiment_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(int(row["t"]))
            for k in need:
                cols[k].append(float(row[k]))

    tt = np.asarray(t, dtype=np.int64)

    step("PLOT — Fig.2")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))

    c_coal_unc = "tab:blue"
    c_norm_unc = "tab:orange"
    c_coal_cor = "tab:red"
    c_norm_cor = "tab:green"

    a_band_unc = 0.20
    a_band_cor = 0.25
    lw_unc = 2.6
    lw_cor = 2.8

    # Uncorrected
    ax.plot(tt, cols["coal_med_X"], color=c_coal_unc, lw=lw_unc, linestyle="-", label="Coalition (uncorrected)")
    ax.fill_between(tt, cols["coal_q05_X"], cols["coal_q95_X"], color=c_coal_unc, alpha=a_band_unc, linewidth=0)

    ax.plot(tt, cols["norm_med_X"], color=c_norm_unc, lw=lw_unc, linestyle="-", label="Normal (uncorrected)")
    ax.fill_between(tt, cols["norm_q05_X"], cols["norm_q95_X"], color=c_norm_unc, alpha=a_band_unc, linewidth=0)

    # Corrected
    ax.plot(tt, cols["coal_med_d"], color=c_coal_cor, lw=lw_cor, linestyle="-", label="Coalition (corrected)")
    ax.fill_between(tt, cols["coal_q05_d"], cols["coal_q95_d"], color=c_coal_cor, alpha=a_band_cor, linewidth=0)

    ax.plot(tt, cols["norm_med_d"], color=c_norm_cor, lw=lw_cor, linestyle="-", label="Normal (corrected)")
    ax.fill_between(tt, cols["norm_q05_d"], cols["norm_q95_d"], color=c_norm_cor, alpha=a_band_cor, linewidth=0)

    ax.set_xlabel(r"Global interval index $t$")
    ax.set_ylabel(r"Cumulative score")
    ax.set_xlim(0, 4000)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log(f"[plot] saved PNG -> {out_png}")
    log(f"[plot] saved PDF -> {out_pdf}")


# ============================================================
# Main
# ============================================================
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Fig2 (new oracle): baseline-correction ablation under intermittency")
    ap.add_argument("--plot_only", action="store_true", default=False)

    # overrides
    ap.add_argument("--mc_reps", type=int, default=int(CFG["mc_reps"]))
    ap.add_argument("--ref_sample_size", type=int, default=int(CFG["ref_sample_size"]))
    ap.add_argument("--baseline_mc_seed", type=int, default=int(CFG["baseline_mc_seed"]))
    ap.add_argument("--no_cache", action="store_true", default=False)
    ap.add_argument("--delta_mu", type=float, default=float(CFG["delta_mu"]))
    ap.add_argument("--p_on", type=float, default=float(CFG["p_on"]))
    ap.add_argument("--R_exp", type=float, default=float(CFG["R_exp_target"]))

    args = ap.parse_args(argv)

    CFG["mc_reps"] = int(args.mc_reps)
    CFG["ref_sample_size"] = int(args.ref_sample_size)
    CFG["baseline_mc_seed"] = int(args.baseline_mc_seed)
    CFG["cache_tables"] = bool(not args.no_cache)
    CFG["delta_mu"] = float(args.delta_mu)
    CFG["p_on"] = float(args.p_on)
    CFG["R_exp_target"] = float(args.R_exp)

    data_dir = os.path.join(THIS_DIR, "data")
    fig_dir = os.path.join(THIS_DIR, "figure")
    baseline_cache_dir = os.path.join(data_dir, "baseline_cache")

    _ensure_dir(data_dir)
    _ensure_dir(fig_dir)
    _ensure_dir(baseline_cache_dir)

    run_with_dir = os.path.join(data_dir, "with_coalition")
    run_nocoal_dir = os.path.join(data_dir, "no_coalition")
    exp_csv = os.path.join(data_dir, "fig2_experiment.csv")

    out_png = os.path.join(fig_dir, "fig2_baseline_correction_ablation.png")
    out_pdf = os.path.join(fig_dir, "fig2_baseline_correction_ablation.pdf")
    config_json = os.path.join(fig_dir, "fig2_config.json")

    step("FIG2 START")
    log(f"Mode: {'PLOT-ONLY' if args.plot_only else 'FULL'}")
    log(f"mc_reps={CFG['mc_reps']}  ref_sample_size={CFG['ref_sample_size']}  baseline_mc_seed={CFG['baseline_mc_seed']}")
    log(f"cache_tables={CFG['cache_tables']}  baseline_cache_dir={os.path.abspath(baseline_cache_dir)}")
    log(f"p_on={CFG['p_on']}  R_exp_target={CFG['R_exp_target']}  delta_mu={CFG['delta_mu']}")

    if not args.plot_only:
        write_config_json(config_json)
        gen_info = generate_two_runs(run_with_dir, run_nocoal_dir)
        exp_info = run_experiment(run_with_dir, run_nocoal_dir, exp_csv, baseline_cache_dir=baseline_cache_dir)

        manifest_path = os.path.join(fig_dir, "fig2_manifest.json")
        manifest = {
            "CFG": dict(CFG),
            "generator": gen_info,
            "experiment": exp_info,
            "baseline_cache_dir": os.path.abspath(baseline_cache_dir),
            "outputs": {"png": os.path.abspath(out_png), "pdf": os.path.abspath(out_pdf)},
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        log(f"[manifest] wrote -> {manifest_path}")

    plot_figure_from_experiment_csv(exp_csv, out_png, out_pdf)

    step("FIG2 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
