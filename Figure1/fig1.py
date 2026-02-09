#!/usr/bin/env python3
"""
fig1.py — Figure 1: Global-clock divergence trajectories.

What this script does:
  1) Generate two synthetic runs under the rotation model:
       - WITH coalition (shifted signals)
       - NO-coal control (coalition behaves like normal; exposure pattern unchanged)
  2) Compute interval-level evidence increments d_t using Script #2.
  3) Accumulate per-account scores S(u) = sum_t a_{u,t} * d_t from exposures_by_t.npz,
     then plot median and 5–95% bands for coalition/normal groups.

Inputs/outputs (relative to this file):
  - Writes run directories under:   ./data/{with_coalition,no_coalition}/
  - Writes experiment cache CSV:   ./data/fig1_experiment.csv
  - Writes figures to:             ./figure/fig1_global_divergence.{png,pdf}
  - Writes config/manifest to:     ./figure/fig1_config.json, ./figure/fig1_manifest.json

Dependencies (repo root on PYTHONPATH):
  - rotation_generator.py          (generate_rotation_dataset)
  - evidence_and_baselines.py      (run_evidence_engine; uses w__ours_w1 as baseline-corrected increment)

Usage:
  # Full pipeline: generate -> evidence -> bands -> plot
  python fig1.py

  # Plot-only mode (reuses ./data/fig1_experiment.csv)
  python fig1.py --plot_only

  # Common parameter overrides
  python fig1.py --delta_mu 0.10 --p_on 1.0 --R_exp 2.0
  python fig1.py --mc_reps 300 --ref_sample_size 200000 --baseline_mc_seed 999

Notes:
  - Baseline MC tables are cached under ./data/baseline_cache/ to speed up re-runs.
  - The script reads exposures from exposures_by_t.npz (compact CSR-by-interval format).

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Path setup (like old script)
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if A_DIR not in sys.path:
    sys.path.insert(0, A_DIR)


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
# Progress printing
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


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


# -------------------------
# Config (close to old)
# -------------------------
CFG: Dict[str, Any] = dict(
    # timeline / population
    T=4000,
    N_norm=20000,
    N_coal=2000,

    # behavior
    p_norm=0.04,
    p_on=1.0,        # always on
    R_exp_target=2.0,

    # histogram support
    bins=50,
    x_min=0.0,
    x_max=1.0,

    # normal model
    mu_norm=0.50,
    var_norm=0.05,

    # coalition shift
    delta_mu=0.10,

    # evidence baseline MC (reasonable for Fig1)
    mc_reps=300,
    ref_sample_size=200_000,
    cache_tables=True,

    # IMPORTANT: fixed baseline MC seed for cache reuse (per-figure)
    baseline_mc_seed=999,

    # plotting
    smooth_w=25,
    q_lo=0.05,
    q_hi=0.95,

    # seeds (generator)
    seed_with=0,
    seed_nocoal=1,
)


def write_config_json(path: str) -> None:
    cfg = dict(CFG)
    cfg["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cfg["script"] = os.path.basename(__file__)
    cfg["root_dir"] = A_DIR
    cfg["figure_dir"] = THIS_DIR
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    log(f"[config] wrote -> {path}")


# -------------------------
# Estimation of N_max (for sizing MC)
# -------------------------
def estimate_Nmax(
    N_norm: int,
    p_norm: float,
    N_coal: int,
    p_on: float,
    R_exp_target: float,
    safety_sigma: float = 3.0,
) -> Dict[str, float]:
    """
    Predict N_max = max_t n_t from generator parameters.

    Model:
      n_t = Binomial(N_norm, p_norm) + k_on * 1{coalition on}
    where
      k_on = R_exp * p_norm * N_coal / p_on    (clamped to [1, N_coal])
    """
    mu_norm = float(N_norm) * float(p_norm)
    sigma_norm = float(np.sqrt(max(mu_norm, 1e-12)))

    k_on = (float(R_exp_target) * float(p_norm) * float(N_coal)) / max(float(p_on), 1e-12)
    k_on = max(1.0, min(float(N_coal), float(k_on)))

    Nmax_est = mu_norm + k_on + float(safety_sigma) * sigma_norm
    return {
        "mu_norm": mu_norm,
        "sigma_norm": sigma_norm,
        "k_on": k_on,
        "Nmax_est": Nmax_est,
    }


# -------------------------
# Small utilities
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


def load_w_series(evidence_csv: str, col: str) -> Tuple[np.ndarray, int]:
    _require(evidence_csv)
    t_list: List[int] = []
    w_list: List[float] = []
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
            w_list.append(float(row[col]) if row[col] != "" else 0.0)
    if not t_list:
        raise ValueError("evidence.csv appears empty")

    t_arr = np.asarray(t_list, dtype=np.int64)
    w_arr = np.asarray(w_list, dtype=np.float64)
    T = int(np.max(t_arr))
    w_t = np.zeros((T,), dtype=np.float64)
    w_t[t_arr - 1] = w_arr
    return w_t, T


def compute_score_bands_from_exposures_npz(
    exposures_npz: str,
    y_is_coal: np.ndarray,
    w_t: np.ndarray,
    T: int,
    q_lo: float,
    q_hi: float,
    want_coal_and_norm: bool,
) -> Dict[str, np.ndarray]:
    """
       want_coal_and_norm:
      - True  => bands for coalition and normal groups (WITH run)
      - False => bands for normal group only (NO-coal run)
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
        idx = t_val - 1
        if idx < 0 or idx >= T_use:
            return
        if want_coal_and_norm:
            cm, cl, ch = quantile_bands(scores[coal_idx], lo=q_lo, hi=q_hi)
            nm, nl, nh = quantile_bands(scores[norm_idx], lo=q_lo, hi=q_hi)
            coal_med[idx], coal_qlo[idx], coal_qhi[idx] = cm, cl, ch
            norm_med[idx], norm_qlo[idx], norm_qhi[idx] = nm, nl, nh
        else:
            nm, nl, nh = quantile_bands(scores[norm_idx], lo=q_lo, hi=q_hi)
            norm_med[idx], norm_qlo[idx], norm_qhi[idx] = nm, nl, nh

    for t in range(1, T_use + 1):
        start = int(t_ptr[t - 1])
        end = int(t_ptr[t])
        if end > start:
            wt = float(w_t[t - 1])
            users = u_ids[start:end]
            weights = a_ut[start:end]
            # per-event updates (keeps old semantics; no other changes)
            for u, a in zip(users, weights):
                uu = int(u)
                if 0 <= uu < N_total:
                    scores[uu] += float(a) * wt
        finalize_time(t)

    # If caller asked for T longer than exposures/evidence, pad remaining with last value
    # (old code finalized all t; here we keep identical output length T)
    if T_use < T:
        for t in range(T_use + 1, T + 1):
            # just repeat final bands deterministically (no new evidence anyway)
            idx = t - 1
            src = T_use - 1 if T_use > 0 else 0
            if want_coal_and_norm:
                coal_med[idx] = coal_med[src]
                coal_qlo[idx] = coal_qlo[src]
                coal_qhi[idx] = coal_qhi[src]
                norm_med[idx] = norm_med[src]
                norm_qlo[idx] = norm_qlo[src]
                norm_qhi[idx] = norm_qhi[src]
            else:
                norm_med[idx] = norm_med[src]
                norm_qlo[idx] = norm_qlo[src]
                norm_qhi[idx] = norm_qhi[src]

    if want_coal_and_norm:
        return dict(
            coal_med=coal_med, coal_qlo=coal_qlo, coal_qhi=coal_qhi,
            norm_med=norm_med, norm_qlo=norm_qlo, norm_qhi=norm_qhi,
        )
    else:
        return dict(
            norm_med=norm_med, norm_qlo=norm_qlo, norm_qhi=norm_qhi,
        )


# ============================================================
# 1) Generate two runs (new oracle)
# ============================================================
def generate_two_runs(run_with_dir: str, run_nocoal_dir: str) -> Dict[str, Any]:
    from rotation_generator import RotationParams, generate_rotation_dataset

    T = int(CFG["T"])
    N_norm = int(CFG["N_norm"])
    N_coal = int(CFG["N_coal"])
    p_norm = float(CFG["p_norm"])
    p_on = float(CFG["p_on"])
    Rexp = float(CFG["R_exp_target"])

    # enforce exposure ratio
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

    # NO-coal control: coalition behaves like normal (still same exposure pattern)
    params_nocoal = RotationParams(
        N_norm=N_norm,
        N_coal=N_coal,
        T=T,
        p_norm=p_norm,
        p_on=p_on,
        k_on=k_on,
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

    step("GENERATE — WITH coalition")
    log(f"run_dir={run_with_dir}")
    log(f"k_on={k_on}  target_Rexp={Rexp:.3g}  realized_Rexp={realized:.3g}")
    generate_rotation_dataset(params_with, out_dir=run_with_dir)

    step("GENERATE — NO-coal control")
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
# 2) Run experiment from the two runs (new oracle)
# ============================================================
def run_experiment_from_runs(
    run_with_dir: str,
    run_nocoal_dir: str,
    experiment_csv: str,
    baseline_cache_dir: str,
) -> Dict[str, Any]:
    from evidence_and_baselines import EngineParams, run_evidence_engine

    # ------------------------------------------------------------
    # Predict N_max before baseline MC (important for mc_reps)
    # ------------------------------------------------------------
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

    _ensure_dir(baseline_cache_dir)

    # compute evidence for both runs (shared cache)
    H_plat_spec = {"type": "normal", "mu": float(CFG["mu_norm"]), "sigma": float(np.sqrt(CFG["var_norm"]))}

    step("EVIDENCE — WITH run")
    engine = EngineParams(
        H_plat_spec=H_plat_spec,
        ref_sample_size=int(CFG["ref_sample_size"]),
        mc_reps=int(CFG["mc_reps"]),
        mc_seed=int(CFG["baseline_mc_seed"]),          # FIXED seed for cache reuse
        cache_tables=bool(CFG["cache_tables"]),
        out_dir=run_with_dir,
        baseline_cache_dir=baseline_cache_dir,         # shared cache dir
    )
    log(f"mc_reps={engine.mc_reps}, ref_sample_size={engine.ref_sample_size}, mc_seed={engine.mc_seed}, cache_tables={engine.cache_tables}")
    run_evidence_engine(run_with_dir, engine)

    step("EVIDENCE — NO-coal run")
    engine2 = EngineParams(
        H_plat_spec=H_plat_spec,
        ref_sample_size=int(CFG["ref_sample_size"]),
        mc_reps=int(CFG["mc_reps"]),
        mc_seed=int(CFG["baseline_mc_seed"]),          # same fixed seed
        cache_tables=bool(CFG["cache_tables"]),
        out_dir=run_nocoal_dir,
        baseline_cache_dir=baseline_cache_dir,         # shared cache dir
    )
    log(f"mc_reps={engine2.mc_reps}, ref_sample_size={engine2.ref_sample_size}, mc_seed={engine2.mc_seed}, cache_tables={engine2.cache_tables}")
    run_evidence_engine(run_nocoal_dir, engine2)

    # load increments + labels
    evidence_with = os.path.join(run_with_dir, "evidence.csv")
    evidence_nc = os.path.join(run_nocoal_dir, "evidence.csv")
    meta_with = os.path.join(run_with_dir, "meta.json")
    meta_nc = os.path.join(run_nocoal_dir, "meta.json")
    exp_with = os.path.join(run_with_dir, "exposures_by_t.npz")
    exp_nc = os.path.join(run_nocoal_dir, "exposures_by_t.npz")
    for p in [evidence_with, evidence_nc, meta_with, meta_nc, exp_with, exp_nc]:
        _require(p)

    y_with = load_labels(meta_with)
    y_nc = load_labels(meta_nc)
    if len(y_with) != len(y_nc):
        raise ValueError("WITH and NO-coal runs have different N_total in meta.json (unexpected).")
    if not np.all(y_with == y_nc):
        log("[warn] coalition membership arrays differ between runs; using WITH labels for grouping.")

    # d_t is w__ours_w1 (already centered by evidence engine)
    w_with, T_with = load_w_series(evidence_with, "w__ours_w1")
    w_nc, T_nc = load_w_series(evidence_nc, "w__ours_w1")
    T = min(T_with, T_nc)
    w_with = w_with[:T]
    w_nc = w_nc[:T]

    # smooth
    d_with_s = smooth_moving_average(w_with, int(CFG["smooth_w"]))
    d_nc_s = smooth_moving_average(w_nc, int(CFG["smooth_w"]))

    # bands (NPZ exposures)
    step("BANDS — WITH run (coalition vs normal)")
    bands_with = compute_score_bands_from_exposures_npz(
        exposures_npz=exp_with,
        y_is_coal=y_with,
        w_t=w_with,
        T=T,
        q_lo=float(CFG["q_lo"]),
        q_hi=float(CFG["q_hi"]),
        want_coal_and_norm=True,
    )

    step("BANDS — NO-coal run (normal only)")
    bands_nc = compute_score_bands_from_exposures_npz(
        exposures_npz=exp_nc,
        y_is_coal=y_with,
        w_t=w_nc,
        T=T,
        q_lo=float(CFG["q_lo"]),
        q_hi=float(CFG["q_hi"]),
        want_coal_and_norm=False,
    )

    # write experiment CSV
    _ensure_dir(os.path.dirname(experiment_csv) or ".")
    step("WRITE — fig1_experiment.csv")
    with open(experiment_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "d_with", "d_with_smooth",
            "d_nocoal", "d_nocoal_smooth",
            "coal_med", "coal_qlo", "coal_qhi",
            "norm_med", "norm_qlo", "norm_qhi",
            "norm_nocoal_med", "norm_nocoal_qlo", "norm_nocoal_qhi",
        ])
        for t in range(1, T + 1):
            i = t - 1
            w.writerow([
                t,
                float(w_with[i]), float(d_with_s[i]),
                float(w_nc[i]), float(d_nc_s[i]),
                float(bands_with["coal_med"][i]), float(bands_with["coal_qlo"][i]), float(bands_with["coal_qhi"][i]),
                float(bands_with["norm_med"][i]), float(bands_with["norm_qlo"][i]), float(bands_with["norm_qhi"][i]),
                float(bands_nc["norm_med"][i]), float(bands_nc["norm_qlo"][i]), float(bands_nc["norm_qhi"][i]),
            ])

    log(f"[experiment] wrote -> {experiment_csv}")
    log(f"[experiment] mean(d_t) with coalition: {float(np.mean(w_with)):.6g}")
    log(f"[experiment] mean(d_t) no coalition:   {float(np.mean(w_nc)):.6g}")

    return {
        "T": T,
        "N_total": int(len(y_with)),
        "coalition_size": int(np.sum(y_with == 1)),
        "normal_size": int(np.sum(y_with == 0)),
        "experiment_csv": os.path.abspath(experiment_csv),
        "run_with_dir": os.path.abspath(run_with_dir),
        "run_nocoal_dir": os.path.abspath(run_nocoal_dir),
    }


# ============================================================
# 3) Plot from experiment CSV (like old)
# ============================================================
def plot_figure_from_experiment_csv(experiment_csv: str, out_png: str, out_pdf: str) -> None:
    _require(experiment_csv)

    t: List[int] = []
    cols: Dict[str, List[float]] = {
        "d_with_smooth": [],
        "d_nocoal_smooth": [],
        "coal_med": [], "coal_qlo": [], "coal_qhi": [],
        "norm_med": [], "norm_qlo": [], "norm_qhi": [],
        "norm_nocoal_med": [], "norm_nocoal_qlo": [], "norm_nocoal_qhi": [],
    }

    with open(experiment_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(int(row["t"]))
            for k in cols.keys():
                cols[k].append(float(row[k]))

    tt = np.asarray(t, dtype=np.int64)

    step("PLOT — Fig.1")
    fig, ax1 = plt.subplots(figsize=(9.2, 4.8))
    ax2 = ax1.twinx()

    ax1.set_ylim(-0.2, 2.8)
    ax1.set_xlim(-20, int(tt[-1]) + 20)
    if int(tt[-1]) >= 4000:
        ax1.set_xticks([0, 1000, 2000, 3000, 4000])

    ax2.set_ylim(-0.002, 0.02)
    ax2.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020])

    ax1.fill_between(tt, cols["coal_qlo"], cols["coal_qhi"], alpha=0.15)
    ax1.plot(tt, cols["coal_med"], lw=2.8, label="Coalition (with coalition)")

    ax1.fill_between(tt, cols["norm_qlo"], cols["norm_qhi"], alpha=0.15)
    ax1.plot(tt, cols["norm_med"], lw=2.8, label="Normal (with coalition)")

    ax1.fill_between(tt, cols["norm_nocoal_qlo"], cols["norm_nocoal_qhi"], alpha=0.15)
    ax1.plot(tt, cols["norm_nocoal_med"], lw=2.8, label="Normal (no coalition)")

    ax2.plot(tt, cols["d_with_smooth"], ":", lw=2.2, label=r"$d_t$ with coalition (smoothed)")
    #ax2.plot(tt, cols["d_nocoal_smooth"], ":", lw=2.2, label=r"$d_t$ no coalition (smoothed)")
    ax2.plot(
    tt, cols["d_nocoal_smooth"],
    ":", lw=2.2, color="tab:green",
    label=r"$d_t$ no coalition (smoothed)"
    )

    ax1.set_xlabel(r"Global interval index $t$")
    ax1.set_ylabel(r"Cumulative divergence score (median / 5--95\%)")
    ax2.set_ylabel(r"Interval-level deviation $d_t$")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(
        h1 + h2, l1 + l2,
        loc="upper left",
        frameon=True,
        fontsize=11,
        handlelength=2.4,
        borderpad=0.6,
        labelspacing=0.4,
    )
    leg.get_frame().set_alpha(0.95)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log(f"[plot] saved PNG -> {out_png}")
    log(f"[plot] saved PDF -> {out_pdf}")
    log("[plot] done.")


# ============================================================
# Main
# ============================================================
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot_only", action="store_true", default=False,
                    help="Only plot from fig1_experiment.csv (no regeneration).")

    # overrides without editing CFG
    ap.add_argument("--seed_with", type=int, default=int(CFG["seed_with"]))
    ap.add_argument("--seed_nocoal", type=int, default=int(CFG["seed_nocoal"]))
    ap.add_argument("--delta_mu", type=float, default=float(CFG["delta_mu"]))
    ap.add_argument("--p_on", type=float, default=float(CFG["p_on"]))
    ap.add_argument("--R_exp", type=float, default=float(CFG["R_exp_target"]))
    ap.add_argument("--mc_reps", type=int, default=int(CFG["mc_reps"]))
    ap.add_argument("--ref_sample_size", type=int, default=int(CFG["ref_sample_size"]))
    ap.add_argument("--baseline_mc_seed", type=int, default=int(CFG["baseline_mc_seed"]))
    ap.add_argument("--no_cache", action="store_true", default=False)

    args = ap.parse_args(argv)

    # apply overrides into CFG (in-memory)
    CFG["seed_with"] = int(args.seed_with)
    CFG["seed_nocoal"] = int(args.seed_nocoal)
    CFG["delta_mu"] = float(args.delta_mu)
    CFG["p_on"] = float(args.p_on)
    CFG["R_exp_target"] = float(args.R_exp)
    CFG["mc_reps"] = int(args.mc_reps)
    CFG["ref_sample_size"] = int(args.ref_sample_size)
    CFG["baseline_mc_seed"] = int(args.baseline_mc_seed)
    CFG["cache_tables"] = bool(not args.no_cache)

    data_dir = os.path.join(THIS_DIR, "data")
    fig_dir = os.path.join(THIS_DIR, "figure")
    baseline_cache_dir = os.path.join(data_dir, "baseline_cache")

    _ensure_dir(data_dir)
    _ensure_dir(fig_dir)
    _ensure_dir(baseline_cache_dir)

    run_with_dir = os.path.join(data_dir, "with_coalition")
    run_nocoal_dir = os.path.join(data_dir, "no_coalition")
    experiment_csv = os.path.join(data_dir, "fig1_experiment.csv")

    out_png = os.path.join(fig_dir, "fig1_global_divergence.png")
    out_pdf = os.path.join(fig_dir, "fig1_global_divergence.pdf")
    config_json = os.path.join(fig_dir, "fig1_config.json")

    step("FIG1 START")
    log(f"Mode: {'PLOT-ONLY' if args.plot_only else 'FULL'}")
    log(f"mc_reps={CFG['mc_reps']}, ref_sample_size={CFG['ref_sample_size']}, baseline_mc_seed={CFG['baseline_mc_seed']}")
    log(f"cache_tables={CFG['cache_tables']}  baseline_cache_dir={os.path.abspath(baseline_cache_dir)}")
    log(f"run_with_dir:   {os.path.abspath(run_with_dir)}")
    log(f"run_nocoal_dir: {os.path.abspath(run_nocoal_dir)}")
    log(f"experiment_csv: {os.path.abspath(experiment_csv)}")

    if not args.plot_only:
        write_config_json(config_json)

        gen_info = generate_two_runs(run_with_dir, run_nocoal_dir)
        exp_info = run_experiment_from_runs(
            run_with_dir, run_nocoal_dir, experiment_csv,
            baseline_cache_dir=baseline_cache_dir
        )

        # manifest
        manifest_path = os.path.join(fig_dir, "fig1_manifest.json")
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

    plot_figure_from_experiment_csv(experiment_csv, out_png, out_pdf)

    step("FIG1 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
