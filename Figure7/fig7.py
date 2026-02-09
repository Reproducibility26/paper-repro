#!/usr/bin/env python3
"""
fig7.py — Figure 7: Rotation-neutralization of nonnegative discrepancy baselines.

Purpose:
  This figure demonstrates that any nonnegative discrepancy statistic (e.g., JS or Chi-square)
  exhibits positive drift under the null when accumulated naively, and that baseline correction
  removes this drift. The structure mirrors Fig. 2, but swaps W1 for a generic discrepancy φ.

Default baseline:
  - φ = bl_js (Jensen–Shannon divergence)
    * uncorrected accumulation uses score__bl_js
    * corrected accumulation uses w__bl_js

Optional baseline:
  - φ = bl_chi2 (Chi-square divergence), via --phi bl_chi2

Pipeline:
  1) Generate two synthetic runs under the rotation model:
       - WITH coalition: intermittent activity (p_on > 0) with enforced exposure ratio.
       - NO-coal control: p_on = 0, k_on = 0.
     Each run produces meta.json, intervals.csv, and exposures_by_t.npz.
  2) Compute interval-level evidence using Script #2, producing score__φ and w__φ.
  3) Accumulate per-account scores using exposures_by_t.npz and plot median and
     5–95% bands for coalition vs normal users, comparing uncorrected vs corrected curves.

Outputs (relative to this file):
  - Data:      ./data/fig7_<phi>_experiment.csv
  - Figures:   ./figure/fig7_<phi>_baseline_correction_ablation.{png,pdf}
  - Metadata:  ./figure/fig7_<phi>_{config,manifest}.json

Dependencies (repo root on PYTHONPATH):
  - rotation_generator.py        (generate_rotation_dataset)
  - evidence_and_baselines.py    (run_evidence_engine)
  - exposures_by_t.npz format    (CSR-by-interval exposure representation)

Usage:
  # Full pipeline (JS baseline)
  python fig7.py

  # Use Chi-square baseline instead
  python fig7.py --phi bl_chi2

  # Plot-only mode (reuse existing experiment CSV)
  python fig7.py --plot_only

  # Common parameter overrides
  python fig7.py --p_on 0.3 --R_exp 1.0 --delta_mu 0.02
  python fig7.py --mc_reps 300 --ref_sample_size 200000 --baseline_mc_seed 999

Notes:
  - Baseline Monte Carlo tables are cached under ./data/baseline_cache/ and reused
    across runs.
  - Under the NO-coal control, the mean of the corrected increments w__φ should be
    approximately zero, while the raw score__φ typically has positive mean.

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Path setup: script in Figure18, modules in parent dir
# -------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if A_DIR not in sys.path:
    sys.path.insert(0, A_DIR)


# -------------------------
# Imports (new oracle stack)
# -------------------------
from rotation_generator import RotationParams, generate_rotation_dataset
from evidence_and_baselines import EngineParams, run_evidence_engine


# -------------------------
# Logging (fig5 style)
# -------------------------
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


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# -------------------------
# Config (mirrors old script; updated for NPZ exposures)
# -------------------------
CFG: Dict[str, Any] = dict(
    figure="fig7_phi_baseline_correction_ablation",

    # timeline / population
    T=4000,
    N_norm=20000,
    N_coal=2000,

    # behavior (intermittency)
    p_norm=0.04,
    p_on=0.20,
    R_exp_target=1.0,

    # histogram support
    bins=50,
    x_min=0.0,
    x_max=1.0,

    # signal model
    mu_norm=0.50,
    var_norm=0.05,
    delta_mu=0.02,  # small shift OK; this figure is about centering

    # evidence baseline MC (shared cache across runs)
    mc_reps=300,
    ref_sample_size=200_000,
    cache_tables=True,
    baseline_mc_seed=999,

    # plotting / smoothing
    smooth_w=25,
    q_lo=0.05,
    q_hi=0.95,

    # seeds (generator)
    seed_with=10,
    seed_nocoal=11,

    # which discrepancy baseline to show
    phi="bl_js",
)


def write_config_json(path: str) -> None:
    cfg = dict(CFG)

    p_norm = float(cfg["p_norm"])
    p_on = float(cfg["p_on"])
    N_coal = int(cfg["N_coal"])
    R_exp = float(cfg["R_exp_target"])

    if p_on <= 0:
        k_on = 0
        realized = 0.0
    else:
        k_on = int(round((R_exp * p_norm * N_coal) / max(p_on, 1e-12)))
        k_on = max(1, min(N_coal, k_on))
        realized = (p_on * k_on / N_coal) / max(p_norm, 1e-12)

    cfg["derived_k_on"] = int(k_on)
    cfg["derived_R_exp_realized"] = float(realized)
    cfg["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cfg["script"] = os.path.basename(__file__)

    _write_json(path, cfg)
    log(f"[config] wrote -> {path}")


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


# -------------------------
# NEW: exposures_by_t.npz loader + cumulative band computation
# -------------------------
def load_exposures_npz(exposures_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    exposures_by_t.npz contains:
      t_ptr: int64 length T+1
      u_ids: int32 length nnz
      a_ut : float32/float64 length nnz
    """
    _require(exposures_path)
    z = np.load(exposures_path, allow_pickle=False)
    if "t_ptr" not in z or "u_ids" not in z or "a_ut" not in z:
        raise ValueError("exposures_by_t.npz missing required arrays: t_ptr, u_ids, a_ut")

    t_ptr = np.asarray(z["t_ptr"], dtype=np.int64)
    u_ids = np.asarray(z["u_ids"], dtype=np.int32)
    a_ut = np.asarray(z["a_ut"], dtype=np.float64)

    if t_ptr.ndim != 1 or u_ids.ndim != 1 or a_ut.ndim != 1:
        raise ValueError("exposures_by_t.npz arrays must be 1D")
    if u_ids.shape[0] != a_ut.shape[0]:
        raise ValueError("exposures_by_t.npz: u_ids and a_ut must have same length")
    if t_ptr.shape[0] < 2:
        raise ValueError("exposures_by_t.npz: t_ptr must have length T+1 >= 2")
    if t_ptr[0] != 0 or t_ptr[-1] != u_ids.shape[0]:
        raise ValueError("exposures_by_t.npz: bad t_ptr endpoints")
    if np.any(t_ptr[1:] < t_ptr[:-1]):
        raise ValueError("exposures_by_t.npz: t_ptr must be nondecreasing")

    return t_ptr, u_ids, a_ut


def compute_bands_from_exposures(
    exposures_npz: str,
    y_is_coal: np.ndarray,
    w_t: np.ndarray,
    T: int,
    q_lo: float,
    q_hi: float,
    want_coal_and_norm: bool,
    progress_every: int = 250,
) -> Dict[str, np.ndarray]:
    """
    scores[u] += a_ut * w_t[t-1], then record quantile bands at each t.
    Uses exposures_by_t.npz sparse-by-interval CSR format.
    """
    t_ptr, u_ids, a_ut = load_exposures_npz(exposures_npz)

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

    # exposures T can differ from evidence T; use overlap
    T_exp = int(t_ptr.shape[0] - 1)
    T_use = int(min(T, T_exp))

    for t in range(1, T_use + 1):
        if (t % progress_every) == 0 or t == 1 or t == T_use:
            log(f"  bands progress t={t}/{T_use}")

        # update scores for participants in this interval
        start = int(t_ptr[t - 1])
        end = int(t_ptr[t])
        if end > start:
            wt = float(w_t[t - 1])
            if wt != 0.0:
                users = u_ids[start:end]
                weights = a_ut[start:end]
                for u, a in zip(users, weights):
                    scores[int(u)] += float(a) * wt

        # record bands at time t
        i = t - 1
        if want_coal_and_norm:
            cm, cl, ch = quantile_bands(scores[coal_idx], lo=q_lo, hi=q_hi)
            nm, nl, nh = quantile_bands(scores[norm_idx], lo=q_lo, hi=q_hi)
            coal_med[i], coal_qlo[i], coal_qhi[i] = cm, cl, ch
            norm_med[i], norm_qlo[i], norm_qhi[i] = nm, nl, nh
        else:
            nm, nl, nh = quantile_bands(scores[norm_idx], lo=q_lo, hi=q_hi)
            norm_med[i], norm_qlo[i], norm_qhi[i] = nm, nl, nh

    # if evidence T > exposures T, fill remaining with last value (mostly for robustness)
    if T > T_use:
        for t in range(T_use + 1, T + 1):
            i = t - 1
            if want_coal_and_norm:
                coal_med[i], coal_qlo[i], coal_qhi[i] = coal_med[T_use - 1], coal_qlo[T_use - 1], coal_qhi[T_use - 1]
                norm_med[i], norm_qlo[i], norm_qhi[i] = norm_med[T_use - 1], norm_qlo[T_use - 1], norm_qhi[T_use - 1]
            else:
                norm_med[i], norm_qlo[i], norm_qhi[i] = norm_med[T_use - 1], norm_qlo[T_use - 1], norm_qhi[T_use - 1]

    if want_coal_and_norm:
        return dict(
            coal_med=coal_med, coal_qlo=coal_qlo, coal_qhi=coal_qhi,
            norm_med=norm_med, norm_qlo=norm_qlo, norm_qhi=norm_qhi,
        )
    else:
        return dict(norm_med=norm_med, norm_qlo=norm_qlo, norm_qhi=norm_qhi)


# ============================================================
# STEP 1) Generate data (two run dirs)
# ============================================================
def generate_two_runs(run_with_dir: str, run_nocoal_dir: str) -> Dict[str, Any]:
    T = int(CFG["T"])
    N_norm = int(CFG["N_norm"])
    N_coal = int(CFG["N_coal"])
    p_norm = float(CFG["p_norm"])
    p_on = float(CFG["p_on"])
    Rexp = float(CFG["R_exp_target"])

    # Enforce exposure ratio: R_exp = (p_on*k_on/N_coal)/p_norm
    if p_on <= 0:
        k_on = 0
        realized = 0.0
    else:
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
        k_on=int(k_on),
        bins=bins,
        x_min=x_min,
        x_max=x_max,
        seed=int(CFG["seed_with"]),
        D_norm=D_norm,
        D_coal=D_coal,
        actions_per_participant_norm=1,
        actions_per_participant_coal=1,
    )

    # No-coal control: p_on=0, k_on=0
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

    step("FIG7 — STEP 1A: GENERATE (WITH coalition)")
    log(f"run_dir={run_with_dir}")
    log(f"k_on={k_on}  target_Rexp={Rexp:.3g}  realized_Rexp={realized:.3g}  p_on={p_on}")
    generate_rotation_dataset(params_with, out_dir=run_with_dir, write_participation_csv=False)

    step("FIG7 — STEP 1B: GENERATE (NO-coal control)")
    log(f"run_dir={run_nocoal_dir}")
    generate_rotation_dataset(params_nocoal, out_dir=run_nocoal_dir, write_participation_csv=False)

    return {
        "k_on": int(k_on),
        "Rexp_realized": float(realized),
        "D_norm": D_norm,
        "D_coal": D_coal,
        "params_with": asdict(params_with),
        "params_nocoal": asdict(params_nocoal),
    }


# ============================================================
# STEP 2) Evidence + bands (chosen phi)
# ============================================================
def run_experiment(
    run_with_dir: str,
    run_nocoal_dir: str,
    out_experiment_csv: str,
    baseline_cache_dir: str,
    phi: str,
) -> Dict[str, Any]:
    score_col = f"score__{phi}"
    w_col = f"w__{phi}"

    H_plat_spec = {
        "type": "normal",
        "mu": float(CFG["mu_norm"]),
        "sigma": float(np.sqrt(float(CFG["var_norm"]))),
    }

    _ensure_dir(baseline_cache_dir)

    step(f"FIG7 — STEP 2A: EVIDENCE (WITH run, phi={phi})")
    eng_with = EngineParams(
        H_plat_spec=H_plat_spec,
        ref_sample_size=int(CFG["ref_sample_size"]),
        mc_reps=int(CFG["mc_reps"]),
        mc_seed=int(CFG["baseline_mc_seed"]),
        cache_tables=bool(CFG["cache_tables"]),
        out_dir=run_with_dir,
        baseline_cache_dir=baseline_cache_dir,
    )
    run_evidence_engine(run_with_dir, eng_with)

    step(f"FIG7 — STEP 2B: EVIDENCE (NO-coal run, phi={phi})")
    eng_nc = EngineParams(
        H_plat_spec=H_plat_spec,
        ref_sample_size=int(CFG["ref_sample_size"]),
        mc_reps=int(CFG["mc_reps"]),
        mc_seed=int(CFG["baseline_mc_seed"]),
        cache_tables=bool(CFG["cache_tables"]),
        out_dir=run_nocoal_dir,
        baseline_cache_dir=baseline_cache_dir,
    )
    run_evidence_engine(run_nocoal_dir, eng_nc)

    # Required files
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

    # Interval series
    X_with, T1 = load_series_from_evidence(ev_with, score_col)
    d_with, T2 = load_series_from_evidence(ev_with, w_col)
    X_nc, T3 = load_series_from_evidence(ev_nc, score_col)
    d_nc, T4 = load_series_from_evidence(ev_nc, w_col)

    T = min(T1, T2, T3, T4)
    X_with = X_with[:T]
    d_with = d_with[:T]
    X_nc = X_nc[:T]
    d_nc = d_nc[:T]

    # Smoothed interval-level curves (optional)
    w_smooth = int(CFG["smooth_w"])
    X_with_s = smooth_moving_average(X_with, w_smooth)
    X_nc_s = smooth_moving_average(X_nc, w_smooth)
    d_with_s = smooth_moving_average(d_with, w_smooth)
    d_nc_s = smooth_moving_average(d_nc, w_smooth)

    qlo = float(CFG["q_lo"])
    qhi = float(CFG["q_hi"])

    step("FIG7 — STEP 2C: BANDS (Uncorrected accumulation)")
    bands_X_with = compute_bands_from_exposures(
        exposures_npz=exp_with,
        y_is_coal=y_with,
        w_t=X_with,
        T=T,
        q_lo=qlo,
        q_hi=qhi,
        want_coal_and_norm=True,
        progress_every=500,
    )
    bands_X_nc = compute_bands_from_exposures(
        exposures_npz=exp_nc,
        y_is_coal=y_with,  # same indexing
        w_t=X_nc,
        T=T,
        q_lo=qlo,
        q_hi=qhi,
        want_coal_and_norm=False,
        progress_every=500,
    )

    step("FIG7 — STEP 2D: BANDS (Corrected accumulation)")
    bands_d_with = compute_bands_from_exposures(
        exposures_npz=exp_with,
        y_is_coal=y_with,
        w_t=d_with,
        T=T,
        q_lo=qlo,
        q_hi=qhi,
        want_coal_and_norm=True,
        progress_every=500,
    )
    bands_d_nc = compute_bands_from_exposures(
        exposures_npz=exp_nc,
        y_is_coal=y_with,
        w_t=d_nc,
        T=T,
        q_lo=qlo,
        q_hi=qhi,
        want_coal_and_norm=False,
        progress_every=500,
    )

    # Write experiment CSV
    step(f"FIG7 — STEP 2E: WRITE EXPERIMENT CSV ({os.path.basename(out_experiment_csv)})")
    _ensure_dir(os.path.dirname(out_experiment_csv) or ".")
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
    log(f"[sanity] mean(raw score) NO-coal: {float(np.mean(X_nc)):.6g}   (typically > 0 for nonnegative phi)")
    log(f"[sanity] mean(centered w_t) NO-coal: {float(np.mean(d_nc)):.6g}   (should be ~0)")

    return {"T": int(T), "experiment_csv": os.path.abspath(out_experiment_csv), "phi": phi}


# ============================================================
# STEP 3) Plot
# ============================================================
def plot_from_experiment_csv(experiment_csv: str, out_png: str, out_pdf: str, phi: str) -> None:
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

    step(f"FIG7 — STEP 3/3: PLOT (phi={phi})")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))

    # Match your earlier palette: uncorrected vs corrected
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
    ax.set_ylabel(rf"Cumulative score (raw vs centered {phi})")
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
    ap = argparse.ArgumentParser(description="Fig7: ablation for nonnegative discrepancy baseline (JS or chi2)")
    ap.add_argument("--plot_only", action="store_true", default=False)
    ap.add_argument("--phi", type=str, default=str(CFG["phi"]), choices=["bl_js", "bl_chi2"])

    # common overrides
    ap.add_argument("--mc_reps", type=int, default=int(CFG["mc_reps"]))
    ap.add_argument("--ref_sample_size", type=int, default=int(CFG["ref_sample_size"]))
    ap.add_argument("--baseline_mc_seed", type=int, default=int(CFG["baseline_mc_seed"]))
    ap.add_argument("--no_cache", action="store_true", default=False)
    ap.add_argument("--delta_mu", type=float, default=float(CFG["delta_mu"]))
    ap.add_argument("--p_on", type=float, default=float(CFG["p_on"]))
    ap.add_argument("--R_exp", type=float, default=float(CFG["R_exp_target"]))

    args = ap.parse_args(argv)

    CFG["phi"] = str(args.phi)
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

    phi = str(CFG["phi"])
    exp_csv = os.path.join(data_dir, f"fig7_{phi}_experiment.csv")

    out_png = os.path.join(fig_dir, f"fig7_{phi}_baseline_correction_ablation.png")
    out_pdf = os.path.join(fig_dir, f"fig7_{phi}_baseline_correction_ablation.pdf")
    config_json = os.path.join(fig_dir, f"fig7_{phi}_config.json")
    manifest_json = os.path.join(fig_dir, f"fig7_{phi}_manifest.json")

    step("FIG7 START")
    log(f"Mode: {'PLOT-ONLY' if args.plot_only else 'FULL'}")
    log(f"phi={phi}")
    log(f"mc_reps={CFG['mc_reps']}  ref_sample_size={CFG['ref_sample_size']}  baseline_mc_seed={CFG['baseline_mc_seed']}")
    log(f"cache_tables={CFG['cache_tables']}  baseline_cache_dir={os.path.abspath(baseline_cache_dir)}")
    log(f"p_on={CFG['p_on']}  R_exp_target={CFG['R_exp_target']}  delta_mu={CFG['delta_mu']}")
    log(f"experiment_csv={os.path.abspath(exp_csv)}")

    gen_info = None
    exp_info = None

    if not args.plot_only:
        write_config_json(config_json)

        gen_info = generate_two_runs(run_with_dir, run_nocoal_dir)

        step("FIG7 — STEP 2/3: EVIDENCE + EXPERIMENT")
        exp_info = run_experiment(
            run_with_dir=run_with_dir,
            run_nocoal_dir=run_nocoal_dir,
            out_experiment_csv=exp_csv,
            baseline_cache_dir=baseline_cache_dir,
            phi=phi,
        )

        manifest = {
            "CFG": dict(CFG),
            "generator": gen_info,
            "experiment": exp_info,
            "baseline_cache_dir": os.path.abspath(baseline_cache_dir),
            "runs": {"with": os.path.abspath(run_with_dir), "nocoal": os.path.abspath(run_nocoal_dir)},
            "outputs": {"png": os.path.abspath(out_png), "pdf": os.path.abspath(out_pdf), "experiment_csv": os.path.abspath(exp_csv)},
        }
        _write_json(manifest_json, manifest)
        log(f"[manifest] wrote -> {manifest_json}")

    plot_from_experiment_csv(exp_csv, out_png, out_pdf, phi=phi)

    step("FIG7 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
