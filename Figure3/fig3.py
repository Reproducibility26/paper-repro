#!/usr/bin/env python3
"""
fig3.py — Figure 3: Attribution accuracy (ROC-AUC) vs exposure ratio $R_{\exp}$.

What this script does:
  For each exposure ratio R_exp on a grid, and for multiple random seeds:
    1) Generate one synthetic run under the rotation model with intermittent coalition activity.
    2) Compute evidence increments for each method (Script #2) using the same baseline cache.
    3) Run exposure-based attribution (Script #3) to produce per-account scores S_m(u).
    4) Evaluate ROC-AUC for each method (Script #4).
  It then aggregates ROC-AUC over seeds to produce mean and 5–95% bands as a function of R_exp.

Methods compared (must exist as columns w__<method> in evidence.csv):
  - ours_w1 (Geo-W1)
  - bl_intensity_mean_shift
  - bl_mean_zscore
  - bl_variance_normalized
  - bl_participation_frequency
  - bl_js
  - bl_chi2
  - bl_energy

Inputs / outputs (relative to this file):
  - Run directories:        ./data/runs/Rexp_<val>/run_<idx>/
  - Shared baseline cache:  ./data/baseline_cache/
  - Experiment cache CSV:   ./data/fig3_experiment.csv
  - Config / manifest:      ./data/fig3_config.json, ./figure/fig3_manifest.json
  - Figures:                ./figure/fig3_auc_vs_rexp.{png,pdf}

Dependencies (repo root on PYTHONPATH):
  - rotation_generator.py       (generate_rotation_dataset)
  - evidence_and_baselines.py   (run_evidence_engine)
  - exposure_attribution.py     (run_attribution; reads exposures_by_t.npz)
  - metric_eval.py              (run_metrics)

Usage:
  # Full sweep: generate runs, compute evidence, attribution, metrics, then plot
  python fig3.py

  # Plot-only mode (reuse existing fig3_experiment.csv)
  python fig3.py --plot_only

  # Common parameter overrides
  python fig3.py --runs 10 --R_exp_points 20 --R_exp_min 0.1 --R_exp_max 2.0
  python fig3.py --p_on 1.0 --delta_mu 0.05
  python fig3.py --mc_reps 250 --ref_sample_size 200000 --baseline_mc_seed 999

Notes:
  - Baseline Monte Carlo tables are cached under ./data/baseline_cache/ and reused
    across all runs in the sweep to reduce total runtime.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


# ------------------------------------------------------------
# Path setup: script in Figure3, modules in parent dir
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if A_DIR not in sys.path:
    sys.path.insert(0, A_DIR)


# ------------------------------------------------------------
# Logging
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


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


# ------------------------------------------------------------
# Config (override via CLI)
# ------------------------------------------------------------
CFG: Dict[str, Any] = dict(
    figure="fig3_auc_vs_rexp",

    # timeline / population
    T=4000,
    N_norm=20000,
    N_coal=2000,

    # behavior
    p_norm=0.04,
    p_on=1.00,              # coalition ON probability
    R_exp_min=0.10,
    R_exp_max=2.00,
    R_exp_points=20,

    # histogram support
    bins=50,
    x_min=0.0,
    x_max=1.0,

    # normal model + attack
    mu_norm=0.50,
    var_norm=0.05,
    delta_mu=0.05,

    # baseline correction MC
    mc_reps=250,
    ref_sample_size=200_000,
    cache_tables=True,

    # repeats
    runs=10,
    seed_base=20260101,

    # plot summary band
    q_lo=0.05,
    q_hi=0.95,

    # baseline cache seed (keep fixed for reproducibility / reuse)
    baseline_mc_seed=999,
)


METHODS: List[str] = [
    "ours_w1",
    "bl_intensity_mean_shift",
    "bl_mean_zscore",
    "bl_variance_normalized",
    "bl_participation_frequency",
    "bl_js",
    "bl_chi2",
    "bl_energy",
]

METHOD_LABEL: Dict[str, str] = {
    "ours_w1": "Geo-W1",
    "bl_intensity_mean_shift": "Intensity",
    "bl_mean_zscore": "MeanZ",
    "bl_variance_normalized": "Variance",
    "bl_participation_frequency": "Freq",
    "bl_js": "JS",
    "bl_chi2": r"$\chi^2$",
    "bl_energy": "Energy",
}


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _quantile_band(x: np.ndarray, qlo: float, qhi: float) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x)), float(np.quantile(x, qlo)), float(np.quantile(x, qhi))


def enforce_k_on(Rexp: float, p_norm: float, N_coal: int, p_on: float) -> int:
    """k_on = round(Rexp * p_norm * N_coal / p_on), clamped to [1, N_coal] if p_on>0 else 0."""
    if p_on <= 0:
        return 0
    k = int(round((float(Rexp) * float(p_norm) * float(N_coal)) / max(float(p_on), 1e-12)))
    return max(1, min(int(N_coal), k))


def realized_Rexp(p_on: float, k_on: int, N_coal: int, p_norm: float) -> float:
    if p_norm <= 0:
        return float("nan")
    return (float(p_on) * float(k_on) / float(N_coal)) / float(p_norm)


def _assert_evidence_columns(evidence_csv: str, methods: List[str]) -> None:
    with open(evidence_csv, "r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    need = [f"w__{m}" for m in methods]
    missing = [c for c in need if c not in header]
    if missing:
        raise RuntimeError(f"Missing required evidence columns in {evidence_csv}: {missing}")


def load_roc_auc_from_metrics_json(metrics_json: str) -> Dict[str, float]:
    """
    metric_eval.py writes:
      {"methods": {"m": {"roc_auc": ... , ...}, ...}}
    """
    obj = _read_json(metrics_json)
    methods_block = obj.get("methods", None)
    if isinstance(methods_block, dict):
        out: Dict[str, float] = {}
        for m, md in methods_block.items():
            if isinstance(md, dict) and ("roc_auc" in md):
                out[str(m)] = float(md["roc_auc"])
        if out:
            return out
    keys = list(obj.keys()) if isinstance(obj, dict) else ["<non-dict-json>"]
    raise ValueError(f"Unrecognized metrics.json format: {metrics_json}. Top-level keys: {keys}")


# ------------------------------------------------------------
# Per-run pipeline
# ------------------------------------------------------------
def run_one(Rexp: float, run_idx: int, out_dir: str, baseline_cache_dir: str) -> Dict[str, Any]:
    from rotation_generator import RotationParams, generate_rotation_dataset
    from evidence_and_baselines import EngineParams, run_evidence_engine
    from exposure_attribution import AttributionParams, run_attribution
    from metric_eval import MetricsParams, run_metrics

    _ensure_dir(out_dir)
    _ensure_dir(baseline_cache_dir)

    seed = int(CFG["seed_base"]) + int(run_idx) + int(round(1_000_000 * float(Rexp)))
    rng_tag = f"Rexp={Rexp:.3f}, run={run_idx}, seed={seed}"
    log(f"[run] {rng_tag}")

    # Generator settings
    T = int(CFG["T"])
    N_norm = int(CFG["N_norm"])
    N_coal = int(CFG["N_coal"])
    p_norm = float(CFG["p_norm"])
    p_on = float(CFG["p_on"])

    k_on = enforce_k_on(Rexp, p_norm, N_coal, p_on)
    R_real = realized_Rexp(p_on, k_on, N_coal, p_norm)

    mu_norm = float(CFG["mu_norm"])
    sigma_norm = float(math.sqrt(float(CFG["var_norm"])))
    delta_mu = float(CFG["delta_mu"])

    bins = int(CFG["bins"])
    x_min = float(CFG["x_min"])
    x_max = float(CFG["x_max"])

    D_norm = {"type": "normal", "mu": mu_norm, "sigma": sigma_norm}
    D_coal = {"type": "normal_shift", "delta": delta_mu}

    gen_params = RotationParams(
        N_norm=N_norm,
        N_coal=N_coal,
        T=T,
        p_norm=p_norm,
        p_on=p_on,
        k_on=int(k_on),
        bins=bins,
        x_min=x_min,
        x_max=x_max,
        seed=int(seed),
        D_norm=D_norm,
        D_coal=D_coal,
        actions_per_participant_norm=1,
        actions_per_participant_coal=1,
    )

    # 1) generate
    generate_rotation_dataset(gen_params, out_dir=out_dir)

    # Optional early assert: new pipeline should write NPZ exposures
    exposures_npz = os.path.join(out_dir, "exposures_by_t.npz")
    _require(exposures_npz)

    # 2) evidence + baselines (shared cache dir)
    engine = EngineParams(
        H_plat_spec=D_norm,
        ref_sample_size=int(CFG["ref_sample_size"]),
        mc_reps=int(CFG["mc_reps"]),
        mc_seed=int(CFG["baseline_mc_seed"]),  # fixed within the figure
        cache_tables=bool(CFG["cache_tables"]),
        out_dir=out_dir,
        baseline_cache_dir=baseline_cache_dir,
    )
    run_evidence_engine(out_dir, engine)

    evidence_csv = os.path.join(out_dir, "evidence.csv")
    _require(evidence_csv)
    _assert_evidence_columns(evidence_csv, METHODS)

    # 3) attribution (Script #3 prefers exposures_by_t.npz automatically)
    attr = AttributionParams(
        out_dir=out_dir,
        methods=METHODS,
        allow_raw_score_if_missing=False,
        write_timeseries=False,
        sort_by="w__ours_w1",
        descending=True,
    )
    run_attribution(out_dir, attr)

    # 4) metrics
    met = MetricsParams(
        out_dir=out_dir,
        methods=METHODS,
        write_curves=False,
        write_ranked_lists=False,
    )
    run_metrics(out_dir, met)

    metrics_json = os.path.join(out_dir, "metrics.json")
    _require(metrics_json)
    auc_map = load_roc_auc_from_metrics_json(metrics_json)

    return {
        "Rexp_target": float(Rexp),
        "Rexp_realized": float(R_real),
        "k_on": int(k_on),
        "seed": int(seed),
        "roc_auc": {m: float(auc_map.get(m, float("nan"))) for m in METHODS},
    }


# ------------------------------------------------------------
# Experiment runner
# ------------------------------------------------------------
def run_experiment_to_csv(exp_csv: str, baseline_cache_dir: str, runs_dir: str) -> None:
    step("FIG3 — RUN EXPERIMENT")

    Rmin = float(CFG["R_exp_min"])
    Rmax = float(CFG["R_exp_max"])
    K = int(CFG["R_exp_points"])
    R_values = np.linspace(Rmin, Rmax, K, dtype=np.float64)

    qlo = float(CFG["q_lo"])
    qhi = float(CFG["q_hi"])

    all_rows: List[Dict[str, Any]] = []
    total = len(R_values) * int(CFG["runs"])
    done = 0

    for i, R in enumerate(R_values):
        step(f"Rexp sweep point {i+1}/{len(R_values)} — target Rexp={R:.3f}")
        for r in range(int(CFG["runs"])):
            done += 1
            run_dir = os.path.join(runs_dir, f"Rexp_{R:.3f}", f"run_{r:02d}")
            log(f"Progress: {done}/{total}  -> {run_dir}")
            res = run_one(float(R), r, run_dir, baseline_cache_dir=baseline_cache_dir)
            all_rows.append(res)

    # Aggregate across runs at each Rexp
    _ensure_dir(os.path.dirname(exp_csv) or ".")
    step(f"WRITE — {os.path.basename(exp_csv)}")

    by_R: Dict[float, List[Dict[str, Any]]] = {}
    for row in all_rows:
        Rt = float(row["Rexp_target"])
        by_R.setdefault(Rt, []).append(row)

    with open(exp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Rexp_target",
            "Rexp_realized_mean",
            "method",
            "roc_auc_mean",
            "roc_auc_qlo",
            "roc_auc_qhi",
            "runs",
        ])

        for Rt in sorted(by_R.keys()):
            rows = by_R[Rt]
            Rreal_vals = np.asarray([float(x["Rexp_realized"]) for x in rows], dtype=np.float64)
            Rreal_mean = float(np.mean(Rreal_vals))

            for m in METHODS:
                aucs = np.asarray([float(x["roc_auc"][m]) for x in rows], dtype=np.float64)
                mu, lo, hi = _quantile_band(aucs, qlo, qhi)
                w.writerow([f"{Rt:.6g}", f"{Rreal_mean:.6g}", m, f"{mu:.6g}", f"{lo:.6g}", f"{hi:.6g}", len(rows)])

    log(f"[ok] wrote experiment CSV -> {exp_csv}")

    cfg_path = os.path.join(os.path.dirname(exp_csv), "fig3_config.json")
    cfg_obj = dict(CFG)
    cfg_obj["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cfg_obj["script"] = os.path.basename(__file__)
    cfg_obj["baseline_cache_dir"] = os.path.abspath(baseline_cache_dir)
    _write_json(cfg_path, cfg_obj)
    log(f"[ok] wrote config JSON -> {cfg_path}")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def plot_from_experiment_csv(exp_csv: str, out_png: str, out_pdf: str) -> None:
    _require(exp_csv)

    rows: List[Dict[str, Any]] = []
    with open(exp_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    by_m: Dict[str, Dict[str, List[float]]] = {m: {"R": [], "mu": [], "lo": [], "hi": []} for m in METHODS}
    for row in rows:
        m = row["method"]
        if m not in by_m:
            continue
        by_m[m]["R"].append(float(row["Rexp_target"]))
        by_m[m]["mu"].append(float(row["roc_auc_mean"]))
        by_m[m]["lo"].append(float(row["roc_auc_qlo"]))
        by_m[m]["hi"].append(float(row["roc_auc_qhi"]))

    step("PLOT — Fig.3")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    for m in METHODS:
        R = np.asarray(by_m[m]["R"], dtype=np.float64)
        mu = np.asarray(by_m[m]["mu"], dtype=np.float64)
        lo = np.asarray(by_m[m]["lo"], dtype=np.float64)
        hi = np.asarray(by_m[m]["hi"], dtype=np.float64)

        order = np.argsort(R)
        R, mu, lo, hi = R[order], mu[order], lo[order], hi[order]

        label = METHOD_LABEL.get(m, m)
        ax.plot(R, mu, linewidth=2.6, label=label)
        ax.fill_between(R, lo, hi, alpha=0.15, linewidth=0)

    ax.set_xlabel(r"Exposure ratio $R_{\exp}$")
    ax.set_ylabel(r"ROC-AUC (coalition vs normal)")
    #ax.set_title(r"Attribution accuracy vs exposure ratio")

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0.1, 2.0)
    ax.grid(alpha=0.15)
    ax.legend(loc="lower right", ncol=2, frameon=True)

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
    ap = argparse.ArgumentParser(description="Fig3 (new oracle): ROC-AUC vs exposure ratio Rexp")
    ap.add_argument("--plot_only", action="store_true", default=False)

    # Overrides
    ap.add_argument("--runs", type=int, default=int(CFG["runs"]))
    ap.add_argument("--R_exp_points", type=int, default=int(CFG["R_exp_points"]))
    ap.add_argument("--R_exp_min", type=float, default=float(CFG["R_exp_min"]))
    ap.add_argument("--R_exp_max", type=float, default=float(CFG["R_exp_max"]))
    ap.add_argument("--p_on", type=float, default=float(CFG["p_on"]))
    ap.add_argument("--delta_mu", type=float, default=float(CFG["delta_mu"]))
    ap.add_argument("--mc_reps", type=int, default=int(CFG["mc_reps"]))
    ap.add_argument("--ref_sample_size", type=int, default=int(CFG["ref_sample_size"]))
    ap.add_argument("--baseline_mc_seed", type=int, default=int(CFG["baseline_mc_seed"]))
    ap.add_argument("--no_cache", action="store_true", default=False)

    args = ap.parse_args(argv)

    CFG["runs"] = int(args.runs)
    CFG["R_exp_points"] = int(args.R_exp_points)
    CFG["R_exp_min"] = float(args.R_exp_min)
    CFG["R_exp_max"] = float(args.R_exp_max)
    CFG["p_on"] = float(args.p_on)
    CFG["delta_mu"] = float(args.delta_mu)
    CFG["mc_reps"] = int(args.mc_reps)
    CFG["ref_sample_size"] = int(args.ref_sample_size)
    CFG["baseline_mc_seed"] = int(args.baseline_mc_seed)
    CFG["cache_tables"] = bool(not args.no_cache)

    data_dir = os.path.join(THIS_DIR, "data")
    fig_dir = os.path.join(THIS_DIR, "figure")
    runs_dir = os.path.join(data_dir, "runs")
    baseline_cache_dir = os.path.join(data_dir, "baseline_cache")

    _ensure_dir(data_dir)
    _ensure_dir(fig_dir)
    _ensure_dir(runs_dir)
    _ensure_dir(baseline_cache_dir)

    exp_csv = os.path.join(data_dir, "fig3_experiment.csv")
    out_png = os.path.join(fig_dir, "fig3_auc_vs_rexp.png")
    out_pdf = os.path.join(fig_dir, "fig3_auc_vs_rexp.pdf")

    step("FIG3 START")
    log(f"Mode: {'PLOT-ONLY' if args.plot_only else 'FULL'}")
    log(f"runs={CFG['runs']}  points={CFG['R_exp_points']}  R∈[{CFG['R_exp_min']},{CFG['R_exp_max']}]")
    log(f"p_on={CFG['p_on']}  delta_mu={CFG['delta_mu']}")
    log(f"mc_reps={CFG['mc_reps']}  ref_sample_size={CFG['ref_sample_size']}  cache_tables={CFG['cache_tables']}")
    log(f"baseline_cache_dir={os.path.abspath(baseline_cache_dir)}")

    if not args.plot_only:
        run_experiment_to_csv(exp_csv, baseline_cache_dir=baseline_cache_dir, runs_dir=runs_dir)

        manifest = {
            "CFG": dict(CFG),
            "methods": METHODS,
            "baseline_cache_dir": os.path.abspath(baseline_cache_dir),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "outputs": {"png": os.path.abspath(out_png), "pdf": os.path.abspath(out_pdf), "experiment_csv": os.path.abspath(exp_csv)},
        }
        _write_json(os.path.join(fig_dir, "fig3_manifest.json"), manifest)

    plot_from_experiment_csv(exp_csv, out_png, out_pdf)

    step("FIG3 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
