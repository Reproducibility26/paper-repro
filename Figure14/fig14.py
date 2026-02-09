#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig14.py — Figure 14: Robustness to platform reference mean bias (Geo-only).

Purpose:
  This figure evaluates how attribution accuracy degrades as the platform-declared
  reference mean deviates from the true population means. The goal is to test robustness
  of the geometric Wasserstein-1 method to misspecification of the platform reference.

Experimental setup:
  - True normal and coalition populations are Beta distributions on [0,1] with fixed means
        μ_norm = 0.20,  μ_coal = 0.80.
  - The platform reference distribution is also Beta on [0,1], with mean μ_plat swept
    from 0.20 to 0.80 in fixed increments.
  - For each μ_plat and each exposure regime R_exp in CFG["R_exp_list"]:
        * Compute baseline-corrected increments
              d_t = W1(H_t, H_ref(μ_plat)) − b_W1(n_t, h)
        * Accumulate per-user scores S[u] = ∑_t a_{u,t} d_t
        * Evaluate ROC–AUC(S, y) at horizon T

Pipeline:
  This script uses the standard four-stage oracle pipeline:
    1) Data generation (rotation_generator.py)
    2) Evidence computation with platform reference (evidence_and_baselines.py)
    3) Exposure-based attribution (exposure_attribution.py)
    4) Metric evaluation (metric_eval.py)

Outputs (unchanged):
  data/
    fig14_results__platform_mean_bias.csv
    fig14_config.json
  figure/
    fig14_platform_mean_bias.(png|pdf)
    fig14_manifest.json

Usage:
  python fig14.py
  python fig14.py --plot_only

Notes:
  - Only the Geo-W1 method is evaluated; no baseline comparisons are shown.
  - Baseline Monte Carlo tables are cached under data/baseline_cache/fig14_cache.
  - All output file names and directory structure are preserved from the legacy script.

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Path setup: script in Figure14, modules in parent dir
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if A_DIR not in sys.path:
    sys.path.insert(0, A_DIR)

from rotation_generator import RotationParams, generate_rotation_dataset
from evidence_and_baselines import EngineParams, run_evidence_engine
from exposure_attribution import AttributionParams, run_attribution
from metric_eval import MetricsParams, run_metrics


# ------------------------------------------------------------
# Logging (match fig9/legacy style)
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

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")


# ------------------------------------------------------------
# Config (same semantics as legacy fig14)
# ------------------------------------------------------------
CFG: Dict[str, Any] = dict(
    figure="fig14_platform_mean_bias_geo_only__oracle",

    # time + populations
    T=2000,
    N_norm=20000,
    N_coal=2000,

    # participation
    p_norm=0.04,
    p_on=0.50,

    # curves
    R_exp_list=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],

    # true populations (Beta on [0,1])
    mu_norm=0.20,
    mu_coal=0.80,

    # platform mean sweep
    mu_plat_min=0.20,
    mu_plat_max=0.80,
    mu_plat_step=0.10,

    # Beta concentration
    beta_kappa=40.0,

    # histogram
    bins=50,
    x_min=0.0,
    x_max=1.0,

    # baseline table b(n,h) settings under platform reference (oracle Script #2)
    baseline_mc_reps=200,
    baseline_mc_seed_offset=999,
    baseline_ref_samples=300_000,   # ref_sample_size for building H_ref

    # replicates for bands
    seeds=list(range(10)),
    seed_base=20260120,
    q_lo=0.05,
    q_hi=0.95,
)

def mu_plat_grid() -> np.ndarray:
    return np.arange(
        float(CFG["mu_plat_min"]),
        float(CFG["mu_plat_max"]) + 1e-12,
        float(CFG["mu_plat_step"]),
        dtype=np.float64,
    )

def beta_ab_from_mu(mu: float, kappa: float) -> Tuple[float, float]:
    a = max(float(mu) * float(kappa), 1e-6)
    b = max((1.0 - float(mu)) * float(kappa), 1e-6)
    return a, b

def compute_k_on(R_exp: float) -> int:
    # Rexp = (p_on * k_on / N_coal) / p_norm  => k_on = Rexp * p_norm * N_coal / p_on
    k = int(round(float(R_exp) * float(CFG["p_norm"]) * int(CFG["N_coal"]) / max(float(CFG["p_on"]), 1e-12)))
    return int(np.clip(k, 1, int(CFG["N_coal"])))


# ------------------------------------------------------------
# One run: generate -> evidence -> attribution -> metrics
# ------------------------------------------------------------
def run_one_setting(
    R_exp: float,
    mu_plat: float,
    seed: int,
    runs_root: str,
    baseline_cache_dir: str,
) -> float:
    """
    Returns ROC-AUC at horizon T for method 'ours_w1' under platform mean bias mu_plat.
    """
    run_dir = os.path.join(
        runs_root,
        f"R{R_exp:.1f}".replace(".", "p"),
        f"mu{mu_plat:.2f}".replace(".", "p"),
        f"seed_{seed}",
    )
    _ensure_dir(run_dir)

    # ---- fast resume: if metrics already computed, reuse it
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            metrics = _read_json(metrics_path)
            return float(metrics["methods"]["ours_w1"]["roc_auc"])
        except Exception:
            # metrics.json exists but is incomplete or different format; fall through to recompute
            pass

    metrics_csv = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(metrics_csv):
        try:
            import pandas as pd
            df = pd.read_csv(metrics_csv)
            row = df[df["method"] == "ours_w1"].iloc[0]
            return float(row["roc_auc"])
        except Exception:
            pass

    # True data distributions (beta on [0,1])
    a_n, b_n = beta_ab_from_mu(float(CFG["mu_norm"]), float(CFG["beta_kappa"]))
    a_c, b_c = beta_ab_from_mu(float(CFG["mu_coal"]), float(CFG["beta_kappa"]))
    D_norm = {"type": "beta", "a": float(a_n), "b": float(b_n)}
    D_coal = {"type": "beta", "a": float(a_c), "b": float(b_c)}

    k_on = compute_k_on(R_exp)

    # Script #1: generate dataset
    params = RotationParams(
        N_norm=int(CFG["N_norm"]),
        N_coal=int(CFG["N_coal"]),
        T=int(CFG["T"]),
        p_norm=float(CFG["p_norm"]),
        p_on=float(CFG["p_on"]),
        k_on=int(k_on),
        bins=int(CFG["bins"]),
        x_min=float(CFG["x_min"]),
        x_max=float(CFG["x_max"]),
        # mu/sigma unused for beta
        mu_norm=0.5,
        sigma_norm=1.0,
        D_norm=D_norm,
        D_coal=D_coal,
        actions_per_participant_norm=1,
        actions_per_participant_coal=1,
        seed=int(seed),
    )
    generate_rotation_dataset(params, out_dir=run_dir)

    # Script #2: evidence under a biased platform reference mean mu_plat
    a_p, b_p = beta_ab_from_mu(float(mu_plat), float(CFG["beta_kappa"]))
    H_plat_spec = {"type": "beta", "a": float(a_p), "b": float(b_p)}

    engine = EngineParams(
        H_plat_spec=H_plat_spec,
        ref_sample_size=int(CFG["baseline_ref_samples"]),
        mc_reps=int(CFG["baseline_mc_reps"]),
        mc_seed=int(CFG["baseline_mc_seed_offset"]),  # fixed for determinism; cache handles reuse
        cache_tables=True,
        out_dir=run_dir,
        baseline_cache_dir=baseline_cache_dir,
    )
    run_evidence_engine(run_dir, engine)

    # Script #3: attribution (geo-only: ours_w1)
    attr = AttributionParams(
        methods=["ours_w1"],
        evidence_filename="evidence.csv",
        out_dir=run_dir,
        allow_raw_score_if_missing=False,
    )
    run_attribution(run_dir, attr)

    # Script #4: metrics (just ROC-AUC at horizon)
    met = MetricsParams(
        attribution_filename="attribution_scores.csv",
        meta_filename="meta.json",
        out_dir=run_dir,
        methods=["ours_w1"],
    )
    run_metrics(run_dir, met)

    metrics = _read_json(os.path.join(run_dir, "metrics.json"))
    # metrics schema: metrics["methods"][method]["roc_auc"]
    auc = float(metrics["methods"]["ours_w1"]["roc_auc"])
    return auc


# ------------------------------------------------------------
# Experiment driver + plot
# ------------------------------------------------------------
def run_experiment(results_csv: str, runs_root: str, baseline_cache_dir: str) -> None:
    _ensure_dir(os.path.dirname(results_csv) or ".")
    rows_out: List[Dict[str, Any]] = []

    mus = mu_plat_grid()
    step("RUN — Fig14 platform mean bias (oracle)")
    log(f"R_exp_list={CFG['R_exp_list']}")
    log(f"mu_plat_grid=[{mus[0]:.2f}..{mus[-1]:.2f}] step={float(CFG['mu_plat_step']):.2f}  (len={len(mus)})")
    log(f"seeds={CFG['seeds'][:5]}... (len={len(CFG['seeds'])})")
    log(f"runs_root={os.path.abspath(runs_root)}")
    log(f"baseline_cache_dir={os.path.abspath(baseline_cache_dir)}")

    for idx_R, R_exp in enumerate(list(CFG["R_exp_list"])):
        step(f"Curve — R_exp={float(R_exp):.1f}")
        for mu_plat in mus.tolist():
            aucs = np.zeros((len(CFG["seeds"]),), dtype=np.float64)

            for i, r in enumerate(list(CFG["seeds"])):
                seed = int(
                    int(CFG["seed_base"])
                    + 100000 * int(idx_R)
                    + 1000 * int(round(100 * float(mu_plat)))
                    + int(r)
                )
                t0 = time.time()
                aucs[i] = run_one_setting(
                    R_exp=float(R_exp),
                    mu_plat=float(mu_plat),
                    seed=seed,
                    runs_root=runs_root,
                    baseline_cache_dir=baseline_cache_dir,
                )
                log(f"[ok] R_exp={float(R_exp):.1f} mu_plat={mu_plat:.2f} seed={seed} auc={aucs[i]:.3f} ({time.time()-t0:.1f}s)")

            rows_out.append(
                dict(
                    R_exp=float(R_exp),
                    mu_plat=float(mu_plat),
                    auc_mean=float(np.mean(aucs)),
                    auc_q05=float(np.quantile(aucs, float(CFG["q_lo"]))),
                    auc_q95=float(np.quantile(aucs, float(CFG["q_hi"]))),
                )
            )
            log(f"[agg] R_exp={float(R_exp):.1f} mu_plat={mu_plat:.2f} mean_auc={rows_out[-1]['auc_mean']:.3f}")

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["R_exp", "mu_plat", "auc_mean", "auc_q05", "auc_q95"])
        w.writeheader()
        w.writerows(rows_out)

    log(f"[write] results -> {results_csv}")


def plot_from_csv(results_csv: str, out_png: str, out_pdf: str) -> None:
    _require(results_csv)
    recs: List[Dict[str, float]] = []
    with open(results_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            recs.append(
                dict(
                    R_exp=float(row["R_exp"]),
                    mu_plat=float(row["mu_plat"]),
                    auc_mean=float(row["auc_mean"]),
                    auc_q05=float(row["auc_q05"]),
                    auc_q95=float(row["auc_q95"]),
                )
            )

    step("PLOT — Fig14 (oracle)")
    fig, ax = plt.subplots(figsize=(9.6, 5.0))

    for R_exp in sorted({x["R_exp"] for x in recs}):
        rows = [x for x in recs if x["R_exp"] == R_exp]
        rows.sort(key=lambda z: z["mu_plat"])

        x = np.asarray([z["mu_plat"] for z in rows], dtype=np.float64)
        m = np.asarray([z["auc_mean"] for z in rows], dtype=np.float64)
        lo = np.asarray([z["auc_q05"] for z in rows], dtype=np.float64)
        hi = np.asarray([z["auc_q95"] for z in rows], dtype=np.float64)

        ax.plot(x, m, marker="o", lw=2.3, label=rf"$R_{{\mathrm{{exp}}}}={R_exp:.1f}$")
        ax.fill_between(x, lo, hi, alpha=0.16)

    ax.axhline(0.5, lw=1.3, linestyle=":", alpha=0.9)
    ax.text(float(CFG["mu_plat_min"]) + 0.005, 0.515, "chance (AUC=0.5)", fontsize=10, alpha=0.75)

    ax.axvline(0.5, lw=1.0, linestyle="--", alpha=0.30)
    ax.text(0.505, 0.02, "midpoint", fontsize=9, alpha=0.55, rotation=90, va="bottom")

    ax.set_xlim(float(CFG["mu_plat_min"]), float(CFG["mu_plat_max"]))
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.15)

    ax.set_xlabel(r"Platform reference mean $\mu_{\mathrm{plat}}$ (Beta on $[0,1]$)")
    ax.set_ylabel(r"ROC--AUC at horizon $T$")
    ax.legend(frameon=True, ncol=2)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    log(f"[plot] wrote: {out_png}")
    log(f"[plot] wrote: {out_pdf}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Fig14: Robustness to platform mean bias (Geo-only) — oracle pipeline")
    ap.add_argument("--plot_only", action="store_true", default=False)

    # minimal overrides (fig9-style)
    ap.add_argument("--baseline_mc_reps", type=int, default=int(CFG["baseline_mc_reps"]))
    ap.add_argument("--baseline_ref_samples", type=int, default=int(CFG["baseline_ref_samples"]))
    args = ap.parse_args(argv)

    CFG["baseline_mc_reps"] = int(args.baseline_mc_reps)
    CFG["baseline_ref_samples"] = int(args.baseline_ref_samples)

    data_dir = os.path.join(THIS_DIR, "data")
    fig_dir = os.path.join(THIS_DIR, "figure")
    runs_root = os.path.join(data_dir, "runs")
    baseline_cache_dir = os.path.join(data_dir, "baseline_cache", "fig14_cache")

    _ensure_dir(data_dir)
    _ensure_dir(fig_dir)
    _ensure_dir(runs_root)
    _ensure_dir(baseline_cache_dir)

    results_csv = os.path.join(data_dir, "fig14_results__platform_mean_bias.csv")
    out_png = os.path.join(fig_dir, "fig14_platform_mean_bias.png")
    out_pdf = os.path.join(fig_dir, "fig14_platform_mean_bias.pdf")

    step("FIG14 START (oracle)")
    log(f"Mode: {'PLOT-ONLY' if args.plot_only else 'FULL'}")
    log(f"T={CFG['T']}  bins={CFG['bins']}  kappa={CFG['beta_kappa']}")
    log(f"R_exp_list={CFG['R_exp_list']}")
    log(f"baseline_mc_reps={CFG['baseline_mc_reps']}  baseline_ref_samples={CFG['baseline_ref_samples']}")
    log(f"runs_root={os.path.abspath(runs_root)}")
    log(f"baseline_cache_dir={os.path.abspath(baseline_cache_dir)}")

    if not args.plot_only:
        cfg_json = os.path.join(data_dir, "fig14_config.json")
        cfg = dict(CFG)
        cfg["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cfg["script"] = os.path.basename(__file__)
        cfg["paths"] = dict(
            results_csv=os.path.abspath(results_csv),
            runs_root=os.path.abspath(runs_root),
            baseline_cache_dir=os.path.abspath(baseline_cache_dir),
            out_png=os.path.abspath(out_png),
            out_pdf=os.path.abspath(out_pdf),
        )
        _write_json(cfg_json, cfg)
        log(f"[config] wrote -> {cfg_json}")

        run_experiment(results_csv=results_csv, runs_root=runs_root, baseline_cache_dir=baseline_cache_dir)

    plot_from_csv(results_csv=results_csv, out_png=out_png, out_pdf=out_pdf)

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
    _write_json(os.path.join(fig_dir, "fig14_manifest.json"), manifest)

    step("FIG14 DONE (oracle)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
