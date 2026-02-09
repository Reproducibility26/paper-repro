#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig15.py — Figure 15: Robustness to platform variance bias (Geo-only).

Purpose:
  This figure evaluates how attribution accuracy changes when the platform-declared
  variance differs from the true data-generating variance, while the mean is held fixed.
  The experiment isolates sensitivity to variance misspecification of the platform
  reference distribution.

Experimental setup:
  - True normal and coalition populations share the same mean (μ = 0.50) but have
    different variances.
  - Two symmetric regimes are considered:
      (a) Coalition variance larger than normal variance.
      (b) Coalition variance smaller than normal variance.
  - The platform reference variance is scaled by a factor
        γ_plat = σ²_plat / σ²_norm
    over a fixed grid.
  - For each γ_plat, each exposure regime R_exp, and multiple random seeds:
        * Compute baseline-corrected increments
              d_t = W1(H_t, H_ref(γ_plat)) − b_W1(n_t, h)
        * Accumulate per-user scores S[u] = ∑_t a_{u,t} d_t
        * Evaluate ROC–AUC(S, y) at horizon T

Pipeline:
  Uses the standard four-stage oracle pipeline:
    1) Data generation (rotation_generator.py)
    2) Evidence computation with platform variance bias (evidence_and_baselines.py)
    3) Exposure-based attribution (exposure_attribution.py)
    4) Metric evaluation (metric_eval.py)

Outputs (unchanged):
  data/
    fig15_results__sideA.csv
    fig15_results__sideB.csv
    fig15_config.json
  figure/
    fig15a_platform_variance_bias.(png|pdf)
    fig15b_platform_variance_bias.(png|pdf)
    fig15_manifest.json

Usage:
  python fig15.py
  python fig15.py --plot_only

Notes:
  - Only the Geo-W1 method is evaluated; no baseline comparisons are shown.
  - Baseline Monte Carlo tables are cached under data/baseline_cache/fig15_cache.
  - Output file names and directory structure are preserved exactly.

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


# ------------------------------------------------------------
# Path setup: script in Figure15, modules in parent dir
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
# Logging (fig5/fig9 style)
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


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ------------------------------------------------------------
# Config (keeps your existing defaults)
# ------------------------------------------------------------
CFG: Dict[str, Any] = dict(
    figure="fig15_platform_variance_bias",

    # Timeline / populations
    T=600,
    N_norm=20000,
    N_coal=2000,

    # Participation
    p_norm=0.04,
    p_on=0.50,
    R_exp_list=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0],

    # Histogram domain
    bins=50,
    x_min=0.0,
    x_max=1.0,

    # Mean matched (true mean for both groups and platform)
    mu=0.50,

    # Two symmetric sides (true variances)
    SIDE_A=dict(tag="a_coal_gt_norm", title=r"(a) $\sigma^2_{\mathrm{coal}} > \sigma^2_{\mathrm{norm}}$", var_norm=0.02, var_coal=0.05),
    SIDE_B=dict(tag="b_coal_lt_norm", title=r"(b) $\sigma^2_{\mathrm{coal}} < \sigma^2_{\mathrm{norm}}$", var_norm=0.05, var_coal=0.02),

    # Platform variance factor sweep
    gamma_min=0.6,
    gamma_max=2.0,
    gamma_step=0.1,

    # Evidence baseline MC (inside evidence engine)
    ref_sample_size=300_000,
    mc_reps=2000,
    mc_seed=0,
    cache_tables=True,

    # Replicates / bands (across seeds)
    runs=5,
    seed_base=20251225,
    q_lo=0.05,
    q_hi=0.95,
)


def BIN_EDGES() -> np.ndarray:
    return np.linspace(float(CFG["x_min"]), float(CFG["x_max"]), int(CFG["bins"]) + 1, dtype=np.float64)


def gamma_grid() -> np.ndarray:
    return np.arange(
        float(CFG["gamma_min"]),
        float(CFG["gamma_max"]) + 1e-12,
        float(CFG["gamma_step"]),
        dtype=np.float64,
    )


def _g_tag(g: float) -> str:
    return f"{g:.2f}".replace(".", "p")


def _r_tag(R: float) -> str:
    return f"{R:.1f}".replace(".", "p")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def compute_k_on(R_exp: float) -> int:
    # Rexp = (p_on * k_on / N_coal) / p_norm  =>  k_on = Rexp * p_norm * N_coal / p_on
    k = int(round(float(R_exp) * float(CFG["p_norm"]) * int(CFG["N_coal"]) / max(float(CFG["p_on"]), 1e-12)))
    return int(np.clip(k, 1, int(CFG["N_coal"])))


def metric_from_metrics_json(metrics_json_path: str, method: str = "ours_w1") -> float:
    obj = _read_json(metrics_json_path)
    try:
        return float(obj["methods"][method]["roc_auc"])
    except Exception as e:
        raise ValueError(f"metrics.json missing methods.{method}.roc_auc") from e


# ------------------------------------------------------------
# One run: generate -> evidence -> attribution -> metrics -> AUC
# ------------------------------------------------------------
def run_one_setting(
    side: Dict[str, Any],
    gamma_plat: float,
    R_exp: float,
    seed: int,
    runs_root: str,
    baseline_cache_dir: str,
) -> float:
    """
    Returns ROC-AUC at horizon T for S__ours_w1 under platform variance mismatch.
    """
    run_dir = os.path.join(
        runs_root,
        side["tag"],
        f"g{_g_tag(gamma_plat)}",
        f"R{_r_tag(R_exp)}",
        f"seed_{seed}",
    )
    _ensure_dir(run_dir)

# ---- fast resume: if metrics already computed, reuse it
    metrics_json_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_json_path):
        try:
            return metric_from_metrics_json(metrics_json_path, method="ours_w1")
        except Exception:
            # metrics.json exists but may be incomplete/corrupt; fall through to recompute
            pass

    # --------------------------
    # STEP 1: Generate dataset
    # --------------------------
    meta_path = os.path.join(run_dir, "meta.json")
    intervals_path = os.path.join(run_dir, "intervals.csv")
    exposures_path = os.path.join(run_dir, "exposures_by_t.npz")

    if not (os.path.exists(meta_path) and os.path.exists(intervals_path) and os.path.exists(exposures_path)):
        mu = float(CFG["mu"])
        sigma_norm = float(math.sqrt(float(side["var_norm"])))
        sigma_coal = float(math.sqrt(float(side["var_coal"])))
        k_on = compute_k_on(R_exp)

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
            seed=int(seed),
            mu_norm=mu,
            sigma_norm=sigma_norm,
            D_norm={"type": "normal", "mu": mu, "sigma": sigma_norm},
            D_coal={"type": "normal", "mu": mu, "sigma": sigma_coal},
            actions_per_participant_norm=1,
            actions_per_participant_coal=1,
        )
        generate_rotation_dataset(params, out_dir=run_dir, write_participation_csv=False)

    # --------------------------
    # STEP 2: Evidence (new oracle)
    # H_plat_spec depends on gamma_plat and var_norm:
    #   sigma_plat^2 = gamma_plat * sigma_norm^2 = gamma_plat * var_norm
    # --------------------------
    evidence_path = os.path.join(run_dir, "evidence.csv")
    if not os.path.exists(evidence_path):
        mu = float(CFG["mu"])
        var_norm = float(side["var_norm"])
        sigma_plat = float(math.sqrt(max(float(gamma_plat) * var_norm, 1e-12)))

        engine = EngineParams(
            H_plat_spec={"type": "normal", "mu": mu, "sigma": sigma_plat},
            ref_sample_size=int(CFG["ref_sample_size"]),
            mc_reps=int(CFG["mc_reps"]),
            mc_seed=int(CFG["mc_seed"]),
            cache_tables=bool(CFG["cache_tables"]),
            out_dir=run_dir,
            baseline_cache_dir=baseline_cache_dir,
        )
        run_evidence_engine(run_dir, engine)

    # --------------------------
    # STEP 3: Attribution (NPZ exposures)
    # --------------------------
    attr_csv = os.path.join(run_dir, "attribution_scores.csv")
    if not os.path.exists(attr_csv):
        params = AttributionParams(
            evidence_filename="evidence.csv",
            exposures_npz_filename="exposures_by_t.npz",
            allow_csv_fallback=False,
            methods=["ours_w1"],
            write_timeseries=False,
            sort_by="ours_w1",
            descending=True,
        )
        run_attribution(run_dir=run_dir, params=params)

    # --------------------------
    # STEP 4: Metrics (ROC-AUC)
    # --------------------------
    metrics_json_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_json_path):
        mparams = MetricsParams(
            attribution_filename="attribution_scores.csv",
            meta_filename="meta.json",
            methods=["ours_w1"],
            write_curves=False,
            write_ranked_lists=False,
            stable_ties=True,
        )
        run_metrics(run_dir=run_dir, params=mparams)

    return metric_from_metrics_json(metrics_json_path, method="ours_w1")


# ------------------------------------------------------------
# Sweep (one side) -> CSV
# ------------------------------------------------------------
def run_side_to_csv(side: Dict[str, Any], out_csv: str, runs_root: str, baseline_cache_dir: str) -> None:
    rows_out: List[Dict[str, Any]] = []
    gammas = gamma_grid()
    R_list = list(CFG["R_exp_list"])

    step(f"FIG15 — RUN SIDE {side['tag']}")
    log(f"title={side['title']}")
    log(f"var_norm={side['var_norm']}  var_coal={side['var_coal']}")
    log(f"gamma_grid=[{gammas[0]:.2f}..{gammas[-1]:.2f}] step={float(CFG['gamma_step']):.2f} (len={len(gammas)})")
    log(f"R_exp_list={R_list}")
    log(f"runs={CFG['runs']}, seed_base={CFG['seed_base']}")
    log(f"runs_root={os.path.abspath(runs_root)}")
    log(f"baseline_cache_dir={os.path.abspath(baseline_cache_dir)}")

    total_jobs = len(gammas) * len(R_list) * int(CFG["runs"])
    done = 0

    for g in gammas.tolist():
        for R_exp in R_list:
            auc_runs = np.zeros((int(CFG["runs"]),), dtype=np.float64)

            for r in range(int(CFG["runs"])):
                # deterministic seed per setting
                seed = int(
                    int(CFG["seed_base"])
                    + 100000 * (0 if side["tag"] == CFG["SIDE_A"]["tag"] else 1)
                    + 1000 * int(round(100 * float(g)))
                    + 10 * int(round(10 * float(R_exp)))
                    + r
                )

                t0 = time.time()
                auc_runs[r] = run_one_setting(
                    side=side,
                    gamma_plat=float(g),
                    R_exp=float(R_exp),
                    seed=seed,
                    runs_root=runs_root,
                    baseline_cache_dir=baseline_cache_dir,
                )
                done += 1
                log(
                    f"[ok] {done}/{total_jobs}  {side['tag']}  g={g:.2f}  R={R_exp:.1f}  "
                    f"seed={seed}  auc={auc_runs[r]:.3f}  ({time.time()-t0:.1f}s)"
                )

            auc_mean = float(np.mean(auc_runs))
            auc_q05 = float(np.quantile(auc_runs, float(CFG["q_lo"])))
            auc_q95 = float(np.quantile(auc_runs, float(CFG["q_hi"])))

            rows_out.append(dict(
                side_tag=side["tag"],
                var_norm=float(side["var_norm"]),
                var_coal=float(side["var_coal"]),
                gamma_plat=float(g),
                R_exp=float(R_exp),
                auc_mean=auc_mean,
                auc_q05=auc_q05,
                auc_q95=auc_q95,
            ))
            log(f"[agg] {side['tag']} g={g:.2f} R={R_exp:.1f} mean_auc={auc_mean:.3f}")

    _ensure_dir(os.path.dirname(out_csv) or ".")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "side_tag", "var_norm", "var_coal",
                "gamma_plat", "R_exp",
                "auc_mean", "auc_q05", "auc_q95",
            ],
        )
        w.writeheader()
        w.writerows(rows_out)

    log(f"[cache] wrote -> {out_csv}")


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_side(csv_path: str, side: Dict[str, Any], out_png: str, out_pdf: str) -> None:
    _require(csv_path)
    recs: List[Dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("side_tag", "") != side["tag"]:
                continue
            recs.append(dict(
                gamma_plat=float(row["gamma_plat"]),
                R_exp=float(row["R_exp"]),
                auc_mean=float(row["auc_mean"]),
                auc_q05=float(row["auc_q05"]),
                auc_q95=float(row["auc_q95"]),
            ))

    curves: Dict[float, List[Dict[str, float]]] = {}
    for rr in recs:
        curves.setdefault(rr["R_exp"], []).append(rr)
    for R in curves:
        curves[R].sort(key=lambda z: z["gamma_plat"])

    step(f"FIG15 — PLOT {side['tag']}")
    fig, ax = plt.subplots(figsize=(9.6, 5.0))

    for R_exp in sorted(curves.keys()):
        rows = curves[R_exp]
        x = np.asarray([z["gamma_plat"] for z in rows], dtype=np.float64)
        m = np.asarray([z["auc_mean"] for z in rows], dtype=np.float64)
        lo = np.asarray([z["auc_q05"] for z in rows], dtype=np.float64)
        hi = np.asarray([z["auc_q95"] for z in rows], dtype=np.float64)

        ax.plot(x, m, marker="o", lw=2.3, label=rf"$R_{{\mathrm{{exp}}}}={R_exp:.1f}$")
        ax.fill_between(x, lo, hi, alpha=0.16)

    ax.axhline(0.5, lw=1.4, linestyle=":", alpha=0.9)
    ax.set_xlim(float(CFG["gamma_min"]), float(CFG["gamma_max"]))
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.15)

    ax.set_xlabel(r"Platform variance factor $\gamma_{\mathrm{plat}}=\sigma^2_{\mathrm{plat}}/\sigma^2_{\mathrm{norm}}$")
    ax.set_ylabel(r"ROC--AUC at horizon $T$ (signed geometric score)")
    ax.set_title(side["title"])

    legend_loc = "lower right" if side["tag"] == CFG["SIDE_B"]["tag"] else "lower left"
    ax.legend(loc=legend_loc, ncol=2, frameon=True)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    log(f"[plot] wrote -> {out_png}")
    log(f"[plot] wrote -> {out_pdf}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Fig15: Robustness to platform variance bias (Geo-only; new oracles)")
    ap.add_argument("--plot_only", action="store_true", default=False)

    # light overrides (optional)
    ap.add_argument("--runs", type=int, default=int(CFG["runs"]))
    ap.add_argument("--mc_reps", type=int, default=int(CFG["mc_reps"]))
    ap.add_argument("--ref_sample_size", type=int, default=int(CFG["ref_sample_size"]))
    ap.add_argument("--no_cache", action="store_true", default=False)

    args = ap.parse_args(argv)

    CFG["runs"] = int(args.runs)
    CFG["mc_reps"] = int(args.mc_reps)
    CFG["ref_sample_size"] = int(args.ref_sample_size)
    CFG["cache_tables"] = bool(not args.no_cache)

    data_dir = os.path.join(THIS_DIR, "data")
    fig_dir = os.path.join(THIS_DIR, "figure")
    runs_root = os.path.join(data_dir, "runs")
    baseline_cache_dir = os.path.join(data_dir, "baseline_cache", "fig15_cache")

    _ensure_dir(data_dir)
    _ensure_dir(fig_dir)
    _ensure_dir(runs_root)
    _ensure_dir(baseline_cache_dir)

    csv_a = os.path.join(data_dir, "fig15_results__sideA.csv")
    csv_b = os.path.join(data_dir, "fig15_results__sideB.csv")
    out_png_a = os.path.join(fig_dir, "fig15a_platform_variance_bias.png")
    out_pdf_a = os.path.join(fig_dir, "fig15a_platform_variance_bias.pdf")
    out_png_b = os.path.join(fig_dir, "fig15b_platform_variance_bias.png")
    out_pdf_b = os.path.join(fig_dir, "fig15b_platform_variance_bias.pdf")

    step("FIG15 START")
    log(f"Mode: {'PLOT-ONLY' if args.plot_only else 'FULL'}")
    log(f"T={CFG['T']}  bins={CFG['bins']}  runs={CFG['runs']}")
    log(f"gamma=[{CFG['gamma_min']}..{CFG['gamma_max']}] step={CFG['gamma_step']}")
    log(f"mc_reps={CFG['mc_reps']}  ref_sample_size={CFG['ref_sample_size']}  cache_tables={CFG['cache_tables']}")
    log(f"runs_root={os.path.abspath(runs_root)}")
    log(f"baseline_cache_dir={os.path.abspath(baseline_cache_dir)}")

    if not args.plot_only:
        cfg_json = os.path.join(data_dir, "fig15_config.json")
        cfg = dict(CFG)
        cfg["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cfg["script"] = os.path.basename(__file__)
        cfg["paths"] = dict(
            csv_a=os.path.abspath(csv_a),
            csv_b=os.path.abspath(csv_b),
            runs_root=os.path.abspath(runs_root),
            baseline_cache_dir=os.path.abspath(baseline_cache_dir),
            out_png_a=os.path.abspath(out_png_a),
            out_pdf_a=os.path.abspath(out_pdf_a),
            out_png_b=os.path.abspath(out_png_b),
            out_pdf_b=os.path.abspath(out_pdf_b),
        )
        _write_json(cfg_json, cfg)
        log(f"[config] wrote -> {cfg_json}")

        run_side_to_csv(CFG["SIDE_A"], out_csv=csv_a, runs_root=runs_root, baseline_cache_dir=baseline_cache_dir)
        run_side_to_csv(CFG["SIDE_B"], out_csv=csv_b, runs_root=runs_root, baseline_cache_dir=baseline_cache_dir)

    plot_side(csv_a, CFG["SIDE_A"], out_png_a, out_pdf_a)
    plot_side(csv_b, CFG["SIDE_B"], out_png_b, out_pdf_b)

    manifest = dict(
        CFG=dict(CFG),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        outputs=dict(
            csv_a=os.path.abspath(csv_a),
            csv_b=os.path.abspath(csv_b),
            out_png_a=os.path.abspath(out_png_a),
            out_pdf_a=os.path.abspath(out_pdf_a),
            out_png_b=os.path.abspath(out_png_b),
            out_pdf_b=os.path.abspath(out_pdf_b),
        ),
        runs_root=os.path.abspath(runs_root),
        baseline_cache_dir=os.path.abspath(baseline_cache_dir),
    )
    _write_json(os.path.join(fig_dir, "fig15_manifest.json"), manifest)

    step("FIG15 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
