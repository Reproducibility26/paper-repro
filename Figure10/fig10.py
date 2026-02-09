#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig10.py — Figure 10: Telemetry resolution $h_t$ effect (mean-matched variance, Geo-only).

Purpose:
  This figure studies how telemetry resolution (number of bins $h_t$) affects
  attribution accuracy when coalition and normal means are matched
  ($\Delta\mu=0$) and variance is varied.
  Only the geometric Wasserstein-1 method is evaluated.

Design:
  - For each (seed, variance ratio $\gamma$), generate data once at a fine base
    resolution $B_{\mathrm{base}}$.
  - Project the same histograms to coarser resolutions $h_t$ by merging bins.
  - For each $h_t$: recompute the baseline table $b(n,h_t)$, run attribution,
    and evaluate ROC-AUC.
  - This isolates the effect of reporting resolution while holding the
    underlying actions fixed.

Outputs (relative to this file):
  - Experiment CSV:   ./data/fig10_experiment.csv
  - Config / manifest: ./data/fig10_config.json, ./figure/fig10_manifest.json
  - Figures:          ./figure/fig10_telemetry_resolution_auc.{png,pdf}

Usage:
  # Full pipeline
  python fig10.py

  # Plot-only mode
  python fig10.py --plot_only

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
# Path setup: script in Figure9, modules in parent dir
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
# Global config (close to old fig9)
# ------------------------------------------------------------
HS: List[int] = [2, 5, 10, 20, 50, 100]
B_BASE: int = 100  # must be divisible by every h

GAMMAS: List[float] = [
    0.50, 0.65, 0.75, 0.85, 0.92, 0.96, 1.00, 1.04, 1.08, 1.15, 1.30, 1.60, 2.00
]

CFG: Dict[str, Any] = dict(
    figure="fig10_telemetry_resolution_auc",

    # experiment design (matches old fig9 build_params_base)
    T=1000,
    p_on=0.5,
    R_exp_target=1.0,

    N_norm=20000,
    N_coal=2000,
    p_norm=0.04,

    # domain/support
    x_min=0.0,
    x_max=1.0,

    # normal model
    mu_norm=0.50,
    var_norm=0.05,

    # seeds
    n_seeds=20,
    base_seed=20251224,

    # baseline calibration (new oracle)
    mc_reps=200,
    ref_sample_size=300_000,
    cache_tables=True,
    baseline_mc_seed=999,
)


# ------------------------------------------------------------
# Helpers: coarsening
# ------------------------------------------------------------
def coarsen_counts(counts: np.ndarray, factor: int) -> np.ndarray:
    """Merge adjacent bins by factor. (..., B) -> (..., B/factor)."""
    B = counts.shape[-1]
    if B % factor != 0:
        raise ValueError(f"B={B} not divisible by factor={factor}")
    newB = B // factor
    reshaped = counts.reshape(*counts.shape[:-1], newB, factor)
    return reshaped.sum(axis=-1)


def coarsen_edges(bin_edges: np.ndarray, factor: int) -> np.ndarray:
    """Take every 'factor' edge: (B+1,) -> (B/factor+1,)."""
    B = len(bin_edges) - 1
    if B % factor != 0:
        raise ValueError(f"B={B} not divisible by factor={factor}")
    return bin_edges[::factor].copy()


# ------------------------------------------------------------
# Read/write intervals.csv in the new pipeline schema
# ------------------------------------------------------------
def read_intervals_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t_arr: (T,)
      n_actions_arr: (T,)  (may be reconstructed from counts if missing)
      counts: (T, B)
    Assumes columns include: t, counts_json, (optionally n_actions)
    """
    _require(path)
    t_list: List[int] = []
    n_list: List[int] = []
    counts_list: List[np.ndarray] = []

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError(f"{path} has no header")
        if "t" not in r.fieldnames or "counts_json" not in r.fieldnames:
            raise ValueError(f"{path} missing required columns t, counts_json")

        has_n = "n_actions" in r.fieldnames
        for row in r:
            t = int(row["t"])
            counts = np.array(json.loads(row["counts_json"]), dtype=np.int64)
            n = int(row["n_actions"]) if has_n and row["n_actions"] != "" else int(counts.sum())
            t_list.append(t)
            n_list.append(n)
            counts_list.append(counts)

    t_arr = np.asarray(t_list, dtype=np.int64)
    n_arr = np.asarray(n_list, dtype=np.int64)
    counts_arr = np.stack(counts_list, axis=0)
    return t_arr, n_arr, counts_arr


def write_intervals_csv(path: str, t_arr: np.ndarray, n_arr: np.ndarray, counts_arr: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", "n_actions", "counts_json", "h_bins"])
        w.writeheader()
        h_bins = int(counts_arr.shape[1])
        for i in range(len(t_arr)):
            w.writerow({
                "t": int(t_arr[i]),
                "n_actions": int(n_arr[i]),
                "counts_json": json.dumps(counts_arr[i].astype(int).tolist()),
                "h_bins": h_bins,
            })


# ------------------------------------------------------------
# Extract ROC-AUC from metric_eval.py output
# ------------------------------------------------------------
def load_roc_auc(metrics_json: str, method: str) -> float:
    obj = _read_json(metrics_json)
    mb = obj.get("methods", None)
    if not isinstance(mb, dict):
        raise ValueError("metrics.json missing 'methods'")
    md = mb.get(method, None)
    if not isinstance(md, dict) or "roc_auc" not in md:
        raise ValueError(f"metrics.json missing methods['{method}']['roc_auc']")
    return float(md["roc_auc"])


# ------------------------------------------------------------
# Config JSON
# ------------------------------------------------------------
def write_config_json(path: str) -> None:
    cfg = dict(CFG)
    cfg.update({"hs": HS, "B_base": B_BASE, "gammas": GAMMAS})
    cfg["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cfg["script"] = os.path.basename(__file__)
    cfg["notes"] = (
        "Mean matched. Sweep var_scale=Var_coal/Var_norm using D_coal=normal_varscale. "
        "For each (seed,gamma), generate once at B_base then project to each h. "
        "For each h, run evidence engine (Geo-W1) + attribution + metrics; plot ROC-AUC vs gamma."
    )
    _write_json(path, cfg)
    log(f"[config] wrote -> {path}")


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------
def build_base_generator_params(seed: int, gamma: float) -> Dict[str, Any]:
    """
    Mean matched; variance sweep:
      Var_coal = gamma * Var_norm
    """
    from rotation_generator import RotationParams  # just for field validation

    T = int(CFG["T"])
    p_on = float(CFG["p_on"])
    R_exp_target = float(CFG["R_exp_target"])

    N_norm = int(CFG["N_norm"])
    N_coal = int(CFG["N_coal"])
    p_norm = float(CFG["p_norm"])

    # exposure match
    k_on = int(round(R_exp_target * p_norm * N_coal / max(p_on, 1e-12)))
    k_on = max(1, min(N_coal, k_on))

    mu_norm = float(CFG["mu_norm"])
    sigma_norm = float(np.sqrt(float(CFG["var_norm"])))

    params: Dict[str, Any] = dict(
        N_norm=N_norm,
        N_coal=N_coal,
        T=T,
        p_norm=p_norm,
        p_on=p_on,
        k_on=k_on,

        bins=B_BASE,
        x_min=float(CFG["x_min"]),
        x_max=float(CFG["x_max"]),

        mu_norm=mu_norm,
        sigma_norm=sigma_norm,
        D_norm={"type": "normal", "mu": mu_norm, "sigma": sigma_norm},
        D_coal={"type": "normal_varscale", "var_scale": float(gamma)},

        actions_per_participant_norm=1,
        actions_per_participant_coal=1,

        seed=int(seed),
    )

    bad = set(params.keys()) - set(RotationParams.__dataclass_fields__.keys())
    if bad:
        raise ValueError(f"Bad keys for RotationParams: {bad}")
    return params


def run_experiment_to_csv(exp_csv: str, runs_root: str, baseline_cache_root: str) -> None:
    from rotation_generator import RotationParams, generate_rotation_dataset
    from evidence_and_baselines import EngineParams, run_evidence_engine
    from exposure_attribution import AttributionParams, run_attribution
    from metric_eval import MetricsParams, run_metrics

    _ensure_dir(os.path.dirname(exp_csv) or ".")
    _ensure_dir(runs_root)
    _ensure_dir(baseline_cache_root)

    factors: Dict[int, int] = {}
    for h in HS:
        if B_BASE % h != 0:
            raise ValueError(f"B_BASE={B_BASE} must be divisible by h={h}")
        factors[h] = B_BASE // h

    rows: List[Dict[str, Any]] = []
    method = "ours_w1"

    for si in range(int(CFG["n_seeds"])):
        seed = int(CFG["base_seed"]) + si

        for gamma in GAMMAS:
            step(f"seed={seed} gamma={gamma:.3f} — generate base at B_BASE={B_BASE}")

            # -------- base generation --------
            base_dir = os.path.join(runs_root, f"seed_{seed}", f"gamma_{gamma:.3f}", "base")
            _ensure_dir(base_dir)

            params_dict = build_base_generator_params(seed, gamma)
            gen_params = RotationParams(**params_dict)
            generate_rotation_dataset(gen_params, out_dir=base_dir)

            # load base artifacts
            base_meta_path = os.path.join(base_dir, "meta.json")
            base_intervals_path = os.path.join(base_dir, "intervals.csv")
            base_exposures_path = os.path.join(base_dir, "exposures_by_t.npz")

            _require(base_meta_path)
            _require(base_intervals_path)
            _require(base_exposures_path)

            base_meta = _read_json(base_meta_path)
            if "bin_edges" not in base_meta:
                raise ValueError("base meta.json missing bin_edges")
            bin_edges_base = np.asarray(base_meta["bin_edges"], dtype=np.float64)

            t_arr, n_arr, counts_base = read_intervals_csv(base_intervals_path)
            if counts_base.shape[1] != B_BASE:
                raise ValueError(f"Expected base bins={B_BASE}, got {counts_base.shape[1]}")

            # -------- per-h projection + oracle pipeline --------
            for h in HS:
                fct = factors[h]
                counts_h = coarsen_counts(counts_base, fct)
                edges_h = coarsen_edges(bin_edges_base, fct)

                run_dir = os.path.join(runs_root, f"seed_{seed}", f"gamma_{gamma:.3f}", f"h_{h}")
                _ensure_dir(run_dir)

                # write derived meta.json: copy base meta but replace bin_edges + config_id
                meta_h = dict(base_meta)
                meta_h["bin_edges"] = edges_h.tolist()
                meta_h["config_id"] = f"fig10_seed{seed}_gamma{gamma:.3f}_h{h}"
                _write_json(os.path.join(run_dir, "meta.json"), meta_h)

                # write derived intervals.csv
                write_intervals_csv(os.path.join(run_dir, "intervals.csv"), t_arr, n_arr, counts_h)

                # copy exposures_by_t.npz (same exposures; only histogram projection changes)
                with open(base_exposures_path, "rb") as src, open(
                    os.path.join(run_dir, "exposures_by_t.npz"), "wb"
                ) as dst:
                    dst.write(src.read())

                # per-h baseline cache dir
                cache_h = os.path.join(baseline_cache_root, f"h_{h}")
                _ensure_dir(cache_h)

                # evidence engine
                eng = EngineParams(
                    H_plat_spec=params_dict["D_norm"],
                    ref_sample_size=int(CFG["ref_sample_size"]),
                    mc_reps=int(CFG["mc_reps"]),
                    mc_seed=int(CFG["baseline_mc_seed"]),
                    cache_tables=bool(CFG["cache_tables"]),
                    out_dir=run_dir,
                    baseline_cache_dir=cache_h,
                )
                run_evidence_engine(run_dir, eng)

                # attribution (Geo only)
                attr = AttributionParams(
                    out_dir=run_dir,
                    methods=[method],
                    allow_raw_score_if_missing=False,
                    write_timeseries=False,
                    sort_by=f"w__{method}",
                    descending=True,
                )
                run_attribution(run_dir, attr)

                # metrics (Geo only)
                met = MetricsParams(
                    out_dir=run_dir,
                    methods=[method],
                    write_curves=False,
                    write_ranked_lists=False,
                )
                run_metrics(run_dir, met)

                auc = load_roc_auc(os.path.join(run_dir, "metrics.json"), method)

                rows.append({
                    "gamma": float(gamma),
                    "h": int(h),
                    "seed": int(seed),
                    "roc_auc_geom": float(auc),
                })

            log(f"[OK] seed={seed}, gamma={gamma:.3f} done")

    # Save CSV
    with open(exp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gamma", "h", "seed", "roc_auc_geom"])
        w.writeheader()
        w.writerows(rows)

    log(f"[OK] wrote: {exp_csv}")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def plot_from_csv(exp_csv: str, out_png: str, out_pdf: str) -> None:
    _require(exp_csv)

    by_h: Dict[int, Dict[float, List[float]]] = {h: {g: [] for g in GAMMAS} for h in HS}

    with open(exp_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            h = int(float(row["h"]))
            g = float(row["gamma"])
            auc = float(row["roc_auc_geom"])
            if h in by_h and g in by_h[h]:
                by_h[h][g].append(auc)

    xs = np.asarray(GAMMAS, dtype=np.float64)
    qlo, qhi = 0.05, 0.95

    step("PLOT — Fig.9")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))

    ax.axhline(0.5, lw=1.3, linestyle=":", alpha=0.9)
    ax.text(xs.min() + 0.02 * (xs.max() - xs.min()), 0.52, r"chance (AUC=0.5)", fontsize=10)
    ax.axvline(1.0, lw=1.3, linestyle="--", alpha=0.9)

    hs_sorted = sorted(HS)
    h_primary = max(hs_sorted)

    for h in hs_sorted:
        means, lo, hi = [], [], []
        for g in GAMMAS:
            vals = np.asarray(by_h[h][g], dtype=np.float64)
            if vals.size == 0:
                means.append(float("nan"))
                lo.append(float("nan"))
                hi.append(float("nan"))
            else:
                means.append(float(np.mean(vals)))
                lo.append(float(np.quantile(vals, qlo)))
                hi.append(float(np.quantile(vals, qhi)))

        means_a = np.asarray(means, dtype=np.float64)
        lo_a = np.asarray(lo, dtype=np.float64)
        hi_a = np.asarray(hi, dtype=np.float64)

        if h == h_primary:
            lw, ms, alpha, z, ls = 3.0, 6.5, 1.0, 5, "-"
        elif h == min(hs_sorted):
            lw, ms, alpha, z, ls = 2.0, 5.2, 0.80, 3, "--"
        else:
            lw, ms, alpha, z, ls = 2.2, 5.5, 0.90, 4, "-"

        ax.plot(xs, means_a, marker="o", markersize=ms, lw=lw, linestyle=ls, alpha=alpha,
                label=fr"$h_t={h}$", zorder=z)
        ax.fill_between(xs, lo_a, hi_a, alpha=0.12, zorder=z - 1)

    ax.set_xlim(0.5, float(xs.max()))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Variance ratio $\gamma=\mathrm{Var}_{\mathrm{coal}}/\mathrm{Var}_{\mathrm{norm}}$")
    ax.set_ylabel(r"ROC--AUC at horizon $T$")
    ax.grid(alpha=0.15)

    ax.legend(
        loc="lower right",
        frameon=True,
        ncol=2,
        borderpad=0.6,
        handlelength=2.2,
        columnspacing=1.2,
    )

    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_png) or ".")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    log(f"[OK] wrote: {out_png}")
    log(f"[OK] wrote: {out_pdf}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Fig10: Telemetry resolution h_t effect (mean-matched variance), Geo-only")
    ap.add_argument("--plot_only", action="store_true", default=False)

    ap.add_argument("--n_seeds", type=int, default=int(CFG["n_seeds"]))
    ap.add_argument("--mc_reps", type=int, default=int(CFG["mc_reps"]))
    ap.add_argument("--ref_sample_size", type=int, default=int(CFG["ref_sample_size"]))
    ap.add_argument("--baseline_mc_seed", type=int, default=int(CFG["baseline_mc_seed"]))
    ap.add_argument("--no_cache", action="store_true", default=False)

    args = ap.parse_args(argv)

    CFG["n_seeds"] = int(args.n_seeds)
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

    exp_csv = os.path.join(data_dir, "fig10_experiment.csv")
    cfg_json = os.path.join(data_dir, "fig10_config.json")
    out_png = os.path.join(fig_dir, "fig10_telemetry_resolution_auc.png")
    out_pdf = os.path.join(fig_dir, "fig10_telemetry_resolution_auc.pdf")

    step("FIG10 START")
    log(f"Mode: {'PLOT-ONLY' if args.plot_only else 'FULL'}")
    log(f"HS={HS}  B_BASE={B_BASE}  gammas={len(GAMMAS)}  n_seeds={CFG['n_seeds']}")
    log(f"mc_reps={CFG['mc_reps']}  ref_sample_size={CFG['ref_sample_size']}  cache_tables={CFG['cache_tables']}")
    log(f"runs_root={os.path.abspath(runs_root)}")
    log(f"baseline_cache_root={os.path.abspath(baseline_cache_root)}")

    if not args.plot_only:
        write_config_json(cfg_json)
        run_experiment_to_csv(exp_csv, runs_root=runs_root, baseline_cache_root=baseline_cache_root)

        manifest = {
            "CFG": dict(CFG),
            "HS": HS,
            "B_BASE": B_BASE,
            "GAMMAS": GAMMAS,
            "baseline_cache_root": os.path.abspath(baseline_cache_root),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "outputs": {
                "png": os.path.abspath(out_png),
                "pdf": os.path.abspath(out_pdf),
                "experiment_csv": os.path.abspath(exp_csv),
            },
        }
        _write_json(os.path.join(fig_dir, "fig10_manifest.json"), manifest)

    plot_from_csv(exp_csv, out_png, out_pdf)

    step("FIG10 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
