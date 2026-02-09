\
#!/usr/bin/env python3
"""
fig11.py — Figure 11: Finite-sample null behavior under the oracle baseline.

Purpose:
  This figure replicates the classic finite-sample sweep under the declared normal null
  using the oracle-first evidence engine. It illustrates how raw geometric scores exhibit
  positive finite-sample bias, while baseline-corrected increments remove this bias.

Design:
  - Fix histogram resolution h and declared normal reference.
  - For each action count n = 1..N_max:
      * Run R iid trials under the null.
      * Record the raw score X(n) and corrected score d(n)=X(n)-b(n).
  - Plot the median and [q_lo, q_hi] bands across trials as a function of n.

Outputs (relative to this script directory):
  data/
    baseline_cache/               (shared tables)
    runs/n_*/trial_*/             (meta.json, intervals.csv, evidence.csv)
    fig11_experiment.csv
    fig11_config.json
  figure/
    fig11_finite_sample_variability.(png|pdf)

Usage:
  # Full run
  python fig11.py

  # Plot-only mode
  python fig11.py --plot_only

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if A_DIR not in sys.path:
    sys.path.insert(0, A_DIR)

from evidence_and_baselines import EngineParams, run_evidence_engine


# -----------------------------
# Config (match old Fig8 intent)
# -----------------------------
CFG: Dict[str, Any] = dict(
    # histogram
    H=20,
    x_min=0.0,
    x_max=1.0,

    # declared platform null (normal)
    mu_norm=0.50,
    var_norm=0.05,

    # sweep
    n_min=1,
    n_max=200,
    n_step=1,
    R=20,

    # baseline engine (match old Fig8 as closely as possible)
    ref_sample_size=300_000,
    mc_reps=800,
    mc_seed=999,          # baseline MC seed in the oracle engine
    sample_seed_base=20251224,  # data sampling seed base (per trial)

    # plot bands
    q_lo=0.15,
    q_hi=0.95,

    # output
    out_root=os.path.join(THIS_DIR, "data"),
    fig_root=os.path.join(THIS_DIR, "figure"),
    csv_name="fig11_experiment.csv",
    config_name="fig11_config.json",
    figure_name="fig11_finite_sample_variability",
)


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _read_evidence_one(evidence_csv: str) -> Tuple[float, float]:
    """
    Returns (raw_score, corrected_score) from t=1 row.
    We use:
      raw = score__ours_w1
      corr = w__ours_w1
    """
    with open(evidence_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("evidence.csv missing header")
        need = ["t", "score__ours_w1", "w__ours_w1"]
        for k in need:
            if k not in r.fieldnames:
                raise ValueError(f"evidence.csv missing column: {k}")
        for row in r:
            if int(row["t"]) == 1:
                return float(row["score__ours_w1"]), float(row["w__ours_w1"])
    raise ValueError("evidence.csv missing t=1 row")


def _make_edges() -> np.ndarray:
    H = int(CFG["H"])
    x_min = float(CFG["x_min"])
    x_max = float(CFG["x_max"])
    return np.linspace(x_min, x_max, H + 1, dtype=np.float64)


def _sample_counts(n: int, seed: int, edges: np.ndarray) -> np.ndarray:
    """
    Sample n iid points from N(mu, sigma^2), clip to [x_min, x_max], and bin into H counts.
    This mirrors the typical synthetic null sampling used elsewhere in the repo.
    """
    rng = np.random.default_rng(int(seed))
    mu = float(CFG["mu_norm"])
    sigma = float(np.sqrt(float(CFG["var_norm"])))
    x = rng.normal(loc=mu, scale=sigma, size=int(n))
    x = np.clip(x, float(CFG["x_min"]), float(CFG["x_max"]))
    counts, _ = np.histogram(x, bins=edges)
    return counts.astype(np.int64)


def _write_oracle_dataset(run_dir: str, n: int, seed: int, edges: np.ndarray) -> None:
    _ensure_dir(run_dir)
    meta = {
        "schema": 1,
        "dataset": "finite_sample_sweep",
        "config_id": f"finite_sample_H{int(CFG['H'])}_x{CFG['x_min']}_{CFG['x_max']}",
        "bin_edges": edges.tolist(),
        "notes": {
            "n_actions_fixed": int(n),
            "sampling_seed": int(seed),
        },
    }
    _write_json(os.path.join(run_dir, "meta.json"), meta)

    counts = _sample_counts(n=n, seed=seed, edges=edges)
    # intervals.csv: one row (t=1) with counts_json as JSON list
    with open(os.path.join(run_dir, "intervals.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", "h_bins", "counts_json"])
        w.writeheader()
        w.writerow({
            "t": 1,
            "h_bins": int(CFG["H"]),
            "counts_json": json.dumps([int(v) for v in counts.tolist()]),
        })


def _prime_baseline_cache(baseline_cache_dir: str, edges: np.ndarray) -> None:
    """
    Ensure cached baseline tables contain entries for ALL n=1..N_MAX.
    We do this by running one dummy dataset with observed_n = N_MAX.
    """
    prime_dir = os.path.join(CFG["out_root"], "runs", "_prime_cache")
    nmax = int(CFG["n_max"])
    seed = int(CFG["sample_seed_base"]) + 999999  # deterministic, distinct
    _write_oracle_dataset(prime_dir, n=nmax, seed=seed, edges=edges)

    H_plat_spec = {
        "type": "normal",
        "mu": float(CFG["mu_norm"]),
        "sigma": float(np.sqrt(float(CFG["var_norm"]))),
    }

    engine = EngineParams(
        H_plat_spec=H_plat_spec,
        ref_sample_size=int(CFG["ref_sample_size"]),
        mc_reps=int(CFG["mc_reps"]),
        mc_seed=int(CFG["mc_seed"]),
        cache_tables=True,
        out_dir=prime_dir,
        baseline_cache_dir=baseline_cache_dir,
        intervals_csv_filename="intervals.csv",
        intervals_npz_filename="intervals_by_t.npz",
        prefer_npz_intervals=False,
    )

    log(f"[prime] Building baseline tables up to n={nmax} in {baseline_cache_dir}")
    run_evidence_engine(prime_dir, engine)


def _quantiles(vals: np.ndarray, q_lo: float, q_hi: float) -> Tuple[float, float, float]:
    return (
        float(np.quantile(vals, 0.50)),
        float(np.quantile(vals, q_lo)),
        float(np.quantile(vals, q_hi)),
    )


def run_full() -> None:
    edges = _make_edges()

    out_root = str(CFG["out_root"])
    fig_root = str(CFG["fig_root"])
    _ensure_dir(out_root)
    _ensure_dir(fig_root)

    baseline_cache_dir = os.path.join(out_root, "baseline_cache")
    _ensure_dir(baseline_cache_dir)

    # Write config for reproducibility
    _write_json(os.path.join(out_root, CFG["config_name"]), CFG)

    # Prime cache to avoid missing b(n) for large n
    _prime_baseline_cache(baseline_cache_dir, edges)

    # Evidence engine params (shared across runs)
    H_plat_spec = {
        "type": "normal",
        "mu": float(CFG["mu_norm"]),
        "sigma": float(np.sqrt(float(CFG["var_norm"]))),
    }
    engine = EngineParams(
        H_plat_spec=H_plat_spec,
        ref_sample_size=int(CFG["ref_sample_size"]),
        mc_reps=int(CFG["mc_reps"]),
        mc_seed=int(CFG["mc_seed"]),
        cache_tables=True,
        out_dir=None,  # write to each run_dir
        baseline_cache_dir=baseline_cache_dir,
        intervals_csv_filename="intervals.csv",
        intervals_npz_filename="intervals_by_t.npz",
        prefer_npz_intervals=False,
    )

    csv_path = os.path.join(out_root, CFG["csv_name"])
    fieldnames = ["n", "trial", "raw", "corr"]

    n_values = list(range(int(CFG["n_min"]), int(CFG["n_max"]) + 1, int(CFG["n_step"])))
    R = int(CFG["R"])
    base_seed = int(CFG["sample_seed_base"])

    step_total = len(n_values) * R
    done = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for j, n in enumerate(n_values):
            for r in range(R):
                seed = base_seed + 10_000 * j + r  # structured like old fig6/fig5 style
                run_dir = os.path.join(out_root, "runs", f"n_{n:03d}", f"trial_{r:03d}")
                _write_oracle_dataset(run_dir, n=n, seed=seed, edges=edges)

                # run evidence engine
                engine2 = EngineParams(**{**engine.__dict__, "out_dir": run_dir})
                paths = run_evidence_engine(run_dir, engine2)
                raw, corr = _read_evidence_one(paths["evidence_csv"])

                w.writerow({"n": n, "trial": r, "raw": raw, "corr": corr})

                done += 1
                if done % 200 == 0 or done == step_total:
                    log(f"[progress] {done}/{step_total} trials complete")

    log(f"[ok] wrote {csv_path}")
    plot_from_csv(csv_path)


def plot_from_csv(csv_path: str | None = None) -> None:
    out_root = str(CFG["out_root"])
    fig_root = str(CFG["fig_root"])
    _ensure_dir(fig_root)

    csv_path = csv_path or os.path.join(out_root, CFG["csv_name"])

    # load
    by_n: Dict[int, Dict[str, List[float]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            n = int(row["n"])
            by_n.setdefault(n, {"raw": [], "corr": []})
            by_n[n]["raw"].append(float(row["raw"]))
            by_n[n]["corr"].append(float(row["corr"]))

    n_values = np.array(sorted(by_n.keys()), dtype=int)
    q_lo = float(CFG["q_lo"])
    q_hi = float(CFG["q_hi"])

    raw_med, raw_lo, raw_hi = [], [], []
    cor_med, cor_lo, cor_hi = [], [], []

    for n in n_values:
        rm, rlo, rhi = _quantiles(np.asarray(by_n[n]["raw"], dtype=np.float64), q_lo, q_hi)
        cm, clo, chi = _quantiles(np.asarray(by_n[n]["corr"], dtype=np.float64), q_lo, q_hi)
        raw_med.append(rm); raw_lo.append(rlo); raw_hi.append(rhi)
        cor_med.append(cm); cor_lo.append(clo); cor_hi.append(chi)

    # plot (match old style: raw blue, corrected green, bands)
    plt.figure(figsize=(9.0, 4.4))

    plt.plot(n_values, raw_med, linewidth=3.0,
             label=r"Raw: $X(n)=W_1(\hat H_n, H_{\mathrm{plat}})$")
    plt.fill_between(n_values, raw_lo, raw_hi, alpha=0.20)

    plt.plot(n_values, cor_med, color="green", linewidth=3.0,
             label=r"Corrected: $d(n)=X(n)-b(n)$")
    plt.fill_between(n_values, cor_lo, cor_hi, color="green", alpha=0.20)

    plt.axhline(0.0, linestyle="--", linewidth=1.5)
    plt.xlim(int(CFG["n_min"]), int(CFG["n_max"]))
    plt.xlabel(r"Action count per interval $n$")
    plt.ylabel("Score")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    png = os.path.join(fig_root, CFG["figure_name"] + ".png")
    pdf = os.path.join(fig_root, CFG["figure_name"] + ".pdf")
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()

    log(f"[ok] wrote {png}")
    log(f"[ok] wrote {pdf}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot_only", action="store_true", help="Only plot from existing CSV.")
    args = ap.parse_args()

    if args.plot_only:
        plot_from_csv()
    else:
        run_full()


if __name__ == "__main__":
    main()
