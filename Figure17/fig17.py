#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig17.py — Figure 17: Impact of platform reference resolution on time-to-separation (Geo-only).

Purpose:
  This figure studies how the resolution of the platform-declared reference histogram
  (number of bins $H_{\mathrm{plat}}$) affects the speed at which coalition and normal
  users separate over time. The metric of interest is the ROC–AUC trajectory of the
  cumulative attribution score.

Definition:
  For each resolution $h$, the cumulative score is
      S_u(t) = \sum_{s \le t} a_{u,s} \, d_s ,
  where
      d_s = W_1\bigl(H_s^{(h)}, H_{\mathrm{plat}}^{(h)}\bigr) - b_h(n_s)
  and $b_h(n)$ is the null baseline calibrated under the platform reference at
  resolution $h$.

Experimental design:
  - Generate fine-grained telemetry once per run (fixed generator binning).
  - Coarsen the same histograms to each platform resolution $h$ using contiguous merges.
  - For each $h$ and each exposure regime $R_{\exp}$, compute the ROC–AUC trajectory
    at regular evaluation intervals.
  - All attribution uses the geometric Wasserstein-1 score; no baseline comparisons
    are shown.

Modes:
  - Full run (default): simulate, cache results to NPZ, then plot.
  - Plot-only (--plot_only): load cached NPZ and regenerate the figure.

Outputs (unchanged):
  - data/fig17_cache_hplat_resolution_auc_traj.npz
  - figure/fig17_hplat_resolution_auc_traj.(png|pdf)
  - data/fig17_config.json
  - figure/fig17_manifest.json

Usage:
  python fig17.py
  python fig17.py --plot_only

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional

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
# Path setup: script in Figure17, modules in parent dir
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if A_DIR not in sys.path:
    sys.path.insert(0, A_DIR)

from rotation_generator import RotationParams, generate_rotation_dataset
from evidence_and_baselines import (
    Ours_W1,
    build_ref_probs_from_spec,
    w1_hist,
)

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
# Config (CLI can override key knobs)
# ------------------------------------------------------------
CFG: Dict[str, Any] = dict(
    figure="fig17_hplat_resolution_auc_traj",

    # timeline
    T=2000,
    eval_every=10,

    # populations
    N_norm=20000,
    N_coal=2000,

    # behavior
    p_norm=0.04,
    p_on=0.40,  # default from old script
    R_list=[1.0, 2.0],

    # distributions
    mu_norm=0.50,
    sigma_norm=0.20,
    delta=0.06,   # coalition mean shift

    # fine telemetry binning (generator bins)
    bins=100,
    x_min=0.0,
    x_max=1.0,

    # platform resolutions to test (h <= bins)
    h_list=[2, 5, 10, 20, 25, 40, 50, 100],

    # baseline + platform ref sampling (per h)
    ref_sample_size=300_000,     # for build_ref_probs_from_spec
    baseline_mc_reps=200,        # for baseline_table_mc
    baseline_seed=999,

    # repeats
    seeds=5,
    base_seed=12345,

    # cache name
    cache_npz_name="fig17_cache_hplat_resolution_auc_traj.npz",
)


# ------------------------------------------------------------
# Coarsening utilities: contiguous merges (no divisibility needed)
# ------------------------------------------------------------
def _partition_edges(Bfine: int, h: int) -> np.ndarray:
    cuts = np.linspace(0, Bfine, h + 1)
    idx = np.floor(cuts).astype(int)
    idx[0] = 0
    idx[-1] = Bfine
    for k in range(1, len(idx)):
        idx[k] = max(idx[k], idx[k - 1])
    idx[-1] = Bfine
    return idx

def coarsen_counts_1d(counts_fine: np.ndarray, h: int) -> np.ndarray:
    counts_fine = np.asarray(counts_fine, dtype=np.int64)
    Bfine = int(counts_fine.shape[0])
    if h == Bfine:
        return counts_fine.copy()
    if h > Bfine:
        raise ValueError(f"h={h} must be <= Bfine={Bfine}")
    idx = _partition_edges(Bfine, h)
    out = np.zeros(h, dtype=np.int64)
    for j in range(h):
        a = int(idx[j])
        b = int(idx[j + 1])
        if b > a:
            out[j] = int(counts_fine[a:b].sum())
    return out

def coarsen_counts_2d(hist_fine: np.ndarray, h: int) -> np.ndarray:
    hist_fine = np.asarray(hist_fine, dtype=np.int64)
    T, Bfine = hist_fine.shape
    if h == Bfine:
        return hist_fine.copy()
    if h > Bfine:
        raise ValueError(f"h={h} must be <= Bfine={Bfine}")
    idx = _partition_edges(Bfine, h)
    out = np.zeros((T, h), dtype=np.int64)
    for j in range(h):
        a = int(idx[j])
        b = int(idx[j + 1])
        if b > a:
            out[:, j] = hist_fine[:, a:b].sum(axis=1)
    return out


# ------------------------------------------------------------
# NPZ exposures loader (CSR-by-interval)
# ------------------------------------------------------------
def load_exposures_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    if "t_ptr" not in z or "u_ids" not in z or "a_ut" not in z:
        raise ValueError("exposures_by_t.npz missing required arrays: t_ptr, u_ids, a_ut")
    t_ptr = np.asarray(z["t_ptr"], dtype=np.int64)
    u_ids = np.asarray(z["u_ids"], dtype=np.int32)
    a_ut = np.asarray(z["a_ut"], dtype=np.float64)
    if t_ptr.ndim != 1 or u_ids.ndim != 1 or a_ut.ndim != 1:
        raise ValueError("exposures_by_t.npz arrays must be 1D")
    if u_ids.shape[0] != a_ut.shape[0]:
        raise ValueError("exposures_by_t.npz: u_ids and a_ut must have same length")
    if t_ptr[0] != 0 or t_ptr[-1] != u_ids.shape[0]:
        raise ValueError("exposures_by_t.npz: bad t_ptr")
    if np.any(t_ptr[1:] < t_ptr[:-1]):
        raise ValueError("exposures_by_t.npz: t_ptr must be nondecreasing")
    return t_ptr, u_ids, a_ut


# ------------------------------------------------------------
# Deterministic ROC-AUC (stable ties by u)
# ------------------------------------------------------------
def roc_auc(scores: np.ndarray, y01: np.ndarray) -> float:
    u = np.arange(scores.shape[0], dtype=np.int64)
    order = np.lexsort((u, -scores))
    y = y01[order].astype(np.int8)
    s = scores[order].astype(np.float64)

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


# ------------------------------------------------------------
# Core: AUC trajectory given hist (T,h), n_actions, exposures CSR, and baseline b(n)
# ------------------------------------------------------------
def auc_trajectory(
    hist_h: np.ndarray,                 # (T, h)
    n_actions: np.ndarray,              # (T,)
    t_ptr: np.ndarray,                  # (T+1,)
    u_ids: np.ndarray,                  # (nnz,)
    a_ut: np.ndarray,                   # (nnz,)
    y01: np.ndarray,                    # (N_total,)
    ref_probs_h: np.ndarray,            # (h,) platform reference probs at resolution h
    bin_edges_h: np.ndarray,            # (h+1,)
    b_table: Dict[int, float],          # b(n) for this h
    eval_every: int,
) -> Tuple[np.ndarray, np.ndarray]:
    T = int(hist_h.shape[0])
    N_total = int(y01.shape[0])
    scores = np.zeros(N_total, dtype=np.float64)

    times: List[int] = []
    aucs: List[float] = []

    for t in range(1, T + 1):
        counts_t = hist_h[t - 1, :]
        n_t = int(n_actions[t - 1])

        if n_t <= 0:
            d_t = 0.0
        else:
            probs_t = counts_t.astype(np.float64) / float(n_t)
            W1 = float(w1_hist(probs_t, ref_probs_h, bin_edges_h))
            d_t = float(W1 - float(b_table.get(n_t, 0.0)))

        # add to participants in interval t
        start = int(t_ptr[t - 1])
        end = int(t_ptr[t])
        if end > start and d_t != 0.0:
            users = u_ids[start:end]
            weights = a_ut[start:end]
            for u, a in zip(users, weights):
                scores[int(u)] += float(a) * d_t

        if (t % int(eval_every) == 0) or (t == T):
            times.append(t)
            aucs.append(roc_auc(scores, y01))

    return np.asarray(times, dtype=np.int64), np.asarray(aucs, dtype=np.float64)


# ------------------------------------------------------------
# Baseline table for a given h (dense up to Nmax)
# ------------------------------------------------------------
def baseline_table_for_h(
    h: int,
    ref_probs_h: np.ndarray,
    bin_edges_h: np.ndarray,
    Nmax: int,
    mc_reps: int,
    mc_seed: int,
) -> Dict[int, float]:
    """
    For fig17, b_h(n) is defined under the platform null at resolution h:
      - draw n samples from ref_probs_h
      - compute W1(hist, ref_probs_h)
      - average over MC, dense for n=1..Nmax
    """
    # minimal ctx object for baseline_table_mc:
    # Ours_W1.score_from_interval only needs edges + ref_probs.
    class _Ctx:
        def __init__(self, edges, ref_probs):
            self.edges = edges
            self.ref_probs = ref_probs
            self.ref_mu = 0.0
            self.ref_var = 1.0
            self.ref_sigma = 1.0
            self.energy_D = None

    ctx = _Ctx(bin_edges_h, ref_probs_h)
    bl = Ours_W1()
    n_grid = list(range(1, int(Nmax) + 1))
    tab = bl.baseline_table_mc(n_grid=n_grid, ctx=ctx, mc_reps=int(mc_reps), mc_seed=int(mc_seed))
    return {int(k): float(v) for k, v in tab.items()}


# ------------------------------------------------------------
# Cache helpers (.npz)
# ------------------------------------------------------------
def cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, CFG["cache_npz_name"])

def save_cache_npz(path: str, times: np.ndarray, R_list: np.ndarray, h_list: np.ndarray, A: np.ndarray, meta: Dict[str, float]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    np.savez_compressed(path, times=times, R_list=R_list, h_list=h_list, auc=A, **meta)
    log(f"[cache] saved -> {path}")

def load_cache_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    _require(path)
    z = np.load(path, allow_pickle=False)
    times = z["times"]
    R_list = z["R_list"]
    h_list = z["h_list"]
    A = z["auc"]
    meta: Dict[str, float] = {}
    for k in ["eval_every", "T"]:
        if k in z.files:
            meta[k] = float(z[k])
    log(f"[cache] loaded <- {path}")
    return times, R_list, h_list, A, meta


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def plot_from_cache(times: np.ndarray, R_list: np.ndarray, h_list: np.ndarray, A: np.ndarray, out_png: str, out_pdf: str) -> None:
    step("FIG17 — STEP 3/3: PLOT")

    n_panels = int(len(R_list))
    fig, axes = plt.subplots(1, n_panels, figsize=(6.6 * n_panels, 4.6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for pi, R_exp in enumerate(R_list):
        ax = axes[pi]
        ax.set_title(rf"$R_{{\mathrm{{exp}}}}={float(R_exp):g}$")

        for hj, h in enumerate(h_list):
            trajs = A[pi, hj, :, :]  # (seeds, T_eval)
            med = np.nanmedian(trajs, axis=0)
            lo = np.nanpercentile(trajs, 5, axis=0)
            hi = np.nanpercentile(trajs, 95, axis=0)

            ax.plot(times, med, lw=2.6, label=rf"$h={int(h)}$")
            ax.fill_between(times, lo, hi, alpha=0.18)

        ax.axhline(0.8, ls="--", lw=1.2, alpha=0.7)
        ax.axhline(0.9, ls="--", lw=1.2, alpha=0.7)
        ax.set_xlabel(r"Time $t$ (intervals)")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlim(float(times.min()) - 10, float(times.max()) + 10)
        ax.legend(loc="lower right", frameon=True)
        ax.grid(False)

    axes[0].set_ylabel(r"ROC--AUC (cumulative score up to time $t$)")
    fig.tight_layout()

    _ensure_dir(os.path.dirname(out_png) or ".")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    log(f"[plot] wrote -> {out_png}")
    log(f"[plot] wrote -> {out_pdf}")


# ------------------------------------------------------------
# Main: full run
# ------------------------------------------------------------
def compute_and_cache(data_dir: str, runs_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    step("FIG17 — STEP 1/3: SIMULATE + SCORE (FULL RUN)")

    R_list = np.asarray([float(x) for x in CFG["R_list"]], dtype=np.float64)
    h_list = np.asarray(sorted(set(int(x) for x in CFG["h_list"])), dtype=np.int64)

    if int(CFG["bins"]) < int(h_list.max()):
        raise RuntimeError(f"bins={CFG['bins']} must be >= max(h_list)={int(h_list.max())}")

    times_ref: Optional[np.ndarray] = None
    A: Optional[np.ndarray] = None  # (R, h, seeds, T_eval)

    # fine bin edges from generator domain
    x_min = float(CFG["x_min"])
    x_max = float(CFG["x_max"])
    bins = int(CFG["bins"])
    bin_edges_fine = np.linspace(x_min, x_max, bins + 1, dtype=np.float64)

    for ri, R_exp in enumerate(R_list):
        step(f"R_exp = {float(R_exp):g}")
        for si in range(int(CFG["seeds"])):
            seed = int(CFG["base_seed"]) + si
            run_dir = os.path.join(runs_dir, f"R{float(R_exp):g}".replace(".", "p"), f"seed_{si}")
            _ensure_dir(run_dir)

            # exposure matching via k_on
            k_on = int(round(float(R_exp) * float(CFG["p_norm"]) * int(CFG["N_coal"]) / max(float(CFG["p_on"]), 1e-9)))
            k_on = max(0, min(int(CFG["N_coal"]), k_on))

            log(f"[run] R_exp={float(R_exp):g} seed_idx={si} seed={seed} k_on={k_on} -> {run_dir}")

            params = RotationParams(
                T=int(CFG["T"]),
                N_norm=int(CFG["N_norm"]),
                N_coal=int(CFG["N_coal"]),
                p_norm=float(CFG["p_norm"]),
                p_on=float(CFG["p_on"]),
                k_on=int(k_on),
                bins=bins,
                x_min=x_min,
                x_max=x_max,
                seed=int(seed),
                mu_norm=float(CFG["mu_norm"]),
                sigma_norm=float(CFG["sigma_norm"]),
                D_norm={"type": "normal", "mu": float(CFG["mu_norm"]), "sigma": float(CFG["sigma_norm"])},
                D_coal={"type": "normal_shift", "delta": float(CFG["delta"])},
                actions_per_participant_norm=1,
                actions_per_participant_coal=1,
            )
            paths = generate_rotation_dataset(params, out_dir=run_dir, write_participation_csv=False)

            meta = _read_json(os.path.join(run_dir, "meta.json"))
            y01 = np.asarray(meta["arrays"]["is_coal_member"], dtype=np.int8)

            # load intervals (fine)
            import csv as _csv
            import json as _json

            intervals_csv = os.path.join(run_dir, "intervals.csv")
            _require(intervals_csv)
            counts_list = []
            n_list = []
            with open(intervals_csv, "r", encoding="utf-8") as f:
                r = _csv.DictReader(f)
                for row in r:
                    n_list.append(int(row["n_actions"]))
                    counts = np.asarray(_json.loads(row["counts_json"]), dtype=np.int64)
                    counts_list.append(counts)
            hist_fine = np.stack(counts_list, axis=0)          # (T, bins)
            n_actions = np.asarray(n_list, dtype=np.int64)      # (T,)

            # load exposures (CSR-by-interval)
            t_ptr, u_ids, a_ut = load_exposures_npz(os.path.join(run_dir, "exposures_by_t.npz"))

            # score for each h
            for hj, h in enumerate(h_list):
                h_int = int(h)
                log(f"  [h] {h_int} ({hj+1}/{len(h_list)})")

                # coarsen telemetry histograms to h
                hist_h = coarsen_counts_2d(hist_fine, h_int)  # (T,h)
                bin_edges_h = np.linspace(x_min, x_max, h_int + 1, dtype=np.float64)

                # platform reference at resolution h via MC, consistent with new oracle
                ref_probs_h = build_ref_probs_from_spec(
                    spec={"type": "normal", "mu": float(CFG["mu_norm"]), "sigma": float(CFG["sigma_norm"])},
                    edges=bin_edges_h,
                    ref_sample_size=int(CFG["ref_sample_size"]),
                    seed=int(CFG["baseline_seed"]) + 777 + 1000 * h_int + si,
                )

                # baseline table b_h(n): dense up to Nmax observed in this run
                Nmax = int(np.max(n_actions)) if int(np.max(n_actions)) > 0 else 1
                b_table = baseline_table_for_h(
                    h=h_int,
                    ref_probs_h=ref_probs_h,
                    bin_edges_h=bin_edges_h,
                    Nmax=Nmax,
                    mc_reps=int(CFG["baseline_mc_reps"]),
                    mc_seed=int(CFG["baseline_seed"]) + 1000 * h_int + si,
                )

                times, aucs = auc_trajectory(
                    hist_h=hist_h,
                    n_actions=n_actions,
                    t_ptr=t_ptr,
                    u_ids=u_ids,
                    a_ut=a_ut,
                    y01=y01,
                    ref_probs_h=ref_probs_h,
                    bin_edges_h=bin_edges_h,
                    b_table=b_table,
                    eval_every=int(CFG["eval_every"]),
                )

                if times_ref is None:
                    times_ref = times
                    A = np.full((len(R_list), len(h_list), int(CFG["seeds"]), len(times_ref)), np.nan, dtype=np.float64)
                else:
                    if not np.array_equal(times_ref, times):
                        raise RuntimeError("Evaluation times mismatch across runs; check T/eval_every.")

                assert A is not None
                A[ri, hj, si, :] = aucs

    assert times_ref is not None and A is not None
    step("FIG17 — STEP 2/3: WRITE CACHE")
    save_cache_npz(
        cache_path(data_dir),
        times_ref,
        R_list,
        h_list,
        A,
        meta={"eval_every": float(CFG["eval_every"]), "T": float(CFG["T"])},
    )
    return times_ref, R_list, h_list, A


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Fig17: impact of platform ref resolution h on AUC trajectory (new oracles)")
    ap.add_argument("--plot_only", action="store_true", default=False)

    # minimal overrides
    ap.add_argument("--T", type=int, default=int(CFG["T"]))
    ap.add_argument("--eval_every", type=int, default=int(CFG["eval_every"]))
    ap.add_argument("--seeds", type=int, default=int(CFG["seeds"]))
    ap.add_argument("--bins", type=int, default=int(CFG["bins"]))
    ap.add_argument("--baseline_mc_reps", type=int, default=int(CFG["baseline_mc_reps"]))
    ap.add_argument("--ref_sample_size", type=int, default=int(CFG["ref_sample_size"]))

    args = ap.parse_args(argv)
    CFG["T"] = int(args.T)
    CFG["eval_every"] = int(args.eval_every)
    CFG["seeds"] = int(args.seeds)
    CFG["bins"] = int(args.bins)
    CFG["baseline_mc_reps"] = int(args.baseline_mc_reps)
    CFG["ref_sample_size"] = int(args.ref_sample_size)

    data_dir = os.path.join(THIS_DIR, "data")
    fig_dir = os.path.join(THIS_DIR, "figure")
    runs_dir = os.path.join(data_dir, "runs")
    _ensure_dir(data_dir)
    _ensure_dir(fig_dir)
    _ensure_dir(runs_dir)

    out_png = os.path.join(fig_dir, "fig17_hplat_resolution_auc_traj.png")
    out_pdf = os.path.join(fig_dir, "fig17_hplat_resolution_auc_traj.pdf")
    cache_file = cache_path(data_dir)

    step("FIG17 START")
    log(f"Mode: {'PLOT-ONLY' if args.plot_only else 'FULL'}")
    log(f"T={CFG['T']} eval_every={CFG['eval_every']} seeds={CFG['seeds']}")
    log(f"bins={CFG['bins']} h_list={CFG['h_list']}")
    log(f"baseline_mc_reps={CFG['baseline_mc_reps']} ref_sample_size={CFG['ref_sample_size']}")
    log(f"cache={os.path.abspath(cache_file)}")

    if args.plot_only:
        times, R_list, h_list, A, meta = load_cache_npz(cache_file)
        plot_from_cache(times, R_list, h_list, A, out_png=out_png, out_pdf=out_pdf)
    else:
        times, R_list, h_list, A = compute_and_cache(data_dir=data_dir, runs_dir=runs_dir)
        plot_from_cache(times, R_list, h_list, A, out_png=out_png, out_pdf=out_pdf)

        # config + manifest
        _write_json(os.path.join(data_dir, "fig17_config.json"), dict(CFG=CFG, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        _write_json(os.path.join(fig_dir, "fig17_manifest.json"), dict(
            CFG=CFG,
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            outputs=dict(out_png=os.path.abspath(out_png), out_pdf=os.path.abspath(out_pdf), cache=os.path.abspath(cache_file)),
            runs_dir=os.path.abspath(runs_dir),
        ))

    step("FIG17 DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
