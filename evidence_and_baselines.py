#!/usr/bin/env python3
"""
evidence_and_baselines.py (KDD static v2): Evidence + baselines, dense b(n) with prefix reuse.

UPDATED (Feb 2026):
  - Adds OPTIONAL support for reading intervals from a compact NPZ file
    (e.g., intervals_by_t.npz) in addition to intervals.csv.
  - Backward compatible: default remains intervals.csv.

Why:
  - Enables "plot-only" and fast experiment modes later while keeping the
    4-stage pipeline modular.
  - Lets you remove intervals.csv in the future if desired.

Inputs (from Script #1 out_dir):
  - meta.json        (must include bin_edges)
  - intervals.csv    (default; produced by Script #1)
  - intervals_by_t.npz (optional alternative format; NOT produced by Script #1 here)

Outputs (to out_dir, default input_dir):
  - evidence.csv
  - baseline_tables/baseline__<name>.{json,csv}   (or to baseline_cache_dir if provided)

Baselines per paper:
  Ours:
    - W1 baseline-corrected: w_t = W1(H_t, H_ref) - b_W1(n)
      (stored as score__/b__/w__ columns in evidence.csv)

  Moment/activity baselines:
    - bl_intensity_mean_shift
    - bl_mean_zscore
    - bl_variance_normalized
    - bl_participation_frequency  (constant 1.0 score; no baseline table; sanity check)
    - bl_moment_activity_5        (DISABLED placeholder scaffold; returns 0.0; not used in reported results)

  Advanced baselines (each needs a baseline table):
    - bl_js
    - bl_chi2
    - bl_energy

NPZ format (optional):
  intervals_by_t.npz should contain at least:
    - counts: int array shape (T, h)  (histogram counts per interval)
  Optional:
    - t: int array shape (T,) with 1-indexed t values
    - n_actions: int array shape (T,) (if missing, computed as counts.sum(axis=1))
  This script infers h from counts.shape[1] and uses meta.json bin_edges for geometry.

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np


SCHEMA_VERSION = "ev_kdd_v2_dense_bn_npz_ok"


# ============================================================
# Utilities
# ============================================================
def _stable_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _counts_to_probs(counts: np.ndarray) -> Tuple[np.ndarray, int]:
    n = int(np.sum(counts))
    if n <= 0:
        return np.zeros_like(counts, dtype=np.float64), 0
    return counts.astype(np.float64) / float(n), n

def _bin_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])

def mean_var_from_hist_probs(probs: np.ndarray, edges: np.ndarray) -> Tuple[float, float]:
    z = _bin_centers(edges)
    mu = float(np.sum(probs * z))
    var = float(np.sum(probs * (z - mu) ** 2))
    return mu, var


# ============================================================
# Distances / discrepancies
# ============================================================
def w1_hist(probs_p: np.ndarray, probs_q: np.ndarray, edges: np.ndarray) -> float:
    widths = (edges[1:] - edges[:-1]).astype(np.float64)
    cdf_diff = np.cumsum(probs_p - probs_q)
    return float(np.sum(np.abs(cdf_diff) * widths))

def js_divergence(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    """
    JS(P,Q) = 0.5 KL(P||M) + 0.5 KL(Q||M), M=(P+Q)/2
    Uses masked logs; eps avoids log(0) in float.
    """
    P = P.astype(np.float64)
    Q = Q.astype(np.float64)
    M = 0.5 * (P + Q)

    def kl(A: np.ndarray, B: np.ndarray) -> float:
        mask = A > 0
        return float(np.sum(A[mask] * np.log((A[mask] + eps) / (B[mask] + eps))))

    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)

def chi2_stat(counts: np.ndarray, ref_probs: np.ndarray, n: int, eps: float = 1e-12) -> float:
    """
    Pearson chi-square: sum_i (c_i - e_i)^2 / (e_i + eps_c)
    where e_i = n * ref_probs_i
    """
    c = counts.astype(np.float64)
    e = (float(n) * ref_probs.astype(np.float64))
    eps_c = float(eps) * max(float(n), 1.0)
    return float(np.sum((c - e) ** 2 / (e + eps_c)))

def energy_distance_sq(P: np.ndarray, Q: np.ndarray, edges: np.ndarray,
                       precomp_D: Optional[np.ndarray] = None) -> float:
    """
    E^2(P,Q)=2 P^T D Q - P^T D P - Q^T D Q
    D_ij = |z_i - z_j| on bin centers z_i.
    """
    P = P.astype(np.float64)
    Q = Q.astype(np.float64)
    if precomp_D is None:
        z = _bin_centers(edges)
        D = np.abs(z[:, None] - z[None, :]).astype(np.float64)
    else:
        D = precomp_D
    PtD = P @ D
    QtD = Q @ D
    return float(2.0 * (PtD @ Q) - (PtD @ P) - (QtD @ Q))


# ============================================================
# Context: reference histogram + sampling
# ============================================================
@dataclass(frozen=True)
class EvidenceContext:
    edges: np.ndarray
    ref_probs: np.ndarray
    ref_mu: float
    ref_var: float
    ref_sigma: float
    energy_D: Optional[np.ndarray] = None

def build_ref_probs_from_spec(spec: Dict[str, Any], edges: np.ndarray,
                              ref_sample_size: int, seed: int) -> np.ndarray:
    """
    Build reference probs at this binning by large-sample Monte Carlo.
    """
    rng = np.random.default_rng(seed)
    x_min = float(edges[0])
    x_max = float(edges[-1])

    typ = spec.get("type", "normal")
    if typ == "normal":
        mu = float(spec["mu"]); sigma = float(spec["sigma"])
        x = rng.normal(mu, sigma, size=int(ref_sample_size))
        x = np.clip(x, x_min, x_max)
    elif typ == "beta":
        a = float(spec["a"]); b = float(spec["b"])
        z = rng.beta(a, b, size=int(ref_sample_size))
        x = x_min + (x_max - x_min) * z
        x = np.clip(x, x_min, x_max)
    else:
        raise ValueError(f"Unknown H_plat_spec type: {typ}")

    counts, _ = np.histogram(x, bins=edges)
    probs, n = _counts_to_probs(counts)
    if n <= 0:
        raise ValueError("Reference sampling produced empty histogram.")
    return probs


# ============================================================
# Baseline interface + dense prefix MC baseline table
# ============================================================
class Baseline(Protocol):
    name: str
    needs_baseline_table: bool

    def score_from_interval(self, counts: np.ndarray, probs: np.ndarray, n: int, ctx: EvidenceContext) -> float:
        ...

    def baseline_table_mc(self, n_grid: List[int], ctx: EvidenceContext, mc_reps: int, mc_seed: int) -> Dict[int, float]:
        ...

class BaseBaseline:
    name: str = "base"
    needs_baseline_table: bool = False

    def score_from_interval(self, counts: np.ndarray, probs: np.ndarray, n: int, ctx: EvidenceContext) -> float:
        raise NotImplementedError

    def baseline_table_mc(self, n_grid: List[int], ctx: EvidenceContext, mc_reps: int, mc_seed: int) -> Dict[int, float]:
        """
        Dense baseline table b(n) for ALL n=1..Nmax under fixed binning (h from edges), using prefix-reuse MC trails.
        """
        if not n_grid:
            return {}
        Nmax = int(max(int(v) for v in n_grid if int(v) > 0))
        if Nmax <= 0:
            return {}

        rng = np.random.default_rng(mc_seed)
        h = int(len(ctx.ref_probs))
        acc = np.zeros(Nmax + 1, dtype=np.float64)

        for _ in range(int(mc_reps)):
            idx = rng.choice(h, size=Nmax, p=ctx.ref_probs)
            counts = np.zeros(h, dtype=np.int64)
            for n in range(1, Nmax + 1):
                counts[idx[n - 1]] += 1
                probs = counts.astype(np.float64) / float(n)
                acc[n] += float(self.score_from_interval(counts, probs, n, ctx))

        return {n: float(acc[n] / float(mc_reps)) for n in range(1, Nmax + 1)}


# ============================================================
# OUR METHOD
# ============================================================
class Ours_W1(BaseBaseline):
    name = "ours_w1"
    needs_baseline_table = True

    def score_from_interval(self, counts: np.ndarray, probs: np.ndarray, n: int, ctx: EvidenceContext) -> float:
        return w1_hist(probs, ctx.ref_probs, ctx.edges)


# ============================================================
# 5 moment/activity baselines
# ============================================================
class BL_IntensityMeanShift(BaseBaseline):
    name = "bl_intensity_mean_shift"
    needs_baseline_table = True

    def score_from_interval(self, counts: np.ndarray, probs: np.ndarray, n: int, ctx: EvidenceContext) -> float:
        mu_t, _ = mean_var_from_hist_probs(probs, ctx.edges)
        return abs(mu_t - ctx.ref_mu)

class BL_MeanZScore(BaseBaseline):
    name = "bl_mean_zscore"
    needs_baseline_table = True

    def score_from_interval(self, counts: np.ndarray, probs: np.ndarray, n: int, ctx: EvidenceContext) -> float:
        if n <= 0 or ctx.ref_sigma <= 0:
            return 0.0
        mu_t, _ = mean_var_from_hist_probs(probs, ctx.edges)
        return abs((mu_t - ctx.ref_mu) / (ctx.ref_sigma / math.sqrt(n)))

class BL_VarianceNormalized(BaseBaseline):
    name = "bl_variance_normalized"
    needs_baseline_table = True

    def score_from_interval(self, counts: np.ndarray, probs: np.ndarray, n: int, ctx: EvidenceContext) -> float:
        _, var_t = mean_var_from_hist_probs(probs, ctx.edges)
        if ctx.ref_var > 0:
            return abs((var_t - ctx.ref_var) / ctx.ref_var)
        return abs(var_t - ctx.ref_var)

class BL_ParticipationFrequency(BaseBaseline):
    name = "bl_participation_frequency"
    needs_baseline_table = False

    def score_from_interval(self, counts: np.ndarray, probs: np.ndarray, n: int, ctx: EvidenceContext) -> float:
        return 1.0

class BL_MomentActivity5_Placeholder(BaseBaseline):
    name = "bl_moment_activity_5"
    needs_baseline_table = False

    def score_from_interval(self, counts: np.ndarray, probs: np.ndarray, n: int, ctx: EvidenceContext) -> float:
        # DISABLED placeholder scaffold: implement a 5th baseline here if desired (not used in reported results).
        return 0.0


# ============================================================
# 3 advanced baselines
# ============================================================
class BL_JS(BaseBaseline):
    name = "bl_js"
    needs_baseline_table = True

    def score_from_interval(self, counts: np.ndarray, probs: np.ndarray, n: int, ctx: EvidenceContext) -> float:
        return js_divergence(probs, ctx.ref_probs)

class BL_ChiSquare(BaseBaseline):
    name = "bl_chi2"
    needs_baseline_table = True

    def score_from_interval(self, counts: np.ndarray, probs: np.ndarray, n: int, ctx: EvidenceContext) -> float:
        return chi2_stat(counts, ctx.ref_probs, n=n)

class BL_Energy(BaseBaseline):
    name = "bl_energy"
    needs_baseline_table = True

    def score_from_interval(self, counts: np.ndarray, probs: np.ndarray, n: int, ctx: EvidenceContext) -> float:
        return energy_distance_sq(probs, ctx.ref_probs, ctx.edges, precomp_D=ctx.energy_D)


# ============================================================
# Interval readers (CSV or NPZ)
# ============================================================
def _load_intervals_csv(intervals_path: str) -> Tuple[List[Dict[str, Any]], List[int], int]:
    """
    Returns:
      rows: list of dicts with keys t, counts, probs, n
      observed_n: list of n>0
      h_bins: int
    """
    import json as _json

    rows: List[Dict[str, Any]] = []
    observed_n: List[int] = []
    h_bins_seen: Optional[int] = None

    with open(intervals_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            counts = np.array(_json.loads(row["counts_json"]), dtype=np.int64)
            probs, n = _counts_to_probs(counts)
            t = int(row["t"])
            rows.append({"t": t, "counts": counts, "probs": probs, "n": n})
            if n > 0:
                observed_n.append(n)
            if "h_bins" in row and row["h_bins"] != "":
                h_bins_seen = int(row["h_bins"])

    if not rows:
        raise ValueError("intervals.csv empty")
    if not observed_n:
        raise ValueError("No intervals with n_actions > 0 found.")
    h_bins = int(h_bins_seen) if h_bins_seen is not None else int(rows[0]["counts"].shape[0])
    return rows, observed_n, h_bins


def _load_intervals_npz(intervals_npz_path: str) -> Tuple[List[Dict[str, Any]], List[int], int]:
    """
    NPZ expected:
      counts: (T,h) int
      optional: t: (T,) int (1-indexed)
      optional: n_actions: (T,) int

    Returns:
      rows: list of dicts with keys t, counts, probs, n
      observed_n: list of n>0
      h_bins: int
    """
    z = np.load(intervals_npz_path, allow_pickle=False)
    if "counts" not in z:
        raise ValueError("intervals_by_t.npz missing required array: counts")
    counts_mat = np.asarray(z["counts"])
    if counts_mat.ndim != 2:
        raise ValueError("intervals_by_t.npz: counts must be 2D (T,h)")
    T, h = counts_mat.shape

    if "t" in z:
        t_arr = np.asarray(z["t"], dtype=np.int64)
        if t_arr.shape != (T,):
            raise ValueError("intervals_by_t.npz: t must have shape (T,)")
        t_list = [int(v) for v in t_arr.tolist()]
    else:
        t_list = list(range(1, T + 1))

    if "n_actions" in z:
        n_arr = np.asarray(z["n_actions"], dtype=np.int64)
        if n_arr.shape != (T,):
            raise ValueError("intervals_by_t.npz: n_actions must have shape (T,)")
        n_list = [int(v) for v in n_arr.tolist()]
    else:
        n_list = [int(np.sum(counts_mat[i, :])) for i in range(T)]

    rows: List[Dict[str, Any]] = []
    observed_n: List[int] = []
    for i in range(T):
        counts = counts_mat[i, :].astype(np.int64)
        probs, n0 = _counts_to_probs(counts)
        # trust provided n_actions if present; else n0
        n = int(n_list[i])
        if n <= 0:
            n = int(n0)
        rows.append({"t": int(t_list[i]), "counts": counts, "probs": probs, "n": int(n)})
        if n > 0:
            observed_n.append(int(n))

    if not rows:
        raise ValueError("intervals_by_t.npz empty")
    if not observed_n:
        raise ValueError("No intervals with n_actions > 0 found.")
    return rows, observed_n, int(h)


# ============================================================
# Orchestration
# ============================================================
@dataclass(frozen=True)
class EngineParams:
    H_plat_spec: Dict[str, Any]
    ref_sample_size: int = 1_000_000

    mc_reps: int = 2000
    mc_seed: int = 0

    # Whether to load/save baseline tables on disk.
    cache_tables: bool = True

    # Where to write evidence.csv (default: input_dir).
    out_dir: Optional[str] = None

    # NEW: shared cache directory for baseline tables (per-figure).
    baseline_cache_dir: Optional[str] = None

    # NEW: interval inputs
    intervals_csv_filename: str = "intervals.csv"
    intervals_npz_filename: str = "intervals_by_t.npz"  # optional
    prefer_npz_intervals: bool = False


def _baseline_key(dataset_config_id: str, edges: np.ndarray, engine: EngineParams, baseline_name: str) -> str:
    # Baseline tables depend only on the null model + binning + MC settings.
    obj = {
        "schema": SCHEMA_VERSION,
        "baseline_name": baseline_name,
        "bin_edges": edges.tolist(),
        "H_plat_spec": engine.H_plat_spec,
        "mc_reps": engine.mc_reps,
        "mc_seed": engine.mc_seed,
        "ref_sample_size": engine.ref_sample_size,
    }
    return _sha256(_stable_dumps(obj))[:16]


def _write_table_csv(path: str, baseline_name: str, key: str, table: Dict[int, float]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["schema", "baseline_name", "baseline_key", "n", "b_n"])
        w.writeheader()
        for n in sorted(table.keys()):
            w.writerow({
                "schema": SCHEMA_VERSION,
                "baseline_name": baseline_name,
                "baseline_key": key,
                "n": int(n),
                "b_n": float(table[n]),
            })


def _write_table_json(path: str, baseline_name: str, key: str, dataset_config_id: str,
                      engine: EngineParams, table: Dict[int, float]) -> None:
    obj = {
        "schema": SCHEMA_VERSION,
        "baseline_name": baseline_name,
        "baseline_key": key,
        "dataset_config_id": dataset_config_id,
        "H_plat_spec": engine.H_plat_spec,
        "mc_reps": engine.mc_reps,
        "mc_seed": engine.mc_seed,
        "ref_sample_size": engine.ref_sample_size,
        "table": {str(int(k)): float(v) for k, v in table.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(_stable_dumps(obj) + "\n")


def _load_table_json(path: str) -> Tuple[str, Dict[int, float]]:
    obj = _read_json(path)
    key = obj["baseline_key"]
    table = {int(k): float(v) for k, v in obj["table"].items()}
    return key, table


def run_evidence_engine(input_dir: str, engine: EngineParams) -> Dict[str, str]:
    meta_path = os.path.join(input_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Expected meta.json in input_dir")

    meta = _read_json(meta_path)
    dataset_config_id = meta.get("config_id", "")

    # edges
    if "bin_edges" in meta:
        edges = np.array(meta["bin_edges"], dtype=np.float64)
    elif "config" in meta and "bin_edges" in meta["config"]:
        edges = np.array(meta["config"]["bin_edges"], dtype=np.float64)
    else:
        raise ValueError("meta.json missing bin_edges")

    # intervals: prefer NPZ if requested and present, else CSV if present, else NPZ if present
    intervals_csv_path = os.path.join(input_dir, engine.intervals_csv_filename)
    intervals_npz_path = os.path.join(input_dir, engine.intervals_npz_filename)

    rows: List[Dict[str, Any]]
    observed_n: List[int]
    h_bins: int

    if engine.prefer_npz_intervals and os.path.exists(intervals_npz_path):
        rows, observed_n, h_bins = _load_intervals_npz(intervals_npz_path)
    elif os.path.exists(intervals_csv_path):
        rows, observed_n, h_bins = _load_intervals_csv(intervals_csv_path)
    elif os.path.exists(intervals_npz_path):
        rows, observed_n, h_bins = _load_intervals_npz(intervals_npz_path)
    else:
        raise FileNotFoundError(
            f"Expected {engine.intervals_csv_filename} or {engine.intervals_npz_filename} in input_dir"
        )

    # build reference histogram at this binning
    ref_probs = build_ref_probs_from_spec(
        engine.H_plat_spec, edges, engine.ref_sample_size, seed=engine.mc_seed + 999
    )
    ref_mu, ref_var = mean_var_from_hist_probs(ref_probs, edges)
    ref_sigma = float(math.sqrt(max(ref_var, 0.0)))

    # precompute energy distance matrix once
    z = _bin_centers(edges)
    energy_D = np.abs(z[:, None] - z[None, :]).astype(np.float64)

    ctx = EvidenceContext(
        edges=edges,
        ref_probs=ref_probs,
        ref_mu=ref_mu,
        ref_var=ref_var,
        ref_sigma=ref_sigma,
        energy_D=energy_D,
    )

    baselines: List[Baseline] = [
        Ours_W1(),
        BL_IntensityMeanShift(),
        BL_MeanZScore(),
        BL_VarianceNormalized(),
        BL_ParticipationFrequency(),
        BL_MomentActivity5_Placeholder(),
        BL_JS(),
        BL_ChiSquare(),
        BL_Energy(),
    ]

    # Dense tables: compute up to Nmax = max observed n, and store for all n=1..Nmax.
    Nmax = int(max(observed_n))
    n_grid = list(range(1, Nmax + 1))

    out_dir = engine.out_dir or input_dir
    _ensure_dir(out_dir)

    tables_dir = engine.baseline_cache_dir or os.path.join(out_dir, "baseline_tables")
    _ensure_dir(tables_dir)

    baseline_tables: Dict[str, Dict[int, float]] = {}
    baseline_keys: Dict[str, str] = {}

    for bl in baselines:
        if not getattr(bl, "needs_baseline_table", False):
            continue

        key = _baseline_key(dataset_config_id, edges, engine, bl.name)
        json_path = os.path.join(tables_dir, f"baseline__{bl.name}.json")
        csv_path = os.path.join(tables_dir, f"baseline__{bl.name}.csv")

        loaded = False
        if engine.cache_tables and os.path.exists(json_path):
            try:
                key0, tab0 = _load_table_json(json_path)
                if key0 == key:
                    baseline_tables[bl.name] = tab0
                    baseline_keys[bl.name] = key
                    loaded = True
            except Exception:
                loaded = False

        if not loaded:
            tab = bl.baseline_table_mc(n_grid=n_grid, ctx=ctx, mc_reps=engine.mc_reps, mc_seed=engine.mc_seed)
            baseline_tables[bl.name] = tab
            baseline_keys[bl.name] = key
            _write_table_json(json_path, bl.name, key, dataset_config_id, engine, tab)
            _write_table_csv(csv_path, bl.name, key, tab)

    evidence_path = os.path.join(out_dir, "evidence.csv")

    fieldnames = ["schema", "dataset_config_id", "t", "n_actions"]
    for bl in baselines:
        fieldnames.append(f"score__{bl.name}")
        if getattr(bl, "needs_baseline_table", False):
            fieldnames.append(f"b__{bl.name}")
        fieldnames.append(f"w__{bl.name}")

    with open(evidence_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for r0 in rows:
            t = int(r0["t"])
            n = int(r0["n"])
            counts = r0["counts"]
            probs = r0["probs"]

            out = {
                "schema": SCHEMA_VERSION,
                "dataset_config_id": dataset_config_id,
                "t": t,
                "n_actions": n,
            }

            for bl in baselines:
                s = float(bl.score_from_interval(counts, probs, n, ctx))
                out[f"score__{bl.name}"] = s

                if getattr(bl, "needs_baseline_table", False):
                    btab = baseline_tables.get(bl.name, {})
                    b = float(btab.get(n, 0.0)) if n > 0 else 0.0
                    out[f"b__{bl.name}"] = b
                    out[f"w__{bl.name}"] = float(s - b)
                else:
                    out[f"w__{bl.name}"] = float(s)

            w.writerow(out)

    return {"evidence_csv": evidence_path, "baseline_tables_dir": tables_dir}


# ============================================================
# CLI
# ============================================================
def _json_arg(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError("Must be a JSON object/dict.")
    return obj


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Evidence engine: ours + 8 baselines, dense b(n,h) via prefix MC. (CSV or NPZ intervals)")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", default=None)

    ap.add_argument("--H_plat_spec", type=_json_arg, required=True,
                    help='e.g. \'{"type":"normal","mu":0.5,"sigma":0.2236}\' or \'{"type":"beta","a":2,"b":5}\'')

    ap.add_argument("--ref_sample_size", type=int, default=1_000_000)
    ap.add_argument("--mc_reps", type=int, default=2000)
    ap.add_argument("--mc_seed", type=int, default=0)

    ap.add_argument("--no_cache", action="store_true", default=False)
    ap.add_argument("--baseline_cache_dir", default=None,
                    help="Optional shared directory for baseline tables (per-figure cache).")

    # NEW interval options
    ap.add_argument("--intervals_csv", default="intervals.csv",
                    help="Intervals CSV filename (default: intervals.csv)")
    ap.add_argument("--intervals_npz", default="intervals_by_t.npz",
                    help="Intervals NPZ filename (default: intervals_by_t.npz)")
    ap.add_argument("--prefer_npz", action="store_true", default=False,
                    help="Prefer NPZ intervals if present.")

    args = ap.parse_args(argv)

    engine = EngineParams(
        H_plat_spec=args.H_plat_spec,
        ref_sample_size=args.ref_sample_size,
        mc_reps=args.mc_reps,
        mc_seed=args.mc_seed,
        cache_tables=(not args.no_cache),
        out_dir=args.out_dir,
        baseline_cache_dir=args.baseline_cache_dir,
        intervals_csv_filename=str(args.intervals_csv),
        intervals_npz_filename=str(args.intervals_npz),
        prefer_npz_intervals=bool(args.prefer_npz),
    )

    paths = run_evidence_engine(args.input_dir, engine)
    print("[ok] wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
