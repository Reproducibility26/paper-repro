#!/usr/bin/env python3
"""
Script #3: Exposure-based attribution — REVISED to support compact NPZ exposures.

Goal:
  Given interval evidence increments w_t (centered, baseline-corrected) and per-account
  exposures a_{u,t}, compute per-account attribution scores:

      S_m(u) = sum_{t} a_{u,t} * w_t^{(m)}

for each method/baseline m.

Inputs (in run_dir):
  - evidence.csv (from Script #2)
      Required: column "t" and one of:
        * w__<method>  (preferred; baseline-corrected)
        * d__<method>  (legacy naming from older Script #2 variants)
        * score__<method> (raw score; only if allow_raw_score_if_missing=True)

  - exposures_by_t.npz (NEW, from revised Script #1)  [preferred]
      Contains:
        t_ptr: int64 length T+1, prefix offsets
        u_ids: int32 length nnz
        a_ut : float32 length nnz

    For interval t in 1..T:
      start=t_ptr[t-1], end=t_ptr[t]
      users=u_ids[start:end], weights=a_ut[start:end]

  - participation.csv (LEGACY fallback, from old Script #1)  [optional]
      Required columns: t, u, a_ut

Outputs (to out_dir, default run_dir):
  - attribution_scores.csv
  - attribution_scores.json
  - (optional) attribution_timeseries.csv
      Writes checkpointed partial sums for users as they appear (can still be huge; off by default)

Usage (from figure script):
  from exposure_attribution import AttributionParams, run_attribution
  run_attribution(run_dir="runs/fig_shift/delta_0.05_seed0", params=AttributionParams())

CLI:
  python exposure_attribution.py --run_dir runs/fig_shift/delta_0.05_seed0

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


SCHEMA_VERSION = "attr_v2_npz_exposures"


# ============================================================
# Params
# ============================================================
@dataclass(frozen=True)
class AttributionParams:
    # Filenames (relative to run_dir)
    evidence_filename: str = "evidence.csv"

    # Prefer NPZ exposures (new pipeline). If missing, optionally fall back to CSV.
    exposures_npz_filename: str = "exposures_by_t.npz"
    participation_csv_filename: str = "participation.csv"
    allow_csv_fallback: bool = True

    # Which evidence columns to use as "interval increments" w_t:
    # Priority order:
    #   1) w__* columns (baseline-corrected, preferred)
    #   2) d__* columns (if your Script #2 used d__ naming)
    #   3) score__* columns (raw score; discouraged unless you want uncentered attribution)
    allow_raw_score_if_missing: bool = False

    # Which baselines/methods to attribute:
    # If None, auto-detect from evidence file columns.
    methods: Optional[List[str]] = None

    # Output controls
    out_dir: Optional[str] = None  # default: run_dir
    write_timeseries: bool = False  # WARNING: can be huge
    # If writing timeseries, you can downsample checkpoints (e.g., every 10 intervals)
    timeseries_stride: int = 1

    # Sorting output by which score column (accepts "w__ours_w1" or "ours_w1")
    sort_by: Optional[str] = "w__ours_w1"
    descending: bool = True


# ============================================================
# Helpers
# ============================================================
def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _read_header(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        return next(r)


def _load_evidence_increments(evidence_path: str, params: AttributionParams) -> Tuple[np.ndarray, List[str], str]:
    """
    Returns:
      w_mat: shape (T, M) where M methods, indexed by t=1..T -> row t-1
      method_names: list of method names (suffixes after chosen_prefix)
      prefix: which prefix used: 'w__' or 'd__' or 'score__'
    """
    header = _read_header(evidence_path)

    # find time column
    if "t" not in header:
        raise ValueError("evidence.csv missing required column 't'")

    prefixes = ["w__", "d__"] + (["score__"] if params.allow_raw_score_if_missing else [])
    chosen_prefix = None
    cols = []
    for pref in prefixes:
        cols = [c for c in header if c.startswith(pref)]
        if cols:
            chosen_prefix = pref
            break
    if chosen_prefix is None:
        raise ValueError(
            "No evidence increment columns found. Expected w__* (preferred), or d__*, "
            "or score__* (only if allow_raw_score_if_missing=True)."
        )

    all_methods = [c[len(chosen_prefix):] for c in cols]
    if params.methods is None:
        method_names = all_methods
    else:
        missing = [m for m in params.methods if m not in all_methods]
        if missing:
            raise ValueError(f"Requested methods not found under prefix {chosen_prefix}: {missing}")
        method_names = list(params.methods)

    t_idx = header.index("t")
    col_idx = [header.index(chosen_prefix + m) for m in method_names]

    t_list: List[int] = []
    vals: List[List[float]] = []
    with open(evidence_path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)  # skip header
        for row in r:
            t = int(row[t_idx])
            t_list.append(t)
            vals.append([float(row[j]) if row[j] != "" else 0.0 for j in col_idx])

    if not t_list:
        raise ValueError("evidence.csv appears empty")

    T = max(t_list)
    M = len(method_names)
    w_mat = np.zeros((T, M), dtype=np.float64)

    # fill; if some t missing, remains 0
    for t, v in zip(t_list, vals):
        if 1 <= t <= T:
            w_mat[t - 1, :] = np.array(v, dtype=np.float64)

    return w_mat, method_names, chosen_prefix


def _load_exposures_npz(exposures_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t_ptr: int64 (T+1,)
      u_ids: int32 (nnz,)
      a_ut : float32/float64 (nnz,)
    """
    z = np.load(exposures_path, allow_pickle=False)
    if "t_ptr" not in z or "u_ids" not in z or "a_ut" not in z:
        raise ValueError("exposures_by_t.npz missing required arrays: t_ptr, u_ids, a_ut")
    t_ptr = np.asarray(z["t_ptr"], dtype=np.int64)
    u_ids = np.asarray(z["u_ids"], dtype=np.int32)
    a_ut = np.asarray(z["a_ut"], dtype=np.float64)  # accumulate in float64
    if t_ptr.ndim != 1 or u_ids.ndim != 1 or a_ut.ndim != 1:
        raise ValueError("exposures_by_t.npz arrays must be 1D")
    if u_ids.shape[0] != a_ut.shape[0]:
        raise ValueError("exposures_by_t.npz: u_ids and a_ut must have same length")
    if t_ptr.shape[0] < 2:
        raise ValueError("exposures_by_t.npz: t_ptr must have length T+1 >= 2")
    if t_ptr[0] != 0:
        raise ValueError("exposures_by_t.npz: t_ptr[0] must be 0")
    if t_ptr[-1] != u_ids.shape[0]:
        raise ValueError("exposures_by_t.npz: t_ptr[-1] must equal nnz")
    if np.any(t_ptr[1:] < t_ptr[:-1]):
        raise ValueError("exposures_by_t.npz: t_ptr must be nondecreasing")
    return t_ptr, u_ids, a_ut


# ============================================================
# Main attribution
# ============================================================
def run_attribution(run_dir: str, params: AttributionParams) -> Dict[str, str]:
    evidence_path = os.path.join(run_dir, params.evidence_filename)
    exposures_path = os.path.join(run_dir, params.exposures_npz_filename)
    part_csv_path = os.path.join(run_dir, params.participation_csv_filename)

    if not os.path.exists(evidence_path):
        raise FileNotFoundError(f"Missing evidence file: {evidence_path}")

    out_dir = params.out_dir or run_dir
    _ensure_dir(out_dir)

    # Load w_t increments for each method
    w_mat, method_names, used_prefix = _load_evidence_increments(evidence_path, params)
    T, M = w_mat.shape

    # Prepare accumulators: dict u -> scores vector length M
    scores: Dict[int, np.ndarray] = {}

    # Optional timeseries (OFF by default)
    write_ts = bool(params.write_timeseries)
    stride = max(1, int(params.timeseries_stride))
    ts_path = os.path.join(out_dir, "attribution_timeseries.csv") if write_ts else None

    ts_f = None
    ts_w = None
    if write_ts:
        ts_f = open(ts_path, "w", newline="", encoding="utf-8")
        ts_w = csv.DictWriter(
            ts_f,
            fieldnames=["schema", "t", "u"] + [f"S__{m}" for m in method_names],
        )
        ts_w.writeheader()

    used_exposure_source = None

    # ----------------------------
    # Preferred: NPZ exposures
    # ----------------------------
    if os.path.exists(exposures_path):
        t_ptr, u_ids, a_ut = _load_exposures_npz(exposures_path)
        T_exp = int(t_ptr.shape[0] - 1)

        # Be tolerant: if evidence has different T than exposures (e.g., you truncated),
        # use the overlap.
        T_use = min(T, T_exp)

        for t in range(1, T_use + 1):
            start = int(t_ptr[t - 1])
            end = int(t_ptr[t])
            if end <= start:
                continue

            w_t = w_mat[t - 1, :]  # (M,)
            users = u_ids[start:end]
            weights = a_ut[start:end]

            # accumulate per participating account in interval t
            for u, a in zip(users, weights):
                uid = int(u)
                if uid not in scores:
                    scores[uid] = np.zeros((M,), dtype=np.float64)
                scores[uid] += float(a) * w_t

                if write_ts and (t % stride) == 0:
                    rec = {"schema": SCHEMA_VERSION, "t": t, "u": uid}
                    Su = scores[uid]
                    for j, m in enumerate(method_names):
                        rec[f"S__{m}"] = float(Su[j])
                    ts_w.writerow(rec)

        used_exposure_source = f"npz:{params.exposures_npz_filename}"

    # ----------------------------
    # Fallback: legacy CSV participation
    # ----------------------------
    else:
        if not params.allow_csv_fallback:
            raise FileNotFoundError(
                f"Missing {params.exposures_npz_filename} in run_dir and allow_csv_fallback=False"
            )
        if not os.path.exists(part_csv_path):
            raise FileNotFoundError(
                f"Missing exposures NPZ ({params.exposures_npz_filename}) and legacy participation CSV ({params.participation_csv_filename})."
            )

        with open(part_csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            required = {"t", "u", "a_ut"}
            missing = required - set(r.fieldnames or [])
            if missing:
                raise ValueError(f"participation.csv missing columns: {sorted(missing)}")

            for row in r:
                t = int(row["t"])
                if t <= 0 or t > T:
                    continue
                u = int(row["u"])
                a = float(row["a_ut"])
                w_t = w_mat[t - 1, :]

                if u not in scores:
                    scores[u] = np.zeros((M,), dtype=np.float64)
                scores[u] += a * w_t

                if write_ts and (t % stride) == 0:
                    rec = {"schema": SCHEMA_VERSION, "t": t, "u": u}
                    Su = scores[u]
                    for j, m in enumerate(method_names):
                        rec[f"S__{m}"] = float(Su[j])
                    ts_w.writerow(rec)

        used_exposure_source = f"csv:{params.participation_csv_filename}"

    if ts_f is not None:
        ts_f.close()

    # Write final per-account scores
    out_csv = os.path.join(out_dir, "attribution_scores.csv")
    out_json = os.path.join(out_dir, "attribution_scores.json")

    # Determine sorting key
    sort_key = params.sort_by
    if sort_key is not None:
        for pref in ("w__", "d__", "score__"):
            if sort_key.startswith(pref):
                sort_key = sort_key[len(pref):]
                break
        if sort_key not in method_names:
            sort_key = None  # fallback: no sort

    rows_out = []
    for u, vec in scores.items():
        rec = {"schema": SCHEMA_VERSION, "u": int(u)}
        for j, m in enumerate(method_names):
            rec[f"S__{m}"] = float(vec[j])
        rows_out.append(rec)

    if sort_key is not None:
        j = method_names.index(sort_key)
        rows_out.sort(key=lambda r0: r0[f"S__{method_names[j]}"], reverse=bool(params.descending))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["schema", "u"] + [f"S__{m}" for m in method_names]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in rows_out:
            w.writerow(rec)

    meta = {
        "schema": SCHEMA_VERSION,
        "run_dir": os.path.abspath(run_dir),
        "evidence_file": params.evidence_filename,
        "used_increment_prefix": used_prefix,
        "methods": method_names,
        "used_exposure_source": used_exposure_source,
        "T_evidence": int(T),
        "outputs": {
            "attribution_scores_csv": os.path.abspath(out_csv),
            "attribution_timeseries_csv": os.path.abspath(ts_path) if write_ts else None,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=2) + "\n")

    out_paths = {"attribution_scores_csv": out_csv, "attribution_scores_json": out_json}
    if write_ts:
        out_paths["attribution_timeseries_csv"] = ts_path
    return out_paths


# ============================================================
# CLI
# ============================================================
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Script #3: exposure-based attribution (NPZ exposures preferred; CSV fallback)")
    ap.add_argument("--run_dir", required=True, help="Directory containing evidence.csv and exposures_by_t.npz (or participation.csv)")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: run_dir)")

    ap.add_argument("--allow_raw_score_if_missing", action="store_true", default=False)

    ap.add_argument("--methods", default=None,
                    help='Optional JSON list of method names (without prefix), e.g. \'["ours_w1","bl_js","bl_chi2"]\'')

    ap.add_argument("--write_timeseries", action="store_true", default=False)
    ap.add_argument("--timeseries_stride", type=int, default=1)

    ap.add_argument("--sort_by", default="w__ours_w1",
                    help='Sort key: "w__ours_w1" or "ours_w1". Set "" to disable sorting.')
    ap.add_argument("--ascending", action="store_true", default=False)

    ap.add_argument("--exposures_npz", default="exposures_by_t.npz",
                    help="NPZ exposures filename (default: exposures_by_t.npz)")
    ap.add_argument("--participation_csv", default="participation.csv",
                    help="Legacy participation CSV filename (default: participation.csv)")
    ap.add_argument("--no_csv_fallback", action="store_true", default=False,
                    help="Disable fallback to participation.csv if NPZ is missing.")

    args = ap.parse_args(argv)

    methods = None
    if args.methods:
        methods = json.loads(args.methods)
        if not isinstance(methods, list):
            raise ValueError("--methods must be a JSON list")

    sort_by = args.sort_by.strip()
    if sort_by == "":
        sort_by = None

    params = AttributionParams(
        out_dir=args.out_dir,
        allow_raw_score_if_missing=args.allow_raw_score_if_missing,
        methods=[str(m) for m in methods] if methods is not None else None,
        write_timeseries=args.write_timeseries,
        timeseries_stride=args.timeseries_stride,
        sort_by=sort_by,
        descending=(not args.ascending),
        exposures_npz_filename=str(args.exposures_npz),
        participation_csv_filename=str(args.participation_csv),
        allow_csv_fallback=(not args.no_csv_fallback),
    )

    outputs = run_attribution(args.run_dir, params)
    print("[ok] wrote:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
