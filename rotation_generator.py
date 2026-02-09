#!/usr/bin/env python3
"""
Rotation-model data generator (Script #1) — UPDATED to avoid huge intermediate CSV.

Key change (Feb 2026):
  - We NO LONGER write the per-event participation log as participation.csv by default.
  - Instead we write a compact CSR-by-interval exposure file: exposures_by_t.npz
    containing arrays (t_ptr, u_ids, a_ut), plus small metadata (schema/config_id/T).

This preserves the 4-stage pipeline:
  (1) generate data -> intervals.csv + exposures_by_t.npz
  (2) compute evidence increments w_t (Script #2) -> evidence.csv
  (3) compute per-account scores S(u) (Script #3) using exposures_by_t.npz
      (participation.csv is only needed as a legacy fallback if NPZ is missing)
  (4) compute AUC/curves (Script #4)

USAGE (from figure scripts):
    from rotation_generator import RotationParams, generate_rotation_dataset

    params = RotationParams(
        N_norm=10000, N_coal=200, T=4000,
        p_norm=0.03, p_on=0.1, k_on=30,
        bins=50, x_min=0.0, x_max=1.0, seed=1,
        D_norm={"type":"normal","mu":0.5,"sigma":0.2236},
        D_coal={"type":"normal_shift","delta":0.05},
    )
    paths = generate_rotation_dataset(params, out_dir="runs/fig3_caseA")

USAGE (CLI, no config.json):
    python rotation_generator.py --out runs/test         --N_norm 10000 --N_coal 100 --T 4000         --p_norm 0.03 --p_on 0.1 --k_on 30         --bins 50 --x_min 0 --x_max 1         --seed 123         --D_norm '{"type":"normal","mu":0.5,"sigma":0.22360679775}'         --D_coal '{"type":"normal_shift","delta":0.05}'

Outputs in out_dir:
    - meta.json
    - intervals.csv
    - exposures_by_t.npz          (NEW default; preferred by downstream scripts)
    - participation.csv           (OPTIONAL, only if --write_participation_csv)

Notes:
  - intervals.csv stores histogram counts as JSON string in one cell (counts_json).
  - exposures_by_t.npz stores a sparse representation of exposures a_{u,t}:
        t_ptr: int64 length (T+1), prefix offsets into u_ids/a_ut
        u_ids: int32 length nnz (total participant entries across all t)
        a_ut : float32 length nnz
    For interval t in {1..T} (1-indexed), with t_ptr length T+1:
        start = t_ptr[t-1], end = t_ptr[t]
        users = u_ids[start:end], weights = a_ut[start:end]
  - meta.json stores bin_edges + full params + hash (config_id) + optional git_commit.

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

SCHEMA_VERSION = "rot_v2_npz_exposures"


# -----------------------------
# Parameters
# -----------------------------
@dataclass(frozen=True)
class RotationParams:
    # Population sizes
    N_norm: int = 10_000
    N_coal: int = 100

    # Time horizon
    T: int = 4_000

    # Participation
    p_norm: float = 0.03      # each normal account active with prob p_norm per interval
    p_on: float = 0.10        # coalition campaign ON with prob p_on per interval
    k_on: int = 30            # if ON, rotate k_on consecutive coalition accounts

    # Signal space / histogram
    bins: int = 50
    x_min: float = 0.0
    x_max: float = 1.0

    # Defaults for normal distribution, used by some spec types
    mu_norm: float = 0.50
    sigma_norm: float = float(np.sqrt(0.05))

    # Distribution specs
    # Examples:
    #   {"type":"normal","mu":0.5,"sigma":0.22}
    #   {"type":"normal_shift","delta":0.05}
    D_norm: Optional[Dict[str, Any]] = None
    D_coal: Optional[Dict[str, Any]] = None

    # Exposure weights a_{u,t}
    actions_per_participant_norm: int = 1
    actions_per_participant_coal: int = 1

    # RNG
    seed: int = 0


def _validate(p: RotationParams) -> None:
    if p.N_norm <= 0 or p.N_coal <= 0:
        raise ValueError("N_norm and N_coal must be positive.")
    if p.T <= 0:
        raise ValueError("T must be positive.")
    if not (0.0 <= p.p_norm <= 1.0):
        raise ValueError("p_norm must be in [0,1].")
    if not (0.0 <= p.p_on <= 1.0):
        raise ValueError("p_on must be in [0,1].")
    if not (0 <= p.k_on <= p.N_coal):
        raise ValueError("k_on must be in [0, N_coal].")
    if p.bins < 2:
        raise ValueError("bins must be >= 2.")
    if not (p.x_max > p.x_min):
        raise ValueError("x_max must be > x_min.")
    if p.actions_per_participant_norm <= 0 or p.actions_per_participant_coal <= 0:
        raise ValueError("actions_per_participant_* must be positive.")
    if p.sigma_norm <= 0:
        raise ValueError("sigma_norm must be positive.")


# -----------------------------
# Sampling helpers
# -----------------------------
def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def _sample(
    spec: Dict[str, Any],
    size: int,
    rng: np.random.Generator,
    *,
    mu_norm: float,
    sigma_norm: float,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    typ = spec.get("type", "normal")

    if typ == "normal":
        mu = float(spec.get("mu", mu_norm))
        sigma = float(spec.get("sigma", sigma_norm))
        return _clip(rng.normal(mu, sigma, size=size), x_min, x_max)

    if typ == "normal_shift":
        delta = float(spec.get("delta", 0.0))
        return _clip(rng.normal(mu_norm + delta, sigma_norm, size=size), x_min, x_max)

    if typ == "normal_varscale":
        var_scale = float(spec.get("var_scale", 1.0))
        sigma = float(np.sqrt(max(var_scale, 0.0)) * sigma_norm)
        return _clip(rng.normal(mu_norm, sigma, size=size), x_min, x_max)

    if typ == "beta":
        a = float(spec["a"])
        b = float(spec["b"])
        z = rng.beta(a, b, size=size)
        return _clip(x_min + (x_max - x_min) * z, x_min, x_max)

    if typ == "bimodal_matched":
        m = float(spec.get("mode_sep", 0.25))
        m = min(m, float(sigma_norm) - 1e-6) if sigma_norm > 1e-6 else 0.0
        sigma_comp2 = max(float(sigma_norm**2) - m**2, 1e-12)
        sigma_comp = float(np.sqrt(sigma_comp2))
        signs = rng.integers(0, 2, size=size) * 2 - 1  # +/-1
        means = mu_norm + signs * m
        return _clip(rng.normal(means, sigma_comp, size=size), x_min, x_max)

    raise ValueError(f"Unknown distribution type: {typ}")


def _rotate_k(ids_local: np.ndarray, k: int, ptr: int) -> Tuple[np.ndarray, int]:
    if k <= 0:
        return np.array([], dtype=np.int32), ptr
    n = len(ids_local)
    if n == 0:
        return np.array([], dtype=np.int32), ptr
    if k >= n:
        return ids_local.copy(), 0

    end = ptr + k
    if end <= n:
        return ids_local[ptr:end].astype(np.int32), (end % n)

    first = ids_local[ptr:]
    rem = end - n
    second = ids_local[:rem]
    chosen = np.concatenate([first, second])
    return chosen.astype(np.int32), (rem % n)


# -----------------------------
# Hashing / git
# -----------------------------
def _stable_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return out or None
    except Exception:
        return None


# -----------------------------
# Main generator API
# -----------------------------
def generate_rotation_dataset(
    params: RotationParams,
    out_dir: str,
    *,
    write_participation_csv: bool = False,
) -> Dict[str, str]:
    """
    Generate one dataset run and write:
      - meta.json
      - intervals.csv
      - exposures_by_t.npz
      - (optional) participation.csv if write_participation_csv=True

    Returns paths dict.
    """
    _validate(params)
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(params.seed)

    # distribution defaults
    D_norm = params.D_norm if params.D_norm is not None else {
        "type": "normal", "mu": params.mu_norm, "sigma": params.sigma_norm
    }
    D_coal = params.D_coal if params.D_coal is not None else {
        "type": "normal_shift", "delta": 0.05
    }

    # ids
    coal_offset = params.N_norm
    N_total = params.N_norm + params.N_coal
    coal_ids_local = np.arange(params.N_coal, dtype=np.int32)

    is_coal_member = np.zeros(N_total, dtype=np.int8)
    is_coal_member[coal_offset:] = 1

    # histogram bin edges
    bin_edges = np.linspace(params.x_min, params.x_max, params.bins + 1, dtype=np.float64)

    # meta and config_id
    p_dict = asdict(params)
    p_dict["D_norm"] = D_norm
    p_dict["D_coal"] = D_coal
    meta_core = {
        "schema": SCHEMA_VERSION,
        "params": p_dict,
        "bin_edges": bin_edges.tolist(),
        "derived": {"N_total": N_total, "coal_offset": coal_offset},
    }
    config_id = _sha256(_stable_dumps(meta_core))[:16]
    meta = {
        "schema": SCHEMA_VERSION,
        "config_id": config_id,
        "git_commit": _git_commit(),
        **meta_core,
        "arrays": {"is_coal_member": is_coal_member.tolist()},
    }

    meta_path = os.path.join(out_dir, "meta.json")
    intervals_path = os.path.join(out_dir, "intervals.csv")
    exposures_path = os.path.join(out_dir, "exposures_by_t.npz")
    participation_path = os.path.join(out_dir, "participation.csv")

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(_stable_dumps(meta) + "\n")

    interval_fields = [
        "schema", "config_id", "t",
        "n_actions", "n_norm_actions", "n_coal_actions",
        "coalition_on", "k_coal_participants",
        "h_bins", "counts_json",
    ]
    part_fields = ["schema", "config_id", "t", "u", "a_ut", "is_coal"]

    rot_ptr = 0
    a_n = int(params.actions_per_participant_norm)
    a_c = int(params.actions_per_participant_coal)

    # Sparse exposures in CSR-by-interval form
    t_ptr = np.zeros(params.T + 1, dtype=np.int64)
    u_chunks = []
    a_chunks = []
    nnz = 0

    # Optional CSV participation writer
    if write_participation_csv:
        f_part = open(participation_path, "w", newline="", encoding="utf-8")
        w_part = csv.DictWriter(f_part, fieldnames=part_fields)
        w_part.writeheader()
    else:
        f_part = None
        w_part = None

    with open(intervals_path, "w", newline="", encoding="utf-8") as f_int:
        w_int = csv.DictWriter(f_int, fieldnames=interval_fields)
        w_int.writeheader()

        for t0 in range(params.T):
            # normals IID participation
            ids_n = np.flatnonzero(rng.random(params.N_norm) < params.p_norm).astype(np.int32)

            # coalition ON/OFF, rotate if ON
            coalition_on = int(rng.random() < params.p_on)
            if coalition_on and params.k_on > 0:
                chosen_local, rot_ptr = _rotate_k(coal_ids_local, params.k_on, rot_ptr)
                ids_c = (chosen_local + coal_offset).astype(np.int32)
            else:
                ids_c = np.array([], dtype=np.int32)

            # sample one signal per action
            x_n = (
                _sample(
                    D_norm, size=len(ids_n) * a_n, rng=rng,
                    mu_norm=params.mu_norm, sigma_norm=params.sigma_norm,
                    x_min=params.x_min, x_max=params.x_max,
                )
                if len(ids_n) else np.empty((0,), dtype=np.float64)
            )
            x_c = (
                _sample(
                    D_coal, size=len(ids_c) * a_c, rng=rng,
                    mu_norm=params.mu_norm, sigma_norm=params.sigma_norm,
                    x_min=params.x_min, x_max=params.x_max,
                )
                if len(ids_c) else np.empty((0,), dtype=np.float64)
            )

            x_all = np.concatenate([x_n, x_c], axis=0) if (x_n.size + x_c.size) else np.empty((0,), dtype=np.float64)

            counts, _ = np.histogram(x_all, bins=bin_edges)
            counts_json = json.dumps(counts.astype(int).tolist(), separators=(",", ":"))

            # write interval row (t is 1-indexed)
            w_int.writerow({
                "schema": SCHEMA_VERSION,
                "config_id": config_id,
                "t": t0 + 1,
                "n_actions": int(x_all.size),
                "n_norm_actions": int(x_n.size),
                "n_coal_actions": int(x_c.size),
                "coalition_on": coalition_on,
                "k_coal_participants": int(len(ids_c)),
                "h_bins": int(params.bins),
                "counts_json": counts_json,
            })

            # --- exposures for this interval t ---
            # Each participating account contributes exposure weight a_ut = actions_per_participant_*
            if ids_n.size + ids_c.size > 0:
                u_t = np.concatenate([ids_n, ids_c], axis=0).astype(np.int32)
                a_t = np.concatenate(
                    [
                        np.full(ids_n.shape, float(a_n), dtype=np.float32),
                        np.full(ids_c.shape, float(a_c), dtype=np.float32),
                    ],
                    axis=0,
                ).astype(np.float32)

                u_chunks.append(u_t)
                a_chunks.append(a_t)
                nnz += int(u_t.size)

                if w_part is not None:
                    for u in ids_n:
                        w_part.writerow({
                            "schema": SCHEMA_VERSION,
                            "config_id": config_id,
                            "t": t0 + 1,
                            "u": int(u),
                            "a_ut": float(a_n),
                            "is_coal": 0,
                        })
                    for u in ids_c:
                        w_part.writerow({
                            "schema": SCHEMA_VERSION,
                            "config_id": config_id,
                            "t": t0 + 1,
                            "u": int(u),
                            "a_ut": float(a_c),
                            "is_coal": 1,
                        })

            t_ptr[t0 + 1] = nnz

    if f_part is not None:
        f_part.close()

    # Write compact exposures file
    if nnz > 0:
        u_ids = np.concatenate(u_chunks, axis=0).astype(np.int32)
        a_ut = np.concatenate(a_chunks, axis=0).astype(np.float32)
    else:
        u_ids = np.zeros((0,), dtype=np.int32)
        a_ut = np.zeros((0,), dtype=np.float32)

    np.savez_compressed(
        exposures_path,
        schema=np.array([SCHEMA_VERSION]),
        config_id=np.array([config_id]),
        T=np.array([params.T], dtype=np.int64),
        t_ptr=t_ptr,
        u_ids=u_ids,
        a_ut=a_ut,
    )

    paths = {"meta": meta_path, "intervals": intervals_path, "exposures_npz": exposures_path}
    if write_participation_csv:
        paths["participation_csv"] = participation_path
    return paths


# -----------------------------
# CLI
# -----------------------------
def _json_arg(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError("Must be a JSON object/dict.")
    return obj


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Rotation-model generator (NPZ exposures; optional CSV participation)")
    ap.add_argument("--out", required=True, help="Output directory")

    ap.add_argument("--N_norm", type=int, default=RotationParams.N_norm)
    ap.add_argument("--N_coal", type=int, default=RotationParams.N_coal)
    ap.add_argument("--T", type=int, default=RotationParams.T)

    ap.add_argument("--p_norm", type=float, default=RotationParams.p_norm)
    ap.add_argument("--p_on", type=float, default=RotationParams.p_on)
    ap.add_argument("--k_on", type=int, default=RotationParams.k_on)

    ap.add_argument("--bins", type=int, default=RotationParams.bins)
    ap.add_argument("--x_min", type=float, default=RotationParams.x_min)
    ap.add_argument("--x_max", type=float, default=RotationParams.x_max)

    ap.add_argument("--mu_norm", type=float, default=RotationParams.mu_norm)
    ap.add_argument("--sigma_norm", type=float, default=RotationParams.sigma_norm)

    ap.add_argument("--actions_per_participant_norm", type=int, default=RotationParams.actions_per_participant_norm)
    ap.add_argument("--actions_per_participant_coal", type=int, default=RotationParams.actions_per_participant_coal)

    ap.add_argument("--seed", type=int, default=RotationParams.seed)

    ap.add_argument("--D_norm", type=_json_arg, default=None, help="JSON dict for normal distribution spec")
    ap.add_argument("--D_coal", type=_json_arg, default=None, help="JSON dict for coalition distribution spec")

    ap.add_argument(
        "--write_participation_csv",
        action="store_true",
        default=False,
        help="Also write participation.csv (WARNING: can be huge). Default: off.",
    )

    args = ap.parse_args(argv)

    params = RotationParams(
        N_norm=args.N_norm,
        N_coal=args.N_coal,
        T=args.T,
        p_norm=args.p_norm,
        p_on=args.p_on,
        k_on=args.k_on,
        bins=args.bins,
        x_min=args.x_min,
        x_max=args.x_max,
        mu_norm=args.mu_norm,
        sigma_norm=args.sigma_norm,
        D_norm=args.D_norm,
        D_coal=args.D_coal,
        actions_per_participant_norm=args.actions_per_participant_norm,
        actions_per_participant_coal=args.actions_per_participant_coal,
        seed=args.seed,
    )

    paths = generate_rotation_dataset(params, out_dir=args.out, write_participation_csv=bool(args.write_participation_csv))
    print("[ok] wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
