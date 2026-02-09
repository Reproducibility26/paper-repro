#!/usr/bin/env python3
"""
Script #4: Metrics & evaluation (AUC, ROC/PR curves, top-k metrics).

Inputs (from run_dir):
  - meta.json                    (from Script #1, must contain arrays.is_coal_member)
  - attribution_scores.csv       (from Script #3, columns: u, S__<method>...)

Outputs (to out_dir, default run_dir):
  - metrics.json
  - roc_curves.csv
  - pr_curves.csv
  - (optional) ranked_lists.csv

Main metrics per method:
  - ROC AUC
  - PR AUC
  - Precision@k and Recall@k for selected k
  - TPR@FPR (e.g., 1%, 5%)

This script assumes:
  - Account ids u are integers matching the generator's indexing (0..N_total-1).
  - meta.json contains arrays.is_coal_member: list[int] with 0/1 labels.

No personal identifying information is written; outputs may include synthetic integer account IDs u.

"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = "metrics_v1"


# ============================================================
# Params
# ============================================================
@dataclass(frozen=True)
class MetricsParams:
    attribution_filename: str = "attribution_scores.csv"
    meta_filename: str = "meta.json"

    out_dir: Optional[str] = None

    # which methods to evaluate (method name = suffix after "S__")
    # if None, auto-detect from attribution_scores.csv
    methods: Optional[List[str]] = None

    # top-k metrics
    k_list: Tuple[int, ...] = (10, 25, 50, 100, 200)

    # TPR at FPR targets
    fpr_targets: Tuple[float, ...] = (0.01, 0.05, 0.10)

    # output controls
    write_curves: bool = True
    write_ranked_lists: bool = False
    ranked_top_n: int = 200  # if write_ranked_lists

    # handle ties:
    # if True, stable tie-handling via sort by (-score, u)
    stable_ties: bool = True


# ============================================================
# IO
# ============================================================
def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _read_attribution_scores(path: str, methods: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      u: (N,) int
      S: (N,M) float
      method_names: list[str] length M
    """
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("attribution_scores.csv missing header")
        if "u" not in r.fieldnames:
            raise ValueError("attribution_scores.csv missing 'u' column")

        # auto-detect methods
        all_methods = [c[len("S__"):] for c in r.fieldnames if c.startswith("S__")]
        if not all_methods:
            raise ValueError("No S__<method> columns found in attribution_scores.csv")

        if methods is None:
            method_names = all_methods
        else:
            missing = [m for m in methods if m not in all_methods]
            if missing:
                raise ValueError(f"Requested methods not found in attribution_scores.csv: {missing}")
            method_names = list(methods)

        rows_u: List[int] = []
        rows_S: List[List[float]] = []
        for row in r:
            u = int(row["u"])
            vals = []
            for m in method_names:
                v = row.get("S__" + m, "")
                vals.append(float(v) if v != "" else 0.0)
            rows_u.append(u)
            rows_S.append(vals)

    u_arr = np.asarray(rows_u, dtype=np.int64)
    S_arr = np.asarray(rows_S, dtype=np.float64)
    return u_arr, S_arr, method_names


# ============================================================
# Metrics core
# ============================================================
def _sort_by_score(scores: np.ndarray, u: np.ndarray, stable_ties: bool) -> np.ndarray:
    """
    Returns indices that sort scores descending.
    If stable_ties: secondary key is u ascending to make deterministic.
    """
    if stable_ties:
        # lexsort sorts by last key first; we want (-scores, u)
        return np.lexsort((u, -scores))
    return np.argsort(-scores, kind="mergesort")


def roc_curve(scores: np.ndarray, y: np.ndarray, stable_ties: bool = True, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve points for all thresholds induced by sorted scores.
    Returns (fpr, tpr, thresholds).

    y: 0/1 labels
    """
    if u is None:
        u = np.arange(len(scores), dtype=np.int64)

    order = _sort_by_score(scores, u, stable_ties)
    y_sorted = y[order]
    s_sorted = scores[order]

    P = float(np.sum(y_sorted == 1))
    N = float(np.sum(y_sorted == 0))
    if P == 0 or N == 0:
        # degenerate
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    # cumulative TP/FP as threshold moves down
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)

    # take points at distinct score values
    distinct = np.where(np.diff(s_sorted) != 0)[0]
    idx = np.r_[distinct, len(s_sorted) - 1]

    tpr = tp[idx] / P
    fpr = fp[idx] / N
    thresholds = s_sorted[idx]

    # add (0,0) start
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[thresholds[0] + 1.0, thresholds]  # sentinel larger than max score

    return fpr, tpr, thresholds


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    """Area under curve using trapezoidal rule; expects x sorted ascending."""
    return float(np.trapz(y, x))


def pr_curve(scores: np.ndarray, y: np.ndarray, stable_ties: bool = True, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precision-Recall curve.
    Returns (recall, precision, thresholds).
    """
    if u is None:
        u = np.arange(len(scores), dtype=np.int64)

    order = _sort_by_score(scores, u, stable_ties)
    y_sorted = y[order]
    s_sorted = scores[order]

    P = float(np.sum(y_sorted == 1))
    if P == 0:
        # degenerate
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([np.inf, -np.inf])

    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)

    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / P

    distinct = np.where(np.diff(s_sorted) != 0)[0]
    idx = np.r_[distinct, len(s_sorted) - 1]

    precision = precision[idx]
    recall = recall[idx]
    thresholds = s_sorted[idx]

    # standard PR curve starts at recall=0 with precision=1
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    thresholds = np.r_[thresholds[0] + 1.0, thresholds]

    return recall, precision, thresholds


def precision_recall_at_k(scores: np.ndarray, y: np.ndarray, k: int, stable_ties: bool = True, u: Optional[np.ndarray] = None) -> Tuple[float, float]:
    if u is None:
        u = np.arange(len(scores), dtype=np.int64)
    order = _sort_by_score(scores, u, stable_ties)
    top = order[: min(k, len(order))]
    tp = float(np.sum(y[top] == 1))
    prec = tp / float(len(top)) if len(top) > 0 else 0.0
    total_pos = float(np.sum(y == 1))
    rec = tp / total_pos if total_pos > 0 else 0.0
    return prec, rec


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """
    Given ROC points (fpr,tpr) with fpr increasing, return max tpr with fpr <= target.
    """
    mask = fpr <= target_fpr + 1e-12
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


# ============================================================
# Main runner
# ============================================================
def run_metrics(run_dir: str, params: MetricsParams) -> Dict[str, str]:
    meta_path = os.path.join(run_dir, params.meta_filename)
    attr_path = os.path.join(run_dir, params.attribution_filename)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")
    if not os.path.exists(attr_path):
        raise FileNotFoundError(f"Missing attribution_scores.csv: {attr_path}")

    meta = _read_json(meta_path)

    # labels
    # expected meta["arrays"]["is_coal_member"] = list[int] length N_total
    try:
        y_full = np.asarray(meta["arrays"]["is_coal_member"], dtype=np.int8)
    except Exception as e:
        raise ValueError("meta.json missing arrays.is_coal_member") from e

    # scores
    u, S, method_names = _read_attribution_scores(attr_path, params.methods)

    # align labels to rows in attribution_scores.csv
    if np.max(u) >= len(y_full) or np.min(u) < 0:
        raise ValueError("Some u ids in attribution_scores.csv are outside meta arrays.is_coal_member range.")
    y = y_full[u].astype(np.int8)

    out_dir = params.out_dir or run_dir
    _ensure_dir(out_dir)

    metrics_out: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "run_dir": os.path.abspath(run_dir),
        "dataset_config_id": meta.get("config_id", ""),
        "n_accounts_scored": int(len(u)),
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "methods": {},
        "k_list": list(params.k_list),
        "fpr_targets": list(params.fpr_targets),
    }

    roc_rows: List[Dict[str, Any]] = []
    pr_rows: List[Dict[str, Any]] = []
    ranked_rows: List[Dict[str, Any]] = []

    for j, m in enumerate(method_names):
        scores = S[:, j].astype(np.float64)

        # ROC/AUC
        fpr, tpr, thr = roc_curve(scores, y, stable_ties=params.stable_ties, u=u)
        roc_auc = auc_trapz(fpr, tpr)

        # PR/AUC
        recall, precision, thr_pr = pr_curve(scores, y, stable_ties=params.stable_ties, u=u)
        pr_auc = auc_trapz(recall, precision)

        # top-k metrics
        topk = {}
        for k in params.k_list:
            prec_k, rec_k = precision_recall_at_k(scores, y, int(k), stable_ties=params.stable_ties, u=u)
            topk[str(int(k))] = {"precision": prec_k, "recall": rec_k}

        # TPR at FPR targets
        tpr_fpr = {}
        for a in params.fpr_targets:
            tpr_fpr[str(a)] = tpr_at_fpr(fpr, tpr, float(a))

        metrics_out["methods"][m] = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "tpr_at_fpr": tpr_fpr,
            "topk": topk,
            "score_summary": {
                "min": float(np.min(scores)) if len(scores) else 0.0,
                "max": float(np.max(scores)) if len(scores) else 0.0,
                "mean": float(np.mean(scores)) if len(scores) else 0.0,
                "std": float(np.std(scores)) if len(scores) else 0.0,
            },
        }

        if params.write_curves:
            for i in range(len(fpr)):
                roc_rows.append({
                    "schema": SCHEMA_VERSION,
                    "method": m,
                    "idx": i,
                    "fpr": float(fpr[i]),
                    "tpr": float(tpr[i]),
                    "threshold": float(thr[i]),
                })
            for i in range(len(recall)):
                pr_rows.append({
                    "schema": SCHEMA_VERSION,
                    "method": m,
                    "idx": i,
                    "recall": float(recall[i]),
                    "precision": float(precision[i]),
                    "threshold": float(thr_pr[i]),
                })

        if params.write_ranked_lists:
            order = _sort_by_score(scores, u=u, stable_ties=params.stable_ties)
            top = order[: min(params.ranked_top_n, len(order))]
            for rank, idx in enumerate(top, start=1):
                ranked_rows.append({
                    "schema": SCHEMA_VERSION,
                    "method": m,
                    "rank": rank,
                    "u": int(u[idx]),
                    "score": float(scores[idx]),
                    "label_is_coal": int(y[idx]),
                })

    # Write outputs
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics_out, indent=2) + "\n")

    outputs = {"metrics_json": metrics_path}

    if params.write_curves:
        roc_path = os.path.join(out_dir, "roc_curves.csv")
        with open(roc_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["schema", "method", "idx", "fpr", "tpr", "threshold"])
            w.writeheader()
            for r0 in roc_rows:
                w.writerow(r0)
        outputs["roc_curves_csv"] = roc_path

        pr_path = os.path.join(out_dir, "pr_curves.csv")
        with open(pr_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["schema", "method", "idx", "recall", "precision", "threshold"])
            w.writeheader()
            for r0 in pr_rows:
                w.writerow(r0)
        outputs["pr_curves_csv"] = pr_path

    if params.write_ranked_lists:
        ranked_path = os.path.join(out_dir, "ranked_lists.csv")
        with open(ranked_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["schema", "method", "rank", "u", "score", "label_is_coal"])
            w.writeheader()
            for r0 in ranked_rows:
                w.writerow(r0)
        outputs["ranked_lists_csv"] = ranked_path

    return outputs


# ============================================================
# CLI
# ============================================================
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Script #4: metrics/evaluation for attribution scores")
    ap.add_argument("--run_dir", required=True, help="Directory containing meta.json + attribution_scores.csv")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: run_dir)")

    ap.add_argument("--methods", default=None,
                    help='Optional JSON list of methods, e.g. \'["ours_w1","bl_js","bl_chi2"]\'')

    ap.add_argument("--k_list", default=None,
                    help='Optional JSON list of k values, e.g. "[10,25,50,100]"')
    ap.add_argument("--fpr_targets", default=None,
                    help='Optional JSON list of FPR targets, e.g. "[0.01,0.05,0.1]"')

    ap.add_argument("--no_curves", action="store_true", default=False)
    ap.add_argument("--ranked_lists", action="store_true", default=False)
    ap.add_argument("--ranked_top_n", type=int, default=200)

    ap.add_argument("--no_stable_ties", action="store_true", default=False)

    args = ap.parse_args(argv)

    methods = json.loads(args.methods) if args.methods else None
    k_list = tuple(int(x) for x in json.loads(args.k_list)) if args.k_list else MetricsParams.k_list
    fpr_targets = tuple(float(x) for x in json.loads(args.fpr_targets)) if args.fpr_targets else MetricsParams.fpr_targets

    params = MetricsParams(
        out_dir=args.out_dir,
        methods=[str(m) for m in methods] if methods is not None else None,
        k_list=k_list,
        fpr_targets=fpr_targets,
        write_curves=(not args.no_curves),
        write_ranked_lists=bool(args.ranked_lists),
        ranked_top_n=int(args.ranked_top_n),
        stable_ties=(not args.no_stable_ties),
    )

    outputs = run_metrics(args.run_dir, params)
    print("[ok] wrote:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
