#!/usr/bin/env python3
"""
scan_reviews_books_eligibility.py

Streaming scan of Amazon Books reviews JSON.GZ (SNAP format) to find "eligible" ASINs
for stable aggregate-time experiments.

Outputs:
  1) eligible_asins.txt   (one ASIN per line)
  2) asin_stats.csv       (summary stats per ASIN; useful for debugging / selection)

Eligibility rule (tunable):
  - total_reviews >= MIN_TOTAL_REVIEWS
  - valid_weeks >= MIN_VALID_WEEKS
where "valid week" means weekly count n_t >= N_MIN.

Windows cmd examples:
  py scan_reviews_books_eligibility.py --input reviews_Books_5.json.gz --outdir .

  py scan_reviews_books_eligibility.py --input reviews_Books_5.json.gz ^
     --outdir . --time-bin W --n-min 30 --min-total 500 --min-valid-weeks 52

Notes:
- This script ONLY counts reviews and weekly volumes per ASIN (fast + memory-safe).
- It does NOT compute histograms or W1; that's the next step after filtering.
"""

import argparse
import csv
import gzip
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Tuple


def week_key_from_unix(ts: Any) -> int:
    """
    Convert unixReviewTime to an integer week key.

    We use ISO week anchored by Monday via datetime.isocalendar, compressed into:
      key = (iso_year * 100) + iso_week

    This avoids timezone issues and is stable for aggregation.
    """
    t = int(ts)
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
    iso_year, iso_week, _ = dt.isocalendar()
    return iso_year * 100 + iso_week


def iter_reviews(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError:
                continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to reviews_Books_*.json.gz")
    ap.add_argument("--outdir", required=True, help="Output directory (e.g., . or C:\\path\\to\\dir)")
    ap.add_argument("--time-bin", default="W", choices=["W"], help="Time binning (currently weekly only)")
    ap.add_argument("--n-min", type=int, default=30, help="A week is 'valid' if count >= n-min")
    ap.add_argument("--min-total", type=int, default=500, help="Min total reviews per ASIN")
    ap.add_argument("--min-valid-weeks", type=int, default=52, help="Min #valid weeks per ASIN")
    ap.add_argument("--progress-every", type=int, default=2_000_000, help="Print progress every N lines")
    ap.add_argument("--limit", type=int, default=0, help="Optional max lines to scan (0 = no limit)")
    args = ap.parse_args()

    in_path = args.input
    outdir = args.outdir.rstrip("\\/")

    eligible_txt = f"{outdir}\\eligible_asins.txt"
    stats_csv = f"{outdir}\\asin_stats.csv"

    N_MIN = args.n_min
    MIN_TOTAL = args.min_total
    MIN_VALID_WEEKS = args.min_valid_weeks

    print("[INFO] Starting review scan")
    print(f"       input          : {in_path}")
    print(f"       outdir         : {outdir}")
    print(f"       weekly valid if: n_t >= {N_MIN}")
    print(f"       eligible if    : total >= {MIN_TOTAL} AND valid_weeks >= {MIN_VALID_WEEKS}")
    print(f"       limit          : {args.limit if args.limit else 'none'}")
    print(f"       progress every : {args.progress_every:,} lines")

    # Counters:
    # total_reviews[asin] = total number of reviews
    total_reviews: Dict[str, int] = defaultdict(int)

    # week_counts[(asin, weekkey)] = count in that week
    # Using tuple key keeps the structure simple and still memory-safe enough after one pass.
    week_counts: Dict[Tuple[str, int], int] = defaultdict(int)

    scanned = 0
    bad_rows = 0
    t0 = time.time()

    for obj in iter_reviews(in_path):
        scanned += 1
        if args.limit and scanned >= args.limit:
            break

        asin = obj.get("asin")
        ts = obj.get("unixReviewTime")
        if not asin or ts is None:
            bad_rows += 1
            continue

        # update totals
        total_reviews[asin] += 1

        # weekly bin
        wk = week_key_from_unix(ts)
        week_counts[(asin, wk)] += 1

        if scanned % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = scanned / max(elapsed, 1e-9)
            print(f"[PROGRESS] scanned={scanned:,} asins_seen={len(total_reviews):,} "
                  f"week_cells={len(week_counts):,} bad={bad_rows:,} ({rate:,.0f} lines/sec)")

    # Compute per-ASIN weekly stats
    # We need: num_weeks, valid_weeks, max_weekly, sum_weekly (=total_reviews sanity)
    num_weeks: Dict[str, int] = defaultdict(int)
    valid_weeks: Dict[str, int] = defaultdict(int)
    max_weekly: Dict[str, int] = defaultdict(int)

    for (asin, _wk), c in week_counts.items():
        num_weeks[asin] += 1
        if c >= N_MIN:
            valid_weeks[asin] += 1
        if c > max_weekly[asin]:
            max_weekly[asin] = c

    # Write stats + eligible list
    kept = 0
    with open(stats_csv, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["asin", "total_reviews", "num_weeks", "valid_weeks", "max_weekly"])

        # Eligible ASINs written to txt
        with open(eligible_txt, "w", encoding="utf-8") as ftxt:
            for asin, tot in total_reviews.items():
                nw = num_weeks.get(asin, 0)
                vw = valid_weeks.get(asin, 0)
                mx = max_weekly.get(asin, 0)

                w.writerow([asin, tot, nw, vw, mx])

                if tot >= MIN_TOTAL and vw >= MIN_VALID_WEEKS:
                    ftxt.write(asin + "\n")
                    kept += 1

    elapsed = time.time() - t0
    print("\n[DONE]")
    print(f"  scanned lines : {scanned:,}")
    print(f"  asins seen    : {len(total_reviews):,}")
    print(f"  week cells    : {len(week_counts):,}")
    print(f"  bad rows      : {bad_rows:,}")
    print(f"  eligible asins: {kept:,}")
    print(f"  elapsed       : {elapsed/60:.1f} min")
    print(f"\n[OUTPUT]")
    print(f"  {eligible_txt}")
    print(f"  {stats_csv}")


if __name__ == "__main__":
    main()
