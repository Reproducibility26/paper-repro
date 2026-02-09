import argparse
import csv
import gzip
import json
import re
import time
from typing import Any, Dict, Iterable, Optional, Set, Tuple


def load_asin_set(path: str) -> Set[str]:
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            a = line.strip()
            if a:
                s.add(a)
    return s


def safe_str(x: Any, maxlen: int = 5000) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, dict)):
        out = json.dumps(x, ensure_ascii=False, separators=(",", ":"))
    else:
        out = str(x)
    if len(out) > maxlen:
        out = out[:maxlen] + "…"
    return out


def parse_price_to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"(\d+(?:\.\d+)?)", x.replace(",", ""))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def extract_rank(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        m = re.search(r"(\d[\d,]*)", x)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                return None
    return None


def categories_contain_books(categories: Any) -> bool:
    if not categories:
        return False
    if isinstance(categories, list):
        for path in categories:
            if isinstance(path, list):
                if any(isinstance(c, str) and c.lower() == "books" for c in path):
                    return True
            elif isinstance(path, str) and path.lower() == "books":
                return True
    return False


def is_usable_meta(
    obj: Dict[str, Any],
    require_books_category: bool,
    require_nonempty_title: bool,
    require_nonempty_description: bool,
    require_positive_price: bool,
) -> Tuple[bool, str]:

    asin = obj.get("asin")
    if not asin:
        return False, "missing_asin"

    if require_nonempty_title:
        title = obj.get("title")
        if not isinstance(title, str) or not title.strip():
            return False, "missing_title"

    if require_books_category:
        if not categories_contain_books(obj.get("categories")):
            return False, "not_books_category"

    if require_nonempty_description:
        desc = obj.get("description")
        if isinstance(desc, list):
            if not any(isinstance(d, str) and d.strip() for d in desc):
                return False, "missing_description"
        else:
            if not isinstance(desc, str) or not desc.strip():
                return False, "missing_description"

    if require_positive_price:
        p = parse_price_to_float(obj.get("price"))
        if p is None or p <= 0:
            return False, "missing_or_nonpositive_price"

    return True, ""


def iter_gz_json_lines(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--asin-list", default=None)
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--require-books-category", action="store_true", default=True)
    ap.add_argument("--no-require-books-category", action="store_false", dest="require_books_category")

    ap.add_argument("--require-nonempty-title", action="store_true", default=True)
    ap.add_argument("--no-require-nonempty-title", action="store_false", dest="require_nonempty_title")

    ap.add_argument("--require-nonempty-description", action="store_true", default=False)
    ap.add_argument("--require-positive-price", action="store_true", default=False)

    args = ap.parse_args()

    asin_keep = None
    if args.asin_list:
        asin_keep = load_asin_set(args.asin_list)
        print(f"[INFO] Loaded {len(asin_keep):,} ASINs from {args.asin_list}")

    print("[INFO] Starting scan")
    print(f"       input : {args.input}")
    print(f"       output: {args.output}")
    print(f"       limit : {args.limit if args.limit else 'none'}")

    fieldnames = [
        "asin", "title", "brand", "price", "rank",
        "categories", "description", "imUrl", "related"
    ]

    scanned = kept = dropped = 0
    drop_reasons: Dict[str, int] = {}

    t0 = time.time()
    last_report = t0

    with open(args.output, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for obj in iter_gz_json_lines(args.input):
            scanned += 1
            if args.limit and scanned >= args.limit:
                break

            asin = obj.get("asin")
            if asin_keep is not None and asin not in asin_keep:
                dropped += 1
                drop_reasons["not_in_asin_list"] = drop_reasons.get("not_in_asin_list", 0) + 1
                continue

            ok, reason = is_usable_meta(
                obj,
                args.require_books_category,
                args.require_nonempty_title,
                args.require_nonempty_description,
                args.require_positive_price,
            )
            if not ok:
                dropped += 1
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                continue

            writer.writerow({
                "asin": safe_str(obj.get("asin"), 64),
                "title": safe_str(obj.get("title"), 500),
                "brand": safe_str(obj.get("brand"), 200),
                "price": parse_price_to_float(obj.get("price")),
                "rank": extract_rank(obj.get("rank")),
                "categories": safe_str(obj.get("categories"), 2000),
                "description": safe_str(obj.get("description"), 4000),
                "imUrl": safe_str(obj.get("imUrl"), 500),
                "related": safe_str(obj.get("related"), 2000),
            })
            kept += 1

            # progress report
            if scanned % 100_000 == 0:
                now = time.time()
                rate = scanned / max(now - t0, 1e-6)
                print(
                    f"[PROGRESS] scanned={scanned:,} "
                    f"kept={kept:,} dropped={dropped:,} "
                    f"({rate:,.0f} lines/sec)"
                )
                last_report = now

    elapsed = time.time() - t0
    print("\n[DONE]")
    print(f"  scanned : {scanned:,}")
    print(f"  kept    : {kept:,}")
    print(f"  dropped : {dropped:,}")
    print(f"  elapsed : {elapsed/60:.1f} min")

    if drop_reasons:
        print("\n[DROP REASONS]")
        for k, v in sorted(drop_reasons.items(), key=lambda x: -x[1]):
            print(f"  {k:30s} {v:,}")


if __name__ == "__main__":
    main()
