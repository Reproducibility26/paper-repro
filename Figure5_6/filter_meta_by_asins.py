import argparse, csv

def load_asins(path: str) -> set[str]:
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            a = line.strip()
            if a:
                s.add(a)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-csv", required=True, help="usable_meta_books.csv")
    ap.add_argument("--asin-list", required=True, help="eligible_asins.txt")
    ap.add_argument("--out-csv", required=True, help="eligible_meta_books.csv")
    ap.add_argument("--progress-every", type=int, default=200000)
    args = ap.parse_args()

    asins = load_asins(args.asin_list)
    print(f"[INFO] loaded eligible asins: {len(asins):,}")

    kept = 0
    scanned = 0

    with open(args.meta_csv, "r", encoding="utf-8", newline="") as fin, \
         open(args.out_csv, "w", encoding="utf-8", newline="") as fout:
        r = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=r.fieldnames)
        w.writeheader()

        for row in r:
            scanned += 1
            a = row.get("asin", "")
            if a in asins:
                w.writerow(row)
                kept += 1
            if scanned % args.progress_every == 0:
                print(f"[PROGRESS] scanned={scanned:,} kept={kept:,}")

    print(f"[DONE] scanned={scanned:,} kept={kept:,}")
    print(f"[OUTPUT] {args.out_csv}")

if __name__ == "__main__":
    main()
