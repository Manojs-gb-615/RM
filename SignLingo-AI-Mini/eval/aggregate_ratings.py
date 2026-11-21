"""
Aggregate ASL-signer ratings into a summary table.

STATUS: PENDING HUMAN EVALUATION until real ratings are entered.

Usage:
    python eval/aggregate_ratings.py --input eval/rating_form.csv --output eval/summary.csv
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

METRICS = [
    ("sign_accuracy_1_5", "sign_accuracy"),
    ("naturalness_1_5", "naturalness"),
    ("sync_quality_1_5", "sync_quality"),
]
PENDING = "PENDING HUMAN EVALUATION"


def _valid_score(value: str) -> bool:
    try:
        score = int(str(value).strip())
    except (TypeError, ValueError):
        return False
    return 1 <= score <= 5


def load_ratings(path: str) -> Tuple[Dict[str, Dict[str, List[int]]], int, int]:
    """Return {mode: {metric: [scores]}}, rows_used, rows_skipped."""
    groups: Dict[str, Dict[str, List[int]]] = {}
    used = skipped = 0
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()
            mode = (row.get("mode") or "").strip()
            if not sample_id or sample_id.upper().startswith("EXAMPLE") or not mode:
                skipped += 1
                continue
            if not all(_valid_score(row.get(key, "")) for key, _ in METRICS):
                skipped += 1
                continue
            bucket = groups.setdefault(mode, {label: [] for _, label in METRICS})
            for key, label in METRICS:
                bucket[label].append(int(str(row[key]).strip()))
            used += 1
    return groups, used, skipped


def summarize(groups: Dict[str, Dict[str, List[int]]]) -> List[Dict]:
    rows = []
    for mode, metrics in sorted(groups.items()):
        for _, label in METRICS:
            values = metrics.get(label, [])
            if not values:
                continue
            rows.append({
                "mode": mode,
                "metric": label,
                "mean": round(statistics.mean(values), 3),
                "std": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
                "n": len(values),
            })
    return rows


def aggregate(input_path: str, output_path: str) -> Dict:
    groups, used, skipped = load_ratings(input_path)
    rows = summarize(groups)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "metric", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(rows)

    pending = used == 0
    if pending:
        print(f"{PENDING} — no valid rating rows in {input_path}. "
              f"Fill the form with ASL signers first.")
    else:
        print(f"Aggregated {used} rating rows ({skipped} skipped).")
        for row in rows:
            print(f"  {row['mode']:16s} {row['metric']:14s} "
                  f"mean={row['mean']:.2f} sd={row['std']:.2f} n={row['n']}")
    print(f"[ratings] Wrote {output_path}")
    return {"pending": pending, "rows": rows, "used": used, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate ASL signer ratings")
    parser.add_argument("--input", default="eval/rating_form.csv")
    parser.add_argument("--output", default="eval/summary.csv")
    args = parser.parse_args()
    try:
        aggregate(args.input, args.output)
    except FileNotFoundError as exc:
        print(f"[ratings] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
