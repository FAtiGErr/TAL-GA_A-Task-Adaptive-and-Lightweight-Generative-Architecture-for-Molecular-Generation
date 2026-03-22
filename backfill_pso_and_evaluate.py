import os
import re
import json
import glob
import argparse
from zipfile import BadZipFile

import numpy as np

import pandas as pd

from config import PSO_RESULTS_DIR, set_working_directory


set_working_directory()


UNI_TASKS = [
    ("LOGP", "logp", "1.0"), ("LOGP", "logp", "2.0"), ("LOGP", "logp", "3.0"), ("LOGP", "logp", "4.0"),
    ("TPSA", "tpsa", "20.0"), ("TPSA", "tpsa", "40.0"), ("TPSA", "tpsa", "60.0"), ("TPSA", "tpsa", "80.0"),
]
TRI_TASKS = [
    ("LOGP-TPSA", "1.0-20.0"), ("LOGP-TPSA", "2.0-40.0"), ("LOGP-TPSA", "3.0-60.0"), ("LOGP-TPSA", "4.0-80.0"),
]


def _is_seed_file_healthy(npz_file):
    if not os.path.exists(npz_file):
        return False, "missing"

    try:
        with np.load(npz_file, allow_pickle=False) as data:
            if "HistX" not in data or "HistY" not in data:
                return False, "missing_HistX_or_HistY"
            _ = data["HistX"][-1]
            _ = data["HistY"][-1]
        return True, "ok"
    except (BadZipFile, EOFError, OSError, ValueError, KeyError, IndexError) as exc:
        return False, f"corrupt:{type(exc).__name__}"


def _existing_seeds(objective, prop, target):
    pattern = os.path.join(PSO_RESULTS_DIR, objective, prop, f"{target}-Seed*.npz")
    rgx = re.compile(r"-Seed(\d+)\.npz$")
    found = set()
    for fp in glob.glob(pattern):
        match = rgx.search(fp)
        if match:
            found.add(int(match.group(1)))
    return found


def _corrupted_seeds(objective, prop, target, expected):
    corrupt = []
    reasons = {}
    for s in expected:
        npz_file = os.path.join(PSO_RESULTS_DIR, objective, prop, f"{target}-Seed{s}.npz")
        ok, reason = _is_seed_file_healthy(npz_file)
        if not ok and reason != "missing":
            corrupt.append(s)
            reasons[s] = reason
    return corrupt, reasons


def collect_missing(rounds=5, chunk_size=100, include_uni=True, include_multi=True):
    report = []

    if include_uni:
        for r in range(1, rounds + 1):
            objective = f"UNI-OBJECTIVE-R{r}"
            expected = set(range((r - 1) * chunk_size, r * chunk_size))
            for prop_upper, prop_cli, target in UNI_TASKS:
                existing = _existing_seeds(objective, prop_upper, target)
                missing = sorted(expected - existing)
                corrupt, corrupt_reason = _corrupted_seeds(objective, prop_upper, target, sorted(existing))
                report.append({
                    "family": "uni",
                    "objective": objective,
                    "prop_upper": prop_upper,
                    "prop_cli": prop_cli,
                    "target": target,
                    "expected": len(expected),
                    "existing": len(existing) - len(corrupt),
                    "missing": len(missing),
                    "corrupted": len(corrupt),
                    "missing_seeds": missing,
                    "corrupted_seeds": corrupt,
                    "corrupted_reason": corrupt_reason,
                })

    if include_multi:
        for r in range(1, rounds + 1):
            objective = f"MULTI-OBJECTIVE-R{r}"
            expected = set(range((r - 1) * chunk_size, r * chunk_size))
            for prop, target in TRI_TASKS:
                existing = _existing_seeds(objective, prop, target)
                missing = sorted(expected - existing)
                corrupt, corrupt_reason = _corrupted_seeds(objective, prop, target, sorted(existing))
                report.append({
                    "family": "multi",
                    "objective": objective,
                    "prop_upper": prop,
                    "prop_cli": "",
                    "target": target,
                    "expected": len(expected),
                    "existing": len(existing) - len(corrupt),
                    "missing": len(missing),
                    "corrupted": len(corrupt),
                    "missing_seeds": missing,
                    "corrupted_seeds": corrupt,
                    "corrupted_reason": corrupt_reason,
                })

    return report


def write_missing_report(report):
    out_dir = os.path.join(PSO_RESULTS_DIR, "reports")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "missing_seed_report.json")
    with open(json_path, "w", encoding="utf-8") as writer:
        json.dump(report, writer, ensure_ascii=True, indent=2)

    rows = []
    for row in report:
        rows.append({
            "family": row["family"],
            "objective": row["objective"],
            "prop": row["prop_upper"],
            "target": row["target"],
            "expected": row["expected"],
            "existing": row["existing"],
            "missing": row["missing"],
            "corrupted": row.get("corrupted", 0),
            "missing_seeds": " ".join(map(str, row["missing_seeds"])),
            "corrupted_seeds": " ".join(map(str, row.get("corrupted_seeds", []))),
        })
    csv_path = os.path.join(out_dir, "missing_seed_report.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return json_path, csv_path


def rerun_missing(report):
    uni_missing = [r for r in report if r["family"] == "uni" and (r["missing_seeds"] or r.get("corrupted_seeds"))]
    multi_missing = [r for r in report if r["family"] == "multi" and (r["missing_seeds"] or r.get("corrupted_seeds"))]

    if uni_missing:
        from OptUni import molopt as uni_molopt

        for row in uni_missing:
            seeds = sorted(set(row["missing_seeds"] + row.get("corrupted_seeds", [])))
            for seed in seeds:
                uni_molopt(seed=seed,
                           targets=row["target"],
                           prop=row["prop_cli"],
                           objective=row["objective"],
                           seed_end=seed + 1)

    if multi_missing:
        from OptTri import molopt as tri_molopt

        for row in multi_missing:
            seeds = sorted(set(row["missing_seeds"] + row.get("corrupted_seeds", [])))
            for seed in seeds:
                tri_molopt(seed=seed,
                           targets=row["target"],
                           objective=row["objective"],
                           seed_end=seed + 1)


def evaluate(rounds=5, chunk_size=100, include_uni=True, include_multi=True, force=False):
    from model_stats import run_chunked_evaluation

    return run_chunked_evaluation(rounds=rounds,
                                  chunk_size=chunk_size,
                                  include_uni=include_uni,
                                  include_multi=include_multi,
                                  force=force)


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill missing PSO seed files and run chunked evaluation.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of round folders: R1..Rn")
    parser.add_argument("--chunk-size", type=int, default=100, help="Seeds per round")
    parser.add_argument("--skip-uni", action="store_true", help="Skip UNI-OBJECTIVE-R* jobs")
    parser.add_argument("--skip-multi", action="store_true", help="Skip MULTI-OBJECTIVE-R* jobs")
    parser.add_argument("--detect-only", action="store_true", help="Only scan and report missing seeds")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip backfill and run evaluation directly")
    parser.add_argument("--force-eval", action="store_true", help="Force rebuilding property/description CSVs")
    return parser.parse_args()


def main():
    args = parse_args()
    include_uni = not args.skip_uni
    include_multi = not args.skip_multi

    report = collect_missing(rounds=args.rounds,
                             chunk_size=args.chunk_size,
                             include_uni=include_uni,
                             include_multi=include_multi)
    json_path, csv_path = write_missing_report(report)

    missing_total = sum(row["missing"] for row in report)
    corrupt_total = sum(row.get("corrupted", 0) for row in report)
    print(f"Missing seed report saved: {json_path}")
    print(f"Missing seed table saved:  {csv_path}")
    print(f"Total missing seeds: {missing_total}")
    print(f"Total corrupted seeds: {corrupt_total}")

    if args.detect_only:
        return

    if not args.evaluate_only and (missing_total > 0 or corrupt_total > 0):
        rerun_missing(report)

    outputs = evaluate(rounds=args.rounds,
                       chunk_size=args.chunk_size,
                       include_uni=include_uni,
                       include_multi=include_multi,
                       force=args.force_eval)
    print("Evaluation outputs:")
    for output in outputs:
        print(f"- {output}")


if __name__ == "__main__":
    main()

