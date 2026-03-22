import os
import argparse
import warnings
from collections import defaultdict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import moses
import pandas as pd

from config import PSO_RESULTS_DIR, set_working_directory


warnings.filterwarnings("ignore")
set_working_directory()


UNI_TASKS = [
    ("LOGP", "1.0"), ("LOGP", "2.0"), ("LOGP", "3.0"), ("LOGP", "4.0"),
    ("TPSA", "20.0"), ("TPSA", "40.0"), ("TPSA", "60.0"), ("TPSA", "80.0"),
]
TRI_TASKS = [
    ("LOGP-TPSA", "1.0-20.0"), ("LOGP-TPSA", "2.0-40.0"),
    ("LOGP-TPSA", "3.0-60.0"), ("LOGP-TPSA", "4.0-80.0"),
]


def _property_file(objective, prop, target):
    return os.path.join(PSO_RESULTS_DIR, objective, prop, f"{target} Property.csv")


def _round_objectives(prefix, rounds):
    return [f"{prefix}-R{i}" for i in range(1, rounds + 1)]


def _safe_metrics(smiles):
    if not smiles:
        return None
    return moses.get_all_metrics(smiles)


def evaluate_round(objective, tasks):
    rows = []
    by_task_smiles = {}

    for prop, target in tasks:
        src = _property_file(objective, prop, target)
        if not os.path.exists(src):
            print(f"[SKIP] Missing property file: {src}")
            continue

        df = pd.read_csv(src)
        if "SMILES" not in df.columns:
            print(f"[SKIP] Missing SMILES column: {src}")
            continue

        smiles = df["SMILES"].dropna().astype(str).tolist()
        metrics = _safe_metrics(smiles)
        if metrics is None:
            print(f"[SKIP] Empty SMILES: {src}")
            continue

        row = {
            "objective": objective,
            "property": prop,
            "target": target,
            "num_smiles": len(smiles),
        }
        row.update(metrics)
        rows.append(row)
        by_task_smiles[(prop, target)] = smiles
        print(f"[OK] {objective} | {prop} | {target} | n={len(smiles)}")

    out_csv = os.path.join(PSO_RESULTS_DIR, objective, "moses_metrics.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv, by_task_smiles


def pooled_summary(base_objective, pooled_smiles_map):
    rows = []
    for (prop, target), smiles in pooled_smiles_map.items():
        metrics = _safe_metrics(smiles)
        if metrics is None:
            continue
        row = {
            "objective": base_objective,
            "property": prop,
            "target": target,
            "num_smiles": len(smiles),
        }
        row.update(metrics)
        rows.append(row)

    out_csv = os.path.join(PSO_RESULTS_DIR, base_objective, "moses_metrics_pooled.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def mean_std_summary(base_objective, round_csv_files):
    frames = []
    for csv_file in round_csv_files:
        if os.path.exists(csv_file):
            frames.append(pd.read_csv(csv_file))
    if not frames:
        out_csv = os.path.join(PSO_RESULTS_DIR, base_objective, "moses_metrics_mean_std.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        pd.DataFrame().to_csv(out_csv, index=False)
        return out_csv

    all_df = pd.concat(frames, ignore_index=True)
    all_numeric = all_df.select_dtypes(include=["number"]).columns.tolist()
    metric_cols = [c for c in all_numeric if c != "num_smiles"]

    grouped = all_df.groupby(["property", "target"], as_index=False)
    mean_df = grouped[metric_cols].mean().add_suffix("_mean")
    std_df = grouped[metric_cols].std(ddof=0).fillna(0).add_suffix("_std")

    mean_df.rename(columns={"property_mean": "property", "target_mean": "target"}, inplace=True)
    std_df.rename(columns={"property_std": "property", "target_std": "target"}, inplace=True)
    out_df = pd.merge(mean_df, std_df, on=["property", "target"], how="inner")
    out_df.insert(0, "objective", base_objective)

    out_csv = os.path.join(PSO_RESULTS_DIR, base_objective, "moses_metrics_mean_std.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_csv


def evaluate_family(prefix, rounds, tasks):
    objectives = _round_objectives(prefix, rounds)
    round_csv_files = []
    pooled = defaultdict(list)

    for objective in objectives:
        out_csv, task_smiles = evaluate_round(objective, tasks)
        round_csv_files.append(out_csv)
        for key, smiles in task_smiles.items():
            pooled[key].extend(smiles)

    base_objective = prefix
    pooled_csv = pooled_summary(base_objective, pooled)
    mean_std_csv = mean_std_summary(base_objective, round_csv_files)
    return round_csv_files, pooled_csv, mean_std_csv


def parse_args():
    parser = argparse.ArgumentParser(description="MOSES benchmark for chunk folders only (R1..Rn).")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds to evaluate, default: 5")
    parser.add_argument("--skip-uni", action="store_true", help="Skip UNI-OBJECTIVE-R* evaluation")
    parser.add_argument("--skip-multi", action="store_true", help="Skip MULTI-OBJECTIVE-R* evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = []

    if not args.skip_uni:
        uni_round_csvs, uni_pooled_csv, uni_mean_std_csv = evaluate_family("UNI-OBJECTIVE", args.rounds, UNI_TASKS)
        outputs.extend(uni_round_csvs)
        outputs.append(uni_pooled_csv)
        outputs.append(uni_mean_std_csv)

    if not args.skip_multi:
        tri_round_csvs, tri_pooled_csv, tri_mean_std_csv = evaluate_family("MULTI-OBJECTIVE", args.rounds, TRI_TASKS)
        outputs.extend(tri_round_csvs)
        outputs.append(tri_pooled_csv)
        outputs.append(tri_mean_std_csv)

    print("MOSES benchmark finished. Generated files:")
    for out in outputs:
        print(f"- {out}")


if __name__ == "__main__":
    main()

