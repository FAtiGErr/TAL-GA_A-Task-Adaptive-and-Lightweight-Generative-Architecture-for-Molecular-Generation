import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm


PROPERTY_RANGES = {
    "LOGP": (0.0, 4.0),
    "TPSA": (25.0, 115.0),
    "SA": (1.5, 3.5),
    "QED": (0.55, 0.95),
}


BASE_DIR = Path(__file__).resolve().parent


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_indices(header):
    lowered = [h.strip().lower() for h in header]
    y_candidates = ["y", "target", "label", "property", "value"]
    smi_candidates = ["smiles", "smile", "drug"]

    y_idx = next((i for i, h in enumerate(lowered) if h in y_candidates), None)
    smi_idx = next((i for i, h in enumerate(lowered) if h in smi_candidates), None)

    if y_idx is None and len(header) >= 2:
        y_idx = len(header) - 1
    if smi_idx is None:
        smi_idx = 0 if y_idx != 0 else 1
    return smi_idx, y_idx


def _read_records(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return []

    header = rows[0]
    has_header = any(str(col).strip().lower() in {"smiles", "smile", "drug", "y", "target", "label", "value"}
                     for col in header)
    smi_idx, y_idx = _infer_indices(header if has_header else ["smiles", "y"])

    start_idx = 1 if has_header else 0
    records = []
    for row in rows[start_idx:]:
        if not row or smi_idx >= len(row) or y_idx >= len(row):
            continue
        y = _to_float(row[y_idx])
        smi = row[smi_idx].strip()
        if y is None or not smi:
            continue
        records.append((smi, float(y)))
    return records


def _range_for_property(dataname, records):
    dataname = dataname.upper()
    if dataname in PROPERTY_RANGES:
        return PROPERTY_RANGES[dataname]

    ys = np.array([y for _, y in records], dtype=np.float32)
    if ys.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(ys, [5, 95])
    if lo == hi:
        lo, hi = float(ys.min()), float(ys.max()) + 1e-6
    return float(lo), float(hi)


def make_param_opt_dataset(dataname="LOGP", label="train"):
    dataname = dataname.upper()
    src = BASE_DIR / dataname / f"{label}.csv"
    dst = BASE_DIR / dataname / f"ParamOpt-{label}.csv"

    records = _read_records(src)
    if not records:
        raise FileNotFoundError(f"No valid samples in {src}")

    min_, max_ = _range_for_property(dataname, records)
    interval = max(max_ - min_, 1e-6)

    if label == "train":
        capacities = [2000] * 7
    else:
        capacities = [500] * 7

    with open(dst, "w", encoding="utf-8", newline="") as csv_writer:
        writer = csv.writer(csv_writer)
        for smi, y in tqdm(records):
            if y < min_ and capacities[0] > 0:
                writer.writerow([smi, y])
                capacities[0] -= 1
            elif min_ <= y < min_ + interval / 5 and capacities[1] > 0:
                writer.writerow([smi, y])
                capacities[1] -= 1
            elif min_ + interval / 5 <= y < min_ + 2 * interval / 5 and capacities[2] > 0:
                writer.writerow([smi, y])
                capacities[2] -= 1
            elif min_ + 2 * interval / 5 <= y < min_ + 3 * interval / 5 and capacities[3] > 0:
                writer.writerow([smi, y])
                capacities[3] -= 1
            elif min_ + 3 * interval / 5 <= y < min_ + 4 * interval / 5 and capacities[4] > 0:
                writer.writerow([smi, y])
                capacities[4] -= 1
            elif min_ + 4 * interval / 5 <= y < max_ and capacities[5] > 0:
                writer.writerow([smi, y])
                capacities[5] -= 1
            elif capacities[6] > 0:
                writer.writerow([smi, y])
                capacities[6] -= 1


def parse_args():
    parser = argparse.ArgumentParser(description="Build ParamOpt datasets for one or more properties.")
    parser.add_argument("--properties", nargs="+", required=True, help="e.g. --properties LOGP TPSA CACO2")
    parser.add_argument("--labels", nargs="+", default=["train", "test"], choices=["train", "test"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for dn in args.properties:
        for lb in args.labels:
            make_param_opt_dataset(dn, lb)
