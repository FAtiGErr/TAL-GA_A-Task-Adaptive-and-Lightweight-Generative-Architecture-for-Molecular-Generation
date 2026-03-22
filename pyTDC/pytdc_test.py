from pathlib import Path

import pandas as pd
from tdc.single_pred import ADME



def export_caco2_dataset(property_name="CACO2"):
    data = ADME(name="Caco2_Wang")
    split = data.get_split()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "pyTDC"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = split["train"]
    valid_df = split["valid"]
    test_df = split["test"]

    # Keep raw pyTDC splits for reference.
    train_df.to_csv(out_dir / "caco2_wang_train.csv", index=False)
    valid_df.to_csv(out_dir / "caco2_wang_valid.csv", index=False)
    test_df.to_csv(out_dir / "caco2_wang_test.csv", index=False)

    # Build project training files: SMILES,property without header.
    prop_dir = root / "molProperties" / property_name.upper()
    prop_dir.mkdir(parents=True, exist_ok=True)

    merged_train = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    merged_train[["Drug", "Y"]].to_csv(prop_dir / "train.csv", index=False, header=False)
    test_df[["Drug", "Y"]].to_csv(prop_dir / "test.csv", index=False, header=False)

    print(f"Saved raw splits to: {out_dir}")
    print(f"Saved property dataset to: {prop_dir}")
    print(merged_train.head())


if __name__ == "__main__":
    export_caco2_dataset("CACO2")
