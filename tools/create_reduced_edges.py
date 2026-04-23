"""Create a reduced edges CSV from the last fraction of the dataset.

Takes the last `train_pct` events (preserving order), computes original
train/val/test percentages, applies them to the tail (by slicing) and
writes a new CSV with the assigned `ext_roll` values.

Usage (from repo root):
    python3 create_reduced_edges.py --data_name darpa_theia_0to25 --train_pct 0.2 --out_dir /work3/s253892/ProvIDS/DATA/DATA/darpa_theia_20
    python3 create_reduced_edges.py --data_name darpa_theia_0to25 --train_pct 0.2
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def split_name_from_extroll_series(ext_roll_series: pd.Series) -> tuple:
    arr = ext_roll_series.values
    n = len(arr)
    train = int((arr <= 0).sum())
    val = int((arr == 1).sum())
    test = int((arr >= 2).sum())
    return train, val, test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/work3/s253892/ProvIDS/DATA/DATA", help="Root folder containing DARPA dataset folders")
    p.add_argument("--data_name", required=True, help="Dataset name, e.g. darpa_theia_0to25")
    p.add_argument("--train_pct", default=1.0, type=float, help="Fraction (0-1] of the original TRAINING set to keep (keeps last events of training)")
    p.add_argument("--out", default=None, help="Output CSV path (file). If provided, this file will be written.")
    p.add_argument("--out_dir", default=None, help="Output dataset directory. If omitted, defaults to DATA/<name>_tailXXpct and will contain edges.csv plus copies of other files.")

    args = p.parse_args()

    path = Path(args.data_dir) / args.data_name / "edges.csv"
    if not path.exists():
        raise SystemExit(f"edges.csv not found at {path}")

    print(f"Loading {path} (this may take some time)...")
    df = pd.read_csv(path)

    # Reduce the TRAINING set only: keep the last `train_pct` fraction of rows
    n = len(df)
    # original split counts
    train_c, val_c, test_c = split_name_from_extroll_series(df['ext_roll'])
    print(f"Total events: {n}")
    print(f"Original splits: train={train_c}, val={val_c}, test={test_c}")

    if not (0.0 < args.train_pct <= 1.0):
        raise SystemExit("--train_pct must be in (0,1]")

    # indices of training rows in original order
    train_mask = df['ext_roll'] <= 0
    train_indices = np.where(train_mask.values)[0]
    train_n = len(train_indices)
    keep_train_n = max(1, int(math.floor(train_n * float(args.train_pct))))
    train_start_pos = train_n - keep_train_n
    kept_train_idx = train_indices[train_start_pos:]

    print(f"Reducing training set: original_train={train_n}, keeping_last={keep_train_n} (train_pct={args.train_pct})")

    # build keep mask: keep selected train rows, and all val/test rows unchanged
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[kept_train_idx] = True
    keep_mask[df['ext_roll'] == 1] = True
    keep_mask[df['ext_roll'] >= 2] = True

    reduced_df = df.iloc[np.nonzero(keep_mask)[0]].copy()
    # report malicious distribution
    total_mal = int((df['malicious'].astype(bool)).sum())
    mal_in_reduced = int((reduced_df['malicious'].astype(bool)).sum())
    print(f"Total malicious in original: {total_mal}; in reduced dataset: {mal_in_reduced}")

    # Determine output location. Prefer explicit --out file, else create a new dataset folder
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        reduced_df.to_csv(out_path, index=False)
        print(f"Wrote reduced edges CSV to: {out_path}")
    else:
        # create new dataset folder (copy of original dataset except edges.csv)
        default_out_dir = Path(args.data_dir) / f"{args.data_name}_reduced_train_{int(args.train_pct*100)}pct"
        out_dir = Path(args.out_dir) if args.out_dir else default_out_dir
        if out_dir.exists():
            print(f"Warning: output directory {out_dir} already exists; files may be overwritten.")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)

        # copy other files from original dataset folder
        import shutil

        src_dir = Path(args.data_dir) / args.data_name
        for entry in src_dir.iterdir():
            # skip original edges.csv, and skip processed folders (we'll let user regenerate)
            if entry.name == "edges.csv":
                continue
            if entry.name in ("temporal_processed", "static_processed"):  # skip processed output
                continue
            dest = out_dir / entry.name
            if entry.is_dir():
                try:
                    shutil.copytree(entry, dest)
                except FileExistsError:
                    pass
            else:
                shutil.copy2(entry, dest)

        # write reduced edges.csv into new dataset folder
        out_edges = out_dir / "edges.csv"
        reduced_df.to_csv(out_edges, index=False)
        print(f"Wrote reduced dataset to: {out_dir} (edges.csv inside)")


if __name__ == "__main__":
    main()
