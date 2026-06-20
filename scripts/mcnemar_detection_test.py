#!/usr/bin/env python3
"""Run McNemar's test on paired DARPA detection outputs.

The detection CSVs are expected to have columns:
    hash_id, prob

In this project, lower link-prediction probability means more anomalous, so
the anomaly score is computed as:
    anomaly_score = 1 - prob
"""

from __future__ import annotations

import argparse
import csv
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path


ATTACK_FILES = {
    "theia": [
        "TC3_theia_firefox_backdoor_final_aggregated.csv",
        "TC3_theia_browser_extension_final_aggregated.csv",
    ],
    "trace": [
        "TC3_trace_firefox_backdoor_final_aggregated.csv",
        "TC3_trace_browser_extension_final_aggregated.csv",
        "TC3_trace_pine_phishing_exe_final._aggregated.csv",
        "TC3_trace_thunderbird_phishing_exe_final_aggregated.csv",
    ],
}


def parse_hash_id(value: str) -> int | None:
    value = str(value).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return int(Decimal(value).to_integral_value())
    except (InvalidOperation, ValueError):
        return None


def load_attack_hashes(ground_truth_dir: Path, dataset: str) -> set[int]:
    attack_hashes = set()
    for filename in ATTACK_FILES[dataset]:
        path = ground_truth_dir / filename
        if not path.exists() and "final._aggregated" in filename:
            # Keep compatibility with the typo used in anomaly_detection.py, but
            # prefer the actual filename present in this repo.
            path = ground_truth_dir / filename.replace("final._aggregated", "final_aggregated")
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hash_id = parse_hash_id(row.get("edge_hash_id", ""))
                if hash_id is not None:
                    attack_hashes.add(hash_id)
    return attack_hashes


def load_detection_predictions(
    ckpt_dir: Path,
    conf_id: str,
    split: str,
    seeds: list[int],
    threshold: float,
) -> dict[int, dict[int, dict[str, float | bool]]]:
    per_seed = {}
    for seed in seeds:
        path = ckpt_dir / f"split_conf_{conf_id}_detection_results-{split}_seed_{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        probs = {}
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hash_id = parse_hash_id(row.get("hash_id", ""))
                if hash_id is None:
                    continue
                try:
                    probs[hash_id] = float(row["prob"])
                except (KeyError, TypeError, ValueError):
                    continue
        per_seed[seed] = probs
    return per_seed


def pair_rows(
    a_by_seed: dict[int, dict[int, float]],
    b_by_seed: dict[int, dict[int, float]],
    seeds: list[int],
    threshold: float,
    attack_hashes: set[int],
) -> list[dict]:
    paired = []
    for seed in seeds:
        a_seed = score_seed_predictions(a_by_seed[seed], threshold)
        b_seed = score_seed_predictions(b_by_seed[seed], threshold)
        for hash_id in set(a_seed) & set(b_seed):
            paired.append(build_paired_row(seed, hash_id, a_seed[hash_id], b_seed[hash_id], attack_hashes))
    return paired


def score_seed_predictions(
    probs: dict[int, float],
    threshold: float,
) -> dict[int, dict[str, float | bool]]:
    scored = {}
    for hash_id, prob in probs.items():
        anomaly_score = 1.0 - prob
        scored[hash_id] = {
            "prob": prob,
            "anomaly_score": anomaly_score,
            "pred_anomaly": anomaly_score >= threshold,
        }
    return scored


def build_paired_row(
    run_id: int | None,
    hash_id: int,
    pred_a: dict[str, float | bool],
    pred_b: dict[str, float | bool],
    attack_hashes: set[int],
) -> dict:
    y_true = hash_id in attack_hashes
    pred_anomaly_a = bool(pred_a["pred_anomaly"])
    pred_anomaly_b = bool(pred_b["pred_anomaly"])
    return {
        "run_id": "" if run_id is None else run_id,
        "hash_id": hash_id,
        "y_true": y_true,
        "prob_a": pred_a["prob"],
        "anomaly_score_a": pred_a["anomaly_score"],
        "pred_anomaly_a": pred_anomaly_a,
        "prob_b": pred_b["prob"],
        "anomaly_score_b": pred_b["anomaly_score"],
        "pred_anomaly_b": pred_anomaly_b,
        "correct_a": pred_anomaly_a == y_true,
        "correct_b": pred_anomaly_b == y_true,
    }


def exact_mcnemar_pvalue(b: int, c: int, max_exact_n: int = 1000) -> float | None:
    """Two-sided exact binomial McNemar p-value with p=0.5."""
    n = b + c
    k = min(b, c)
    if n == 0:
        return 1.0
    if n > max_exact_n:
        return None

    # Sum the smaller binomial tail exactly enough for normal test sizes.
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def chi_square_mcnemar(b: int, c: int) -> tuple[float, float]:
    """Continuity-corrected McNemar chi-square statistic and p-value."""
    n = b + c
    if n == 0:
        return 0.0, 1.0

    statistic = (abs(b - c) - 1) ** 2 / n
    # Survival function for chi-square with 1 degree of freedom.
    pvalue = math.erfc(math.sqrt(statistic / 2.0))
    return statistic, pvalue


def confusion_counts(rows: list[dict], pred_key: str) -> dict[str, int]:
    return {
        "tp": sum(1 for row in rows if row["y_true"] and row[pred_key]),
        "fp": sum(1 for row in rows if not row["y_true"] and row[pred_key]),
        "tn": sum(1 for row in rows if not row["y_true"] and not row[pred_key]),
        "fn": sum(1 for row in rows if row["y_true"] and not row[pred_key]),
    }


def print_counts_table(names: list[str], counts: list[dict[str, int]]) -> None:
    name_width = max(len(name) for name in names)
    print(f"{'':{name_width}}  {'tp':>8}  {'fp':>8}  {'tn':>8}  {'fn':>8}")
    for name, row in zip(names, counts):
        print(f"{name:{name_width}}  {row['tp']:8d}  {row['fp']:8d}  {row['tn']:8d}  {row['fn']:8d}")


def print_mcnemar_table(
    name_a: str,
    name_b: str,
    both_correct: int,
    a_correct_b_wrong: int,
    a_wrong_b_correct: int,
    both_wrong: int,
) -> None:
    rows = [
        (f"{name_a} correct", both_correct, a_correct_b_wrong),
        (f"{name_a} wrong", a_wrong_b_correct, both_wrong),
    ]
    col_a = f"{name_b} correct"
    col_b = f"{name_b} wrong"
    label_width = max(len(row[0]) for row in rows)
    value_width = max(len(col_a), len(col_b), 8)
    print(f"{'':{label_width}}  {col_a:>{value_width}}  {col_b:>{value_width}}")
    for label, first, second in rows:
        print(f"{label:{label_width}}  {first:{value_width}d}  {second:{value_width}d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True, type=Path, help="TGN ckpt/results folder")
    parser.add_argument("--model-b", required=True, type=Path, help="TGN+mLSTM ckpt/results folder")
    parser.add_argument("--name-a", default="TGN")
    parser.add_argument("--name-b", default="TGN+mLSTM")
    parser.add_argument("--ground-truth-dir", required=True, type=Path)
    parser.add_argument("--dataset", choices=sorted(ATTACK_FILES), default="theia")
    parser.add_argument("--split", default="0to25")
    parser.add_argument("--conf-id", default="0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    attack_hashes = load_attack_hashes(args.ground_truth_dir, args.dataset)
    a_by_seed = load_detection_predictions(args.model_a, args.conf_id, args.split, args.seeds, args.threshold)
    b_by_seed = load_detection_predictions(args.model_b, args.conf_id, args.split, args.seeds, args.threshold)
    paired = pair_rows(a_by_seed, b_by_seed, args.seeds, args.threshold, attack_hashes)

    a_correct_b_wrong = sum(1 for row in paired if row["correct_a"] and not row["correct_b"])
    a_wrong_b_correct = sum(1 for row in paired if not row["correct_a"] and row["correct_b"])
    both_correct = sum(1 for row in paired if row["correct_a"] and row["correct_b"])
    both_wrong = sum(1 for row in paired if not row["correct_a"] and not row["correct_b"])

    exact_p = exact_mcnemar_pvalue(a_correct_b_wrong, a_wrong_b_correct)
    chi2, chi2_p = chi_square_mcnemar(a_correct_b_wrong, a_wrong_b_correct)

    counts_a = confusion_counts(paired, "pred_anomaly_a")
    counts_b = confusion_counts(paired, "pred_anomaly_b")

    print(f"Paired examples: {len(paired):,}")
    print(f"Positive/anomaly examples: {sum(1 for row in paired if row['y_true']):,}")
    print("Combination mode: concat by run_id")
    print()
    print("Model confusion counts")
    print_counts_table([args.name_a, args.name_b], [counts_a, counts_b])
    print()
    print("McNemar table, based on correctness")
    print_mcnemar_table(
        args.name_a,
        args.name_b,
        both_correct,
        a_correct_b_wrong,
        a_wrong_b_correct,
        both_wrong,
    )
    print()
    print(f"Discordant counts: b={a_correct_b_wrong}, c={a_wrong_b_correct}")
    if exact_p is None:
        print("Exact binomial p-value: skipped because discordant count is > 1000")
    else:
        print(f"Exact binomial p-value: {exact_p:.6g}")
    print(f"Continuity-corrected chi-square: {chi2:.6g}, p-value: {chi2_p:.6g}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            fieldnames = list(paired[0]) if paired else []
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(paired)
        print(f"\nSaved paired predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
