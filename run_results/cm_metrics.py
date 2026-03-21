#!/usr/bin/env python3
"""
Compute precision, recall, accuracy and FPR from a 2x2 confusion matrix.

Expected matrix format (scikit-learn default):
[[TN, FP],
 [FN, TP]]
"""

from __future__ import annotations

import ast
import argparse


def metrics_from_cm(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "fpr": fpr,
    }


def parse_cm(cm_str: str) -> tuple[int, int, int, int]:
    """
    Accepts strings like:
      '[[3924380  749146]\n [  66997 4606529]]'
    or:
      '[[3924380, 749146], [66997, 4606529]]'
    """
    s = cm_str.strip()

    # If it's missing commas (numpy-style), add commas between numbers.
    # This makes it valid Python list syntax for ast.literal_eval.
    if "," not in s:
        s = s.replace("\n", " ")
        s = " ".join(s.split())  # normalize spaces
        s = s.replace("[ ", "[").replace(" ]", "]")
        s = s.replace("][", "], [")  # safety for edge cases
        # insert commas between adjacent integers
        out = []
        prev_was_digit = False
        for ch in s:
            if ch.isdigit():
                out.append(ch)
                prev_was_digit = True
            else:
                if prev_was_digit and ch == " ":
                    out.append(", ")
                else:
                    out.append(ch)
                prev_was_digit = False
        s = "".join(out).replace(", ]", "]")
    cm = ast.literal_eval(s)

    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    return tn, fp, fn, tp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm", required=True, help='Confusion matrix string, e.g. \'[[TN FP],[FN TP]]\'')
    args = ap.parse_args()

    tn, fp, fn, tp = parse_cm(args.cm)
    m = metrics_from_cm(tn, fp, fn, tp)

    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"precision = {m['precision']:.6f}")
    print(f"recall    = {m['recall']:.6f}")
    print(f"accuracy  = {m['accuracy']:.6f}")
    print(f"FPR       = {m['fpr']:.6f}")


if __name__ == "__main__":
    main()