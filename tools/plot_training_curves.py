#!/usr/bin/env python3

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# Workaround for macOS OpenMP duplicate runtime issue (unsafe): allow duplicates
# This must be set before loading libraries that initialize OpenMP (numpy, torch).
import os as _os_for_kmp
_os_for_kmp.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass(frozen=True)
class Curve:
    label: str
    epochs: List[int]
    train_metric: List[float]
    val_metric: List[float]
    train_loss: List[float]
    val_loss: List[float]


@dataclass(frozen=True)
class AggregatedCurve:
    label: str
    epochs: List[int]
    train_metric: List[float]
    val_metric: List[float]
    train_loss: List[float]
    val_loss: List[float]
    train_metric_std: List[float]
    val_metric_std: List[float]
    train_loss_std: List[float]
    val_loss_std: List[float]


def _safe_get(d: dict, key: str) -> Optional[float]:
    v = d.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def load_curve(ckpt_path: str, metric: str) -> Curve:
    # Load full checkpoint (weights_only=False) when available
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    history = ckpt.get("history")
    if not isinstance(history, list) or len(history) == 0:
        raise ValueError(f"Checkpoint {ckpt_path} does not contain a non-empty 'history' list")

    epochs = list(range(len(history)))

    train_metric: List[float] = []
    val_metric: List[float] = []
    train_loss: List[float] = []
    val_loss: List[float] = []

    for i, h in enumerate(history):
        if not isinstance(h, dict) or "train" not in h or "val" not in h:
            raise ValueError(f"Malformed history entry at idx={i}: expected dict with train/val")
        tr = h["train"]
        vl = h["val"]
        if not isinstance(tr, dict) or not isinstance(vl, dict):
            raise ValueError(f"Malformed train/val score at idx={i}: expected dict")

        tr_m = _safe_get(tr, metric)
        vl_m = _safe_get(vl, metric)
        tr_l = _safe_get(tr, "loss")
        vl_l = _safe_get(vl, "loss")

        if tr_m is None or vl_m is None:
            available = sorted(set(tr.keys()) | set(vl.keys()))
            raise KeyError(
                f"Metric '{metric}' not found at idx={i}. Available keys include: {available}"
            )
        if tr_l is None or vl_l is None:
            raise KeyError(
                f"'loss' not found at idx={i}. This script expects eval() to have stored loss in history."
            )

        train_metric.append(tr_m)
        val_metric.append(vl_m)
        train_loss.append(tr_l)
        val_loss.append(vl_l)

    base = os.path.basename(ckpt_path)
    # Prefer short labels that include the 'conf' and/or 'seed' parts when present
    # Examples: conf_0_seed_0.pt -> label 'conf_0 seed_0'; conf_2.pt -> 'conf_2'
    conf_m = re.search(r"(conf[_-]?\d+)", base)
    seed_m = re.search(r"(seed[_-]?\d+)", base)
    if conf_m and seed_m:
        label = f"{conf_m.group(1)} {seed_m.group(1)}"
    elif conf_m:
        label = conf_m.group(1)
    elif seed_m:
        label = seed_m.group(1)
    else:
        label = base

    # keep best info available (used in metric plot label)
    best_epoch = ckpt.get("epoch")
    best_score = ckpt.get("best_score")
    if isinstance(best_epoch, int) and best_score is not None:
        label = f"{label} (best@{best_epoch}={best_score:.4g})"

    return Curve(
        label=label,
        epochs=epochs,
        train_metric=train_metric,
        val_metric=val_metric,
        train_loss=train_loss,
        val_loss=val_loss,
    )


def plot_curves(
    curves: Iterable[Curve],
    metric: str,
    out_path: str,
    title: Optional[str] = None,
    dpi: int = 600,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    curves = list(curves)
    if len(curves) == 0:
        raise ValueError("No curves to plot")

    # set x-axis ticks every 10 epochs (include last epoch if not multiple of 10)
    max_epoch = max(len(c.epochs) for c in curves) - 1
    xticks = list(range(0, max_epoch + 1, 10))
    if len(xticks) == 0 or xticks[-1] != max_epoch:
        xticks.append(max_epoch)

    # default smaller width (suitable for embedding at ~0.5-0.6\linewidth)
    if figsize is None:
        figsize = (6, 5)

    fig, (ax_loss, ax_metric) = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)

    ax_metric.set_xticks(xticks)

    for c in curves:
        # metric label keeps best info; loss label is the short version
        label_metric = c.label
        label_loss = c.label.split(" (best@")[0] if " (best@" in c.label else c.label

        # training: dashed, less prominent
        ax_loss.plot(c.epochs, c.train_loss, linestyle="--", linewidth=1.5, alpha=0.8, label=f"train - {label_loss}")
        ax_metric.plot(c.epochs, c.train_metric, linestyle="--", linewidth=1.5, alpha=0.8, label=f"train - {label_metric}")

        # validation: solid, more visible
        ax_loss.plot(c.epochs, c.val_loss, linestyle="-", linewidth=1.2, alpha=0.8, label=f"val - {label_loss}")
        ax_metric.plot(c.epochs, c.val_metric, linestyle="-", linewidth=1.2, alpha=0.8, label=f"val - {label_metric}")

        # support AggregatedCurve shading if present
        if hasattr(c, "val_metric_std") and getattr(c, "val_metric_std") is not None:
            try:
                epochs = np.array(c.epochs)
                vm = np.array(c.val_metric)
                vms = np.array(c.val_metric_std)
                ax_metric.fill_between(epochs, vm - vms, vm + vms, alpha=0.15)
            except Exception:
                pass
        if hasattr(c, "val_loss_std") and getattr(c, "val_loss_std") is not None:
            try:
                epochs = np.array(c.epochs)
                vl = np.array(c.val_loss)
                vls = np.array(c.val_loss_std)
                ax_loss.fill_between(epochs, vl - vls, vl + vls, alpha=0.12)
            except Exception:
                pass

    # styling and larger fonts for readability
    ax_loss.set_ylabel("loss", fontsize=10)
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend(fontsize=8, ncol=1)
    ax_loss.tick_params(axis="both", which="major", labelsize=8)

    ax_metric.set_ylabel(metric, fontsize=10)
    ax_metric.set_xlabel("epoch", fontsize=10)
    ax_metric.grid(True, alpha=0.25)
    ax_metric.legend(fontsize=8, ncol=1)
    ax_metric.tick_params(axis="both", which="major", labelsize=8)

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Save: use vector formats for best quality when requested, otherwise use provided dpi
    ext = os.path.splitext(out_path)[1].lower()
    vector_exts = {".pdf", ".svg", ".eps"}
    if ext in vector_exts:
        fig.savefig(out_path, bbox_inches="tight")
    else:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss/metric convergence from ProvIDS TGN checkpoints (.pt).")
    parser.add_argument(
        "--ckpt",
        nargs="+",
        required=True,
        help="Path(s) to checkpoint .pt file(s) or directory(ies) containing checkpoints matching conf_*_seed_*.pt",
    )
    parser.add_argument(
        "--metric",
        default="auc",
        help="Metric key to plot from history entries (default: auc). Common: auc, ap, f1_score, accuracy, mse, mae.",
    )
    parser.add_argument(
        "--out",
        default="training_curves.png",
        help="Output image path (default: training_curves.png)",
    )
    parser.add_argument(
        "--title", 
        default=None, 
        help="Optional figure title"
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate multiple runs into mean±std across runs (requires same epoch lengths)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for raster output (default: 600). Ignored for vector formats (.pdf/.svg).",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=None,
        help="Figure size in inches as two floats: WIDTH HEIGHT (e.g. --figsize 14 9)",
    )

    args = parser.parse_args()

    # Resolve ckpt inputs: files, globs or directories containing conf_*_seed_*.pt
    ckpt_inputs = args.ckpt
    ckpt_paths: List[str] = []
    for p in ckpt_inputs:
        p = os.path.expanduser(os.path.expandvars(p))
        if os.path.isdir(p):
            matches = sorted(glob.glob(os.path.join(p, "conf_*_seed_*.pt")))
            if not matches:
                matches = sorted(glob.glob(os.path.join(p, "*.pt")))
            ckpt_paths.extend(matches)
        else:
            expanded = sorted(glob.glob(p))
            if expanded:
                ckpt_paths.extend(expanded)
            else:
                ckpt_paths.append(p)

    if len(ckpt_paths) == 0:
        raise SystemExit("No checkpoint files found for the provided --ckpt inputs")

    curves = [load_curve(p, args.metric) for p in ckpt_paths]

    # Determine title: use provided --title, otherwise derive from ckpt path
    if args.title:
        title = args.title
    else:
        first = ckpt_paths[0]
        dirpart = os.path.basename(os.path.dirname(first))
        title = dirpart if dirpart else os.path.splitext(os.path.basename(first))[0]

    if args.aggregate:
        # aggregate mean/std across runs (requires same epoch lengths)
        def aggregate_curves(curves: Iterable[Curve]) -> AggregatedCurve:
            curves = list(curves)
            if len(curves) == 0:
                raise ValueError("No curves to aggregate")
            lengths = {len(c.epochs) for c in curves}
            if len(lengths) != 1:
                raise ValueError("All runs must have the same number of epochs to aggregate")
            epochs = curves[0].epochs
            tr_m = np.stack([np.array(c.train_metric) for c in curves], axis=0)
            vl_m = np.stack([np.array(c.val_metric) for c in curves], axis=0)
            tr_l = np.stack([np.array(c.train_loss) for c in curves], axis=0)
            vl_l = np.stack([np.array(c.val_loss) for c in curves], axis=0)
            mean_tr_m = np.mean(tr_m, axis=0).tolist()
            mean_vl_m = np.mean(vl_m, axis=0).tolist()
            mean_tr_l = np.mean(tr_l, axis=0).tolist()
            mean_vl_l = np.mean(vl_l, axis=0).tolist()
            std_tr_m = np.std(tr_m, axis=0).tolist()
            std_vl_m = np.std(vl_m, axis=0).tolist()
            std_tr_l = np.std(tr_l, axis=0).tolist()
            std_vl_l = np.std(vl_l, axis=0).tolist()
            label = f"mean ({len(curves)} runs)"
            return AggregatedCurve(
                label=label,
                epochs=epochs,
                train_metric=mean_tr_m,
                val_metric=mean_vl_m,
                train_loss=mean_tr_l,
                val_loss=mean_vl_l,
                train_metric_std=std_tr_m,
                val_metric_std=std_vl_m,
                train_loss_std=std_tr_l,
                val_loss_std=std_vl_l,
            )

        agg = aggregate_curves(curves)
        plot_curves([agg], args.metric, args.out, title=title, dpi=args.dpi, figsize=tuple(args.figsize) if args.figsize else None)
    else:
        plot_curves(curves, args.metric, args.out, title=title, dpi=args.dpi, figsize=tuple(args.figsize) if args.figsize else None)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
