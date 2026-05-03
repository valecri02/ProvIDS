#!/usr/bin/env python3
"""Compute per-node mean cosine similarity to all other nodes from a checkpoint.

Usage example:
  python scripts/compute_node_similarity.py \
    --ckpt ablations/oversquashing_ablation/no_mem_sage/300/TGN/ckpt/conf_0_seed_0.pt \
    --out results/node_mean_sim.npy

The script picks a candidate 2D tensor from the checkpoint state_dict that likely
contains per-node vectors, preferring keys containing 'memory'.

For each node, it computes the mean cosine similarity to all other nodes:

  cos_sim(i, j) = (v_i · v_j) / (||v_i|| * ||v_j||)

Then it averages across all j != i to get one score per node.

Important:
  This implementation correctly handles zero vectors. For zero vectors, the
  normalized vector remains zero, so its self-similarity is 0, not 1.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch


def find_candidate_tensor(msd: dict):
    """Find a likely node embedding / memory tensor in the model state_dict."""
    candidates = []

    for k, v in msd.items():
        if not hasattr(v, "ndim"):
            continue

        if v.ndim != 2:
            continue

        candidates.append((k, v.shape[0], v.shape[1]))

    # Prefer keys containing "memory"
    mem_keys = [c for c in candidates if "memory" in c[0].lower()]

    if mem_keys:
        # Choose memory tensor with the largest number of rows
        mem_keys.sort(key=lambda x: x[1], reverse=True)
        return mem_keys[0][0]

    if candidates:
        # Fallback: choose largest 2D tensor by number of rows
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return None


def infer_model_and_layers(ckpt_path: str):
    """Infer model name and number of layers from checkpoint path."""
    path = Path(ckpt_path).resolve()
    parts = path.parts

    idx = None

    for marker in ("oversquashing_ablation", "oversquashing_abl"):
        if marker in parts:
            idx = parts.index(marker)
            break

    if idx is None:
        return None, None

    model = parts[idx + 1] if idx + 1 < len(parts) else None

    n_layers = None
    if idx + 2 < len(parts):
        try:
            n_layers = int(parts[idx + 2])
        except ValueError:
            n_layers = None

    return model, n_layers


def compute_mean_cosine_similarity(
    mat: np.ndarray,
    out_path: str | None = None,
    save_per_node: bool = True,
):
    """Compute mean cosine similarity of each row to all other rows.

    Args:
        mat:
            Matrix of shape (N, D), where N is number of nodes and D is
            embedding dimension.
        out_path:
            Path where per-node mean similarities should be saved.
        save_per_node:
            Whether to save per-node mean similarities as a .npy file.

    Returns:
        Dictionary with summary statistics.
    """
    if mat.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {mat.shape}")

    N, D = mat.shape

    if N <= 1:
        raise ValueError("Need at least 2 nodes to compute cosine similarities")

    mat = mat.astype(np.float32, copy=False)

    # Normalize rows.
    # Zero vectors remain zero.
    norms = np.linalg.norm(mat, axis=1)
    eps = 1e-12
    V = mat / np.maximum(norms[:, None], eps)

    # Efficient computation of:
    #
    #   mean_sim_i = (1 / (N - 1)) * sum_{j != i} cos(v_i, v_j)
    #
    # Since rows of V are normalized, cosine similarity is dot product.
    #
    # For nonzero vectors:
    #   self_sim_i = V[i] dot V[i] = 1
    #
    # For zero vectors:
    #   self_sim_i = V[i] dot V[i] = 0
    #
    # This is why we subtract self_sim instead of blindly subtracting 1.0.
    sum_all = V.sum(axis=0)
    dot_all = V.dot(sum_all)

    self_sim = np.sum(V * V, axis=1)
    mean_sim = (dot_all - self_sim) / (N - 1)

    stats = {
        "per_node_mean": None,
        "mean_of_means": float(np.mean(mean_sim)),
        "min_of_means": float(np.min(mean_sim)),
        "max_of_means": float(np.max(mean_sim)),
        "num_nodes": int(N),
        "dim": int(D),
        "num_zero_vectors": int(np.sum(norms == 0)),
    }

    if save_per_node and out_path is not None:
        out_path = os.path.expanduser(out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        np.save(out_path, mean_sim.astype(np.float32))
        stats["per_node_mean"] = out_path

    return stats


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to .pt checkpoint",
    )

    parser.add_argument(
        "--tensor-key",
        default=None,
        help="Explicit state_dict key to use",
    )

    parser.add_argument(
        "--out",
        default="node_mean_sim.npy",
        help="Output .npy path for per-node mean similarities",
    )

    parser.add_argument(
        "--csv-out",
        default="/Users/valentinacristoferi/THESIS/ProvIDS/ablations/oversquashing_ablation/node_similarity_stats.csv",
        help="CSV file to append summary results to",
    )

    parser.add_argument(
        "--no-save-per-node",
        dest="save_per_node",
        action="store_false",
        help="Do not save per-node mean similarities",
    )

    args = parser.parse_args()

    ckpt_path = os.path.expanduser(args.ckpt)

    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise SystemExit(
            "Unsupported checkpoint format: expected a dict with key "
            "'model_state_dict'."
        )

    msd = ckpt["model_state_dict"]

    key = args.tensor_key

    if key is None:
        key = find_candidate_tensor(msd)

    if key is None:
        raise SystemExit(
            "Could not locate a suitable 2D tensor in the checkpoint state_dict."
        )

    if key not in msd:
        raise SystemExit(f"Tensor key not found in model_state_dict: {key}")

    print(f"Using tensor key: {key}")

    tensor = msd[key]

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    mat = tensor.detach().cpu().numpy().astype(np.float32)

    print(f"Loaded matrix shape: {mat.shape}")

    stats = compute_mean_cosine_similarity(
        mat,
        out_path=args.out if args.save_per_node else None,
        save_per_node=args.save_per_node,
    )

    print("Summary:")
    print(f"  num_nodes={stats['num_nodes']}")
    print(f"  dim={stats['dim']}")
    print(f"  num_zero_vectors={stats['num_zero_vectors']}")
    print(f"  mean_of_means={stats['mean_of_means']:.6g}")
    print(f"  min_of_means={stats['min_of_means']:.6g}")
    print(f"  max_of_means={stats['max_of_means']:.6g}")

    if stats["per_node_mean"] is not None:
        print(f"Saved per-node means to: {stats['per_node_mean']}")

    model, n_layers = infer_model_and_layers(ckpt_path)

    csv_path = os.path.expanduser(args.csv_out)
    csv_dir = os.path.dirname(csv_path)

    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    row = {
        "checkpoint_path": ckpt_path,
        "model": model,
        "n_layers": n_layers,
        "tensor_key": key,
        "num_nodes": stats["num_nodes"],
        "dim": stats["dim"],
        "num_zero_vectors": stats["num_zero_vectors"],
        "mean_of_means": stats["mean_of_means"],
        "min_of_means": stats["min_of_means"],
        "max_of_means": stats["max_of_means"],
    }

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"Appended results to: {csv_path}")


if __name__ == "__main__":
    main()