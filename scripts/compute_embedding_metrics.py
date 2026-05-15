#!/usr/bin/env python3
"""Compute oversmoothing statistics from node embedding snapshots.

The embedding CSVs are expected to be snapshot files produced by
``--save_embeddings``:

    snapshot,node_id,observed,last_time,last_batch_idx,last_event_idx,dim_0,...

For each state, this script compares every observed node embedding against
every other observed node embedding. It does not use graph edges.

It can also compute the same metrics on the final exposed TGN memory stored in
``ckpt['final_memory_state'][strategy]``.
"""

import argparse
import csv
import os

import numpy as np
import pandas as pd


def get_embedding_columns(df: pd.DataFrame):
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    if not dim_cols:
        raise SystemExit("No embedding columns found. Expected dim_0, dim_1, ...")
    return sorted(dim_cols, key=lambda c: int(c.split("_")[1]))


def finite_rows(X: np.ndarray):
    return np.isfinite(X).all(axis=1)


def compute_pairwise_cosine_stats(X: np.ndarray):
    """Compute all-pairs cosine mean/std without materializing an NxN matrix."""
    if X.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {X.shape}")

    X = X.astype(np.float32, copy=False)
    X = X[finite_rows(X)]
    n_nodes, _ = X.shape
    if n_nodes <= 1:
        raise SystemExit("Need at least two finite node embeddings.")

    norms = np.linalg.norm(X, axis=1)
    zero_mask = norms == 0
    V = X / np.maximum(norms[:, None], 1e-12)

    self_sim = np.sum(V * V, axis=1)
    sum_v = V.sum(axis=0)

    ordered_off_sum = float(np.dot(sum_v, sum_v) - np.sum(self_sim))
    mean = ordered_off_sum / (n_nodes * (n_nodes - 1))

    # Sum_{i,j} cos(i,j)^2 = ||V^T V||_F^2. Remove diagonal/self terms.
    feature_gram = V.T @ V
    ordered_off_sq_sum = float(np.sum(feature_gram * feature_gram) - np.sum(self_sim * self_sim))
    mean_sq = ordered_off_sq_sum / (n_nodes * (n_nodes - 1))
    std = float(np.sqrt(max(mean_sq - mean * mean, 0.0)))

    return {
        "mean_cosine": float(mean),
        "std_cosine": std,
        "num_zero_vectors": int(np.sum(zero_mask)),
    }


def compute_complete_graph_dirichlet_stats(X: np.ndarray):
    """Compute all-pairs squared-distance energy over the complete graph.

    The total energy is Sum_{i<j} ||x_i - x_j||^2.
    ``mean_dirichlet_pairwise`` and ``std_dirichlet_pairwise`` summarize the
    pairwise squared-distance distribution.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {X.shape}")

    X = X.astype(np.float32, copy=False)
    X = X[finite_rows(X)]
    n_nodes, _ = X.shape
    if n_nodes <= 1:
        raise SystemExit("Need at least two finite node embeddings.")

    sq_norm = np.sum(X * X, axis=1, dtype=np.float64)
    sum_x = np.sum(X, axis=0, dtype=np.float64)

    # Sum_{i<j} ||x_i - x_j||^2 = n * Sum_i ||x_i||^2 - ||Sum_i x_i||^2.
    total_energy = float(n_nodes * np.sum(sq_norm) - np.dot(sum_x, sum_x))
    num_pairs = n_nodes * (n_nodes - 1) / 2
    mean_pairwise = total_energy / num_pairs

    # Closed form for the ordered off-diagonal sum of squared squared-distances.
    feature_gram = X.T.astype(np.float64) @ X.astype(np.float64)
    dot_to_sum = X.astype(np.float64) @ sum_x
    ordered_sq_dist_sq_sum = (
        2 * n_nodes * np.sum(sq_norm * sq_norm)
        + 4 * np.sum(feature_gram * feature_gram)
        + 2 * np.sum(sq_norm) ** 2
        - 8 * np.dot(sq_norm, dot_to_sum)
    )
    mean_sq = ordered_sq_dist_sq_sum / (n_nodes * (n_nodes - 1))
    std_pairwise = float(np.sqrt(max(mean_sq - mean_pairwise * mean_pairwise, 0.0)))

    return {
        "dirichlet_energy": total_energy,
        "mean_dirichlet_pairwise": float(mean_pairwise),
        "std_dirichlet_pairwise": std_pairwise,
    }


def read_snapshot_csv(path: str):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise SystemExit(f"Embedding CSV not found: {path}")

    df = pd.read_csv(path)
    dim_cols = get_embedding_columns(df)

    if "observed" in df.columns:
        df = df[df["observed"].astype(bool)].copy()

    X = df[dim_cols].to_numpy(dtype=np.float32)
    keep = finite_rows(X)
    df = df.iloc[keep].reset_index(drop=True)
    X = X[keep]
    return df, X, path


def load_torch_checkpoint(path: str):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_final_memory_from_ckpt(path: str, strategy: str):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise SystemExit(f"Checkpoint not found: {path}")

    ckpt = load_torch_checkpoint(path)
    if "final_memory_state" not in ckpt:
        raise SystemExit(
            f"Checkpoint does not contain final_memory_state. "
            f"Re-run inference with --save_embeddings first: {path}"
        )
    if strategy not in ckpt["final_memory_state"]:
        available = sorted(ckpt["final_memory_state"].keys())
        raise SystemExit(
            f"Strategy {strategy!r} not found in final_memory_state. Available: {available}"
        )

    state = ckpt["final_memory_state"][strategy]
    memory = state["memory"].detach().cpu().float().numpy()
    return memory, path


def compute_matrix_metrics(
    X: np.ndarray,
    state_name: str,
    embedding_source: str,
    model_name: str,
    num_layers: int,
):
    cosine_stats = compute_pairwise_cosine_stats(X)
    energy_stats = compute_complete_graph_dirichlet_stats(X)

    finite_count = int(np.sum(finite_rows(X)))
    return {
        "model_name": model_name,
        "num_layers": int(num_layers),
        "state": state_name,
        "embedding_source": embedding_source,
        "num_nodes": finite_count,
        "embedding_dim": int(X.shape[1]),
        "num_zero_vectors": cosine_stats["num_zero_vectors"],
        "mean_cosine": cosine_stats["mean_cosine"],
        "std_cosine": cosine_stats["std_cosine"],
        "dirichlet_energy": energy_stats["dirichlet_energy"],
        "mean_dirichlet_pairwise": energy_stats["mean_dirichlet_pairwise"],
        "std_dirichlet_pairwise": energy_stats["std_dirichlet_pairwise"],
    }


def seed_paths(ckpt_dir: str, conf_id: int, seed: int):
    ckpt_dir = os.path.expanduser(ckpt_dir)
    prefix = f"conf_{conf_id}_seed_{seed}"
    return {
        "train": os.path.join(ckpt_dir, f"{prefix}_train_best_node_embeddings.csv"),
        "val": os.path.join(ckpt_dir, f"{prefix}_val_best_node_embeddings.csv"),
        "test": os.path.join(ckpt_dir, f"{prefix}_test_best_node_embeddings.csv"),
        "ckpt": os.path.join(ckpt_dir, f"{prefix}.pt"),
    }


def compute_rows(
    model_name: str,
    num_layers: int,
    strategy: str,
    train_emb_csv: str = None,
    val_emb_csv: str = None,
    test_emb_csv: str = None,
    ckpt: str = None,
    seed: int = None,
):
    rows = []

    for state_name, path in [
        ("train", train_emb_csv),
        ("val", val_emb_csv),
        ("test", test_emb_csv),
    ]:
        if path is None:
            continue
        _, X, _ = read_snapshot_csv(path)
        row = compute_matrix_metrics(
            X=X,
            state_name=state_name,
            embedding_source="gnn",
            model_name=model_name,
            num_layers=num_layers,
        )
        if seed is not None:
            row["seed"] = seed
        rows.append(row)

    if ckpt is not None:
        memory, _ = read_final_memory_from_ckpt(ckpt, strategy)
        row = compute_matrix_metrics(
            X=memory,
            state_name="test",
            embedding_source="memory",
            model_name=model_name,
            num_layers=num_layers,
        )
        if seed is not None:
            row["seed"] = seed
        rows.append(row)

    return rows


def aggregate_seed_rows(rows: list[dict]):
    if not rows:
        return []

    df = pd.DataFrame(rows)
    group_cols = ["model_name", "num_layers", "state", "embedding_source"]
    metric_cols = [
        "num_nodes",
        "num_zero_vectors",
        "mean_cosine",
        "std_cosine",
        "dirichlet_energy",
        "mean_dirichlet_pairwise",
        "std_dirichlet_pairwise",
    ]

    out_rows = []
    for keys, group in df.groupby(group_cols, sort=False):
        out = dict(zip(group_cols, keys))
        out["num_layers"] = int(out["num_layers"])
        out["num_seeds"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        out["embedding_dim"] = int(group["embedding_dim"].iloc[0])

        for col in metric_cols:
            values = group[col].to_numpy(dtype=np.float64)
            out[f"{col}_mean"] = float(np.mean(values))
            out[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        out_rows.append(out)

    return out_rows


def append_rows(csv_path: str, rows: list[dict]):
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-emb-csv", help="Path to train snapshot CSV.")
    parser.add_argument("--val-emb-csv", help="Path to train+val snapshot CSV.")
    parser.add_argument("--test-emb-csv", help="Path to train+val+test snapshot CSV.")
    parser.add_argument("--ckpt", help="Checkpoint containing final_memory_state.")
    parser.add_argument("--ckpt-dir", help="Directory with conf_<id>_seed_<seed> files.")
    parser.add_argument("--conf-id", default=0, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, help="Seeds to aggregate, e.g. --seeds 0 1 2.")
    parser.add_argument("--strategy", default="split", help="Strategy key inside final_memory_state.")
    parser.add_argument("--csv-out", default="node_similarity_stats.csv")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--num-layers", required=True, type=int)

    args = parser.parse_args()

    if args.seeds is not None:
        if args.ckpt_dir is None:
            raise SystemExit("--ckpt-dir is required when using --seeds.")
        seed_rows = []
        for seed in args.seeds:
            paths = seed_paths(args.ckpt_dir, args.conf_id, seed)
            seed_rows.extend(
                compute_rows(
                    model_name=args.model_name,
                    num_layers=args.num_layers,
                    strategy=args.strategy,
                    train_emb_csv=paths["train"],
                    val_emb_csv=paths["val"],
                    test_emb_csv=paths["test"],
                    ckpt=paths["ckpt"],
                    seed=seed,
                )
            )
        rows = aggregate_seed_rows(seed_rows)
    else:
        rows = compute_rows(
            model_name=args.model_name,
            num_layers=args.num_layers,
            strategy=args.strategy,
            train_emb_csv=args.train_emb_csv,
            val_emb_csv=args.val_emb_csv,
            test_emb_csv=args.test_emb_csv,
            ckpt=args.ckpt,
        )

    if not rows:
        raise SystemExit("Provide at least one embedding CSV or --ckpt.")

    csv_out = os.path.expanduser(args.csv_out)
    append_rows(csv_out, rows)

    print("Done.")
    for row in rows:
        print(f"\nState: {row['state']} ({row['embedding_source']})")
        if "num_seeds" in row:
            print(f"  seeds: {row['num_seeds']}")
            print(f"  mean cosine: {row['mean_cosine_mean']:.6f} +/- {row['mean_cosine_std']:.6f}")
            print(
                f"  mean pairwise Dirichlet: "
                f"{row['mean_dirichlet_pairwise_mean']:.6f} +/- "
                f"{row['mean_dirichlet_pairwise_std']:.6f}"
            )
        else:
            print(f"  nodes: {row['num_nodes']}")
            print(f"  mean cosine: {row['mean_cosine']:.6f}")
            print(f"  std cosine: {row['std_cosine']:.6f}")
            print(f"  Dirichlet energy: {row['dirichlet_energy']:.6f}")
            print(f"  mean pairwise Dirichlet: {row['mean_dirichlet_pairwise']:.6f}")
            print(f"  std pairwise Dirichlet: {row['std_dirichlet_pairwise']:.6f}")

    print(f"\nAppended to: {csv_out}")


if __name__ == "__main__":
    main()
