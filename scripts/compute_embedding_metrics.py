#!/usr/bin/env python3
"""Compute oversmoothing statistics from long-format node embeddings.

This script computes three cumulative states:

    train state = train_best
    val state   = train_best + val_best
    test state  = train_best + val_best + test_best

Metrics:

1. Cosine similarity:
   - aggregate to latest unique embedding per node
   - compute mean cosine similarity over unique node embeddings

2. Dirichlet energy:
   - computed event-by-event over observed positive edges
   - uses the src and pos_dst embeddings saved for that same edge event
   - ignores negative sampled edges

Expected embedding CSV format:

    split,batch_idx,event_idx,time,node_id,role,edge_label,dim_0,dim_1,...

where role is one of:

    src
    pos_dst
    neg_dst

and edge_label is one of:

    pos
    neg

Usage:

python /work3/s253892/ProvIDS/scripts/compute_embedding_metrics.py \
    --train-emb-csv ./train_best_node_embeddings.csv \
    --val-emb-csv ./val_best_node_embeddings.csv \
    --test-emb-csv ./test_best_node_embeddings.csv \
    --csv-out /work3/s253892/ProvIDS/final_experiments/oversquashing_ablation/node_similarity_stats.csv \
    --model-name \
    --num-layers
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


def latest_unique_embeddings(df: pd.DataFrame):
    """Keep the latest observed embedding for each node_id."""
    required = ["node_id", "time", "batch_idx", "event_idx"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Embedding CSV is missing required columns: {missing}")

    df = df.sort_values(["time", "batch_idx", "event_idx"], kind="mergesort")
    df = df.drop_duplicates("node_id", keep="last")

    return df.reset_index(drop=True)


def compute_cosine_stats(X: np.ndarray):
    """Compute mean per-node cosine similarity to all other unique nodes."""
    if X.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {X.shape}")

    n_nodes, _ = X.shape

    if n_nodes <= 1:
        raise SystemExit("Need at least two nodes to compute cosine similarity.")

    X = X.astype(np.float32, copy=False)

    norms = np.linalg.norm(X, axis=1)
    zero_mask = norms == 0

    eps = 1e-12
    V = X / np.maximum(norms[:, None], eps)

    sum_all = V.sum(axis=0)
    dot_all = V @ sum_all

    # For nonzero vectors, self-similarity is 1.
    # For zero vectors, self-similarity is 0.
    self_sim = np.sum(V * V, axis=1)

    per_node_mean = (dot_all - self_sim) / (n_nodes - 1)

    return {
        "mean_cosine": float(np.mean(per_node_mean)),
        "min_cosine": float(np.min(per_node_mean)),
        "max_cosine": float(np.max(per_node_mean)),
        "num_zero_vectors": int(np.sum(zero_mask)),
    }


def compute_temporal_dirichlet_energy(df: pd.DataFrame):
    """Compute event-level Dirichlet energy over positive observed edges.

    For each positive edge event, this uses the source and positive destination
    embeddings saved for that exact event.

    Negative sampled edges are ignored.

    Degree normalization uses event-count degree over the cumulative positive
    edge events in the given state.
    """
    required = [
        "split",
        "node_id",
        "role",
        "edge_label",
        "time",
        "batch_idx",
        "event_idx",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Embedding CSV is missing required columns: {missing}")

    dim_cols = get_embedding_columns(df)

    group_cols = ["split", "time", "batch_idx", "event_idx"]

    pos_df = df[
        (df["edge_label"] == "pos") &
        (df["role"].isin(["src", "pos_dst"]))
    ].copy()

    if pos_df.empty:
        return {
            "dirichlet_energy": None,
            "dirichlet_energy_per_edge": None,
            "num_positive_edge_events": 0,
        }

    # Avoid duplicated rows if embedding files were accidentally appended twice.
    pos_df = pos_df.drop_duplicates(group_cols + ["role"], keep="last")

    src_df = pos_df[pos_df["role"] == "src"][group_cols + ["node_id"] + dim_cols].copy()
    dst_df = pos_df[pos_df["role"] == "pos_dst"][group_cols + ["node_id"] + dim_cols].copy()

    src_df = src_df.rename(
        columns={
            "node_id": "src",
            **{c: f"src_{c}" for c in dim_cols},
        }
    )

    dst_df = dst_df.rename(
        columns={
            "node_id": "dst",
            **{c: f"dst_{c}" for c in dim_cols},
        }
    )

    edge_df = src_df.merge(dst_df, on=group_cols, how="inner")

    if edge_df.empty:
        return {
            "dirichlet_energy": None,
            "dirichlet_energy_per_edge": None,
            "num_positive_edge_events": 0,
        }

    # Remove self-loops.
    edge_df = edge_df[edge_df["src"] != edge_df["dst"]].reset_index(drop=True)

    if edge_df.empty:
        return {
            "dirichlet_energy": None,
            "dirichlet_energy_per_edge": None,
            "num_positive_edge_events": 0,
        }

    # Event-count degree over positive observed edge events.
    degree = {}

    for src, dst in zip(edge_df["src"], edge_df["dst"]):
        src = int(src)
        dst = int(dst)

        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1

    src_X = edge_df[[f"src_{c}" for c in dim_cols]].to_numpy(dtype=np.float32)
    dst_X = edge_df[[f"dst_{c}" for c in dim_cols]].to_numpy(dtype=np.float32)

    src_deg = edge_df["src"].map(lambda n: degree[int(n)]).to_numpy(dtype=np.float32)
    dst_deg = edge_df["dst"].map(lambda n: degree[int(n)]).to_numpy(dtype=np.float32)

    src_X_norm = src_X / np.sqrt(1.0 + src_deg[:, None])
    dst_X_norm = dst_X / np.sqrt(1.0 + dst_deg[:, None])

    sq_dists = np.sum((src_X_norm - dst_X_norm) ** 2, axis=1)

    energy = float(np.sum(sq_dists))
    energy_per_edge = float(np.mean(sq_dists))

    return {
        "dirichlet_energy": energy,
        "dirichlet_energy_per_edge": energy_per_edge,
        "num_positive_edge_events": int(len(edge_df)),
    }


def read_embedding_csv(path: str, split_name: str):
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        raise SystemExit(f"Embedding CSV not found: {path}")

    df = pd.read_csv(path)

    # Override/ensure split label so cumulative grouping is unambiguous.
    df["split"] = split_name

    return df, path


def compute_state_metrics(
    state_name: str,
    state_df: pd.DataFrame,
    source_files: list[str],
    model_name: str,
    num_layers: int,
):
    """Compute cosine and event-level Dirichlet energy for one cumulative state."""
    latest_df = latest_unique_embeddings(state_df)
    dim_cols = get_embedding_columns(latest_df)

    X = latest_df[dim_cols].to_numpy(dtype=np.float32)

    cosine_stats = compute_cosine_stats(X)
    energy_stats = compute_temporal_dirichlet_energy(state_df)

    return {
        "model_name": model_name,
        "num_layers": int(num_layers),
        "state": state_name,
        "embedding_csvs": " + ".join(source_files),
        "num_observations": int(len(state_df)),
        "num_unique_nodes": int(len(latest_df)),
        "embedding_dim": int(X.shape[1]),
        "num_zero_vectors": cosine_stats["num_zero_vectors"],
        "mean_cosine": cosine_stats["mean_cosine"],
        "min_cosine": cosine_stats["min_cosine"],
        "max_cosine": cosine_stats["max_cosine"],
        "dirichlet_energy": energy_stats["dirichlet_energy"],
        "dirichlet_energy_per_edge": energy_stats["dirichlet_energy_per_edge"],
        "num_positive_edge_events": energy_stats["num_positive_edge_events"],
    }


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

    parser.add_argument(
        "--train-emb-csv",
        required=True,
        help="Path to train_best_node_embeddings.csv",
    )

    parser.add_argument(
        "--val-emb-csv",
        required=True,
        help="Path to val_best_node_embeddings.csv",
    )

    parser.add_argument(
        "--test-emb-csv",
        required=True,
        help="Path to test_best_node_embeddings.csv",
    )

    parser.add_argument(
        "--csv-out",
        default="node_similarity_stats.csv",
        help="Summary CSV to append results to.",
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name to store in the output CSV.",
    )

    parser.add_argument(
        "--num-layers",
        required=True,
        type=int,
        help="Number of layers to store in the output CSV.",
    )

    args = parser.parse_args()

    train_df, train_path = read_embedding_csv(args.train_emb_csv, "train_best")
    val_df, val_path = read_embedding_csv(args.val_emb_csv, "val_best")
    test_df, test_path = read_embedding_csv(args.test_emb_csv, "test_best")

    train_state_df = train_df
    val_state_df = pd.concat([train_df, val_df], ignore_index=True)
    test_state_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    rows = [
        compute_state_metrics(
            state_name="train",
            state_df=train_state_df,
            source_files=[train_path],
            model_name=args.model_name,
            num_layers=args.num_layers,
        ),
        compute_state_metrics(
            state_name="val",
            state_df=val_state_df,
            source_files=[train_path, val_path],
            model_name=args.model_name,
            num_layers=args.num_layers,
        ),
        compute_state_metrics(
            state_name="test",
            state_df=test_state_df,
            source_files=[train_path, val_path, test_path],
            model_name=args.model_name,
            num_layers=args.num_layers,
        ),
    ]

    csv_out = os.path.expanduser(args.csv_out)
    append_rows(csv_out, rows)

    print("Done.")

    for row in rows:
        print(f"\nState: {row['state']}")
        print(f"  observations: {row['num_observations']}")
        print(f"  unique nodes: {row['num_unique_nodes']}")
        print(f"  positive edge events: {row['num_positive_edge_events']}")
        print(f"  mean cosine: {row['mean_cosine']:.6f}")
        print(f"  dirichlet energy: {row['dirichlet_energy']}")
        print(f"  dirichlet energy per edge: {row['dirichlet_energy_per_edge']}")

    print(f"\nAppended to: {csv_out}")


if __name__ == "__main__":
    main()