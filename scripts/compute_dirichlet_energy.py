#!/usr/bin/env python3
"""Compute Dirichlet Energy for node embeddings across all checkpoints.

Dirichlet Energy with degree normalization (from oversquashing literature):
  E(X) = (1/2) * sum_{a_ij} a_ij * ||x_i / sqrt(1 + d_i) - x_j / sqrt(1 + d_j)||_2^2
  
where a_ij is edge weight, d_i is node degree.

Updates the existing CSV with this metric.
"""
import os
import csv
import pandas as pd
import torch
import numpy as np
from pathlib import Path

EDGE_FILE = '/Users/valentinacristoferi/THESIS/ProvIDS/DATA/darpa_theia_reduced_005/edges.csv'
CSV_PATH = Path('/Users/valentinacristoferi/THESIS/ProvIDS/ablations/oversquashing_ablation/node_similarity_stats.csv')

print("Loading edge list and computing degrees...")
edges_df = pd.read_csv(EDGE_FILE)

# Extract edges
edges = edges_df[['src', 'dst']].values.astype(np.int64)
weights = np.ones(len(edges), dtype=np.float32)

print(f"  Loaded {len(edges)} edges")

n_nodes = max(max(edges[:, 0]), max(edges[:, 1])) + 1
degree_array = np.zeros(n_nodes, dtype=np.float32)

# Undirected degree count from edge rows: each (src, dst) adds +1 to both endpoints.
np.add.at(degree_array, edges[:, 0], 1)
np.add.at(degree_array, edges[:, 1], 1)

# Degree normalization: sqrt(1 + degree)
degree_norm = np.sqrt(1.0 + degree_array)

# Convert to torch tensors for efficiency
edges_tensor = torch.from_numpy(edges).long()
weights_tensor = torch.from_numpy(weights).float()
degree_norm_tensor = torch.from_numpy(degree_norm).float()

print(f"  Max node ID: {n_nodes - 1}, mean degree: {degree_array.mean():.2f}")

print("\nComputing Dirichlet Energy for all checkpoints...")

# Read existing CSV
results = []
with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append(row)

# Compute Dirichlet Energy for each checkpoint
for idx, row in enumerate(results):
    ckpt_path = row['checkpoint_path']
    
    if not os.path.exists(ckpt_path):
        print(f"[{idx+1}/{len(results)}] SKIP (not found): {ckpt_path}")
        row['dirichlet_energy'] = None
        continue
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        X = ckpt['model_state_dict']['memory.memory'].float()
        
        # Degree-normalized embeddings: x_i / sqrt(1 + d_i)
        X_norm = X / degree_norm_tensor[:X.shape[0], None]
        
        # Compute Dirichlet Energy: (1/2) * sum_edges a_ij * ||x_i_norm - x_j_norm||_2^2
        X_src = X_norm[edges_tensor[:, 0]]
        X_dst = X_norm[edges_tensor[:, 1]]
        diffs = X_src - X_dst
        sq_dists = (diffs ** 2).sum(dim=1)
        weighted_sq_dists = weights_tensor * sq_dists
        dirichlet_energy = weighted_sq_dists.sum().item()
        
        row['dirichlet_energy'] = dirichlet_energy
        print(f"[{idx+1}/{len(results)}] {row['model']:12} layers={row['n_layers']:3}: E_D = {dirichlet_energy:.6f}")
    except Exception as e:
        print(f"[{idx+1}/{len(results)}] ERROR: {e}")
        row['dirichlet_energy'] = None

# Write updated CSV
print(f"\nWriting updated CSV to {CSV_PATH}...")
if results:
    fieldnames = list(results[0].keys())
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print("✓ Done")
else:
    print("ERROR: No results to write")
