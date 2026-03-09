import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────
EDGE_THRESHOLD_ANGSTROMS = 8.0  # connect residues within 8Å of each other

# 20 standard amino acids in a fixed order for one-hot encoding
AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# ── Feature builders ──────────────────────────────────────────────────────────
def one_hot_residue(res_name: str) -> list[float]:
    """20-dim one-hot vector for residue type. Unknown → all zeros."""
    vec = [0.0] * 20
    idx = AA_TO_IDX.get(res_name.upper())
    if idx is not None:
        vec[idx] = 1.0
    return vec

def build_graph(group: pd.DataFrame) -> Data | None:
    """
    Build a PyTorch Geometric graph from a single protein structure.

    Nodes  = residues
    Edges  = pairs of residues within EDGE_THRESHOLD_ANGSTROMS in 3D space
    Target = mean B-factor of the structure (normalized)
    """
    # drop residues with missing coordinates
    group = group.dropna(subset=['ca_x', 'ca_y', 'ca_z', 'b_factor'])
    if len(group) < 5:
        return None

    coords = group[['ca_x', 'ca_y', 'ca_z']].values.astype(np.float32)
    b_factors = group['b_factor'].values.astype(np.float32)

    # ── Node features (23-dim per residue) ───────────────────────────────────
    node_feats = []
    for _, row in group.iterrows():
        one_hot = one_hot_residue(row['res_name'])          # 20 dims
        b_norm  = float(row['b_factor']) / 100.0            # 1 dim (normalized)
        seq_pos = float(row['res_seq']) / 1000.0            # 1 dim (normalized)
        chain   = 1.0 if row['chain'] == 'A' else 0.0      # 1 dim
        node_feats.append(one_hot + [b_norm, seq_pos, chain])

    x = torch.tensor(node_feats, dtype=torch.float)

    # ── Edges (spatial proximity) ─────────────────────────────────────────────
    edge_index = []
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < EDGE_THRESHOLD_ANGSTROMS:
                edge_index.append([i, j])
                edge_index.append([j, i])  # undirected → add both directions

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # ── Graph-level target ────────────────────────────────────────────────────
    mean_b = float(np.mean(b_factors))
    y = torch.tensor([mean_b], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)

# ── Dataset builder ───────────────────────────────────────────────────────────
def build_dataset(csv_path: str) -> list[Data]:
    """Load CSV and build one graph per structure."""
    df = pd.read_csv(csv_path)
    graphs = []
    skipped = 0

    for pdb_id, group in df.groupby('pdb_id'):
        graph = build_graph(group)
        if graph is None:
            skipped += 1
            continue
        graphs.append(graph)

    logger.info(f"Built {len(graphs)} graphs, skipped {skipped}")
    return graphs

# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    graphs = build_dataset("residues_dask.csv")

    if graphs:
        g = graphs[0]
        logger.info(f"Example graph:")
        logger.info(f"  Nodes:      {g.num_nodes}")
        logger.info(f"  Edges:      {g.num_edges}")
        logger.info(f"  Node feats: {g.x.shape}")
        logger.info(f"  Target:     {g.y.item():.4f}")