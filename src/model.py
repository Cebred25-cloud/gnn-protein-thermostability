import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from loguru import logger

class ProteinGNN(nn.Module):
    """
    Graph Neural Network for predicting protein thermostability
    from molecular structure graphs.

    Architecture:
        2x GCNConv message-passing layers
        → global mean pooling (node embeddings → graph embedding)
        → 2x linear layers
        → scalar B-factor prediction
    """

    def __init__(
        self,
        in_channels: int = 23,
        hidden_channels: int = 64,
        use_attention: bool = False,
    ):
        super().__init__()
        self.use_attention = use_attention

        # ── Message passing layers ────────────────────────────────────────────
        if use_attention:
            self.conv1 = GATConv(in_channels, hidden_channels, heads=4,
                                 concat=False, dropout=0.1)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4,
                                 concat=False, dropout=0.1)
        else:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # ── Readout MLP ───────────────────────────────────────────────────────
        self.lin1 = nn.Linear(hidden_channels, 32)
        self.lin2 = nn.Linear(32, 1)

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index, batch):
        """
        x          : node feature matrix  [num_nodes, in_channels]
        edge_index : graph connectivity   [2, num_edges]
        batch      : batch vector         [num_nodes] — maps each node to its graph
        """
        # ── Graph convolutions ────────────────────────────────────────────────
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        # ── Global pooling → graph-level embedding ────────────────────────────
        x = global_mean_pool(x, batch)   # [num_graphs, hidden_channels]

        # ── MLP readout ───────────────────────────────────────────────────────
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)

        return x.squeeze(-1)   # [num_graphs] scalar per graph

# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.graph_builder import build_dataset
    from torch_geometric.loader import DataLoader

    graphs = build_dataset("residues_dask.csv")

    loader = DataLoader(graphs, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    # test GCN
    model_gcn = ProteinGNN(use_attention=False)
    out = model_gcn(batch.x, batch.edge_index, batch.batch)
    logger.info(f"GCN output shape:  {out.shape}")
    logger.info(f"GCN predictions:   {out.detach().numpy().round(3)}")

    # test GAT
    model_gat = ProteinGNN(use_attention=True)
    out_gat = model_gat(batch.x, batch.edge_index, batch.batch)
    logger.info(f"GAT output shape:  {out_gat.shape}")
    logger.info(f"GAT predictions:   {out_gat.detach().numpy().round(3)}")

    total_params = sum(p.numel() for p in model_gcn.parameters())
    logger.info(f"GCN total params:  {total_params:,}")