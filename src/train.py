import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

sys.path.insert(0, '.')
from src.graph_builder import build_dataset
from src.model import ProteinGNN

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS      = 200
LR          = 0.001
BATCH_SIZE  = 4
HIDDEN      = 64
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Metrics ───────────────────────────────────────────────────────────────────
def mae(preds, targets):
    return torch.mean(torch.abs(preds - targets)).item()

def r2(preds, targets):
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return (1 - ss_res / ss_tot).item()

# ── Train / eval loops ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    for batch in loader:
        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        total_loss += loss.item() * batch.num_graphs
        all_preds.append(out)
        all_targets.append(batch.y)
    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return total_loss / len(loader.dataset), mae(preds, targets), r2(preds, targets)

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_loss(train_losses, val_losses, label="GCN"):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")
    ax.tick_params(colors="#8B8FA8")
    ax.xaxis.label.set_color("#C4C6D4")
    ax.yaxis.label.set_color("#C4C6D4")
    ax.title.set_color("#E8E9F0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#23263A")

    ax.plot(train_losses, color="#7C6AF7", linewidth=2, label="Train loss")
    ax.plot(val_losses,   color="#3ECFCF", linewidth=2, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Curve — {label}")
    ax.legend(facecolor="#1A1D27", labelcolor="#C4C6D4", edgecolor="#23263A")
    ax.grid(True, color="#23263A", linewidth=0.7)

    plt.tight_layout()
    out = OUTPUT_DIR / f"loss_curve_{label.lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.success(f"Loss curve saved to {out}")

# ── Main training function ────────────────────────────────────────────────────
def run_training(use_attention: bool = False) -> dict:
    label = "GAT" if use_attention else "GCN"
    logger.info(f"Training {label} model...")

    # load and split dataset
    graphs = build_dataset("residues_dask.csv")
    train_graphs, test_graphs = train_test_split(
        graphs, test_size=0.2, random_state=SEED
    )
    train_graphs, val_graphs = train_test_split(
        train_graphs, test_size=0.15, random_state=SEED
    )

    logger.info(f"Split — train: {len(train_graphs)}, "
                f"val: {len(val_graphs)}, test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE)

    # model, optimizer, loss
    model     = ProteinGNN(hidden_channels=HIDDEN, use_attention=use_attention)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_mae, v_r2 = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch:3d} | "
                        f"Train {t_loss:.4f} | "
                        f"Val {v_loss:.4f} | "
                        f"MAE {v_mae:.4f} | "
                        f"R2 {v_r2:.4f}")

    # load best model and evaluate on test set
    model.load_state_dict(best_state)
    test_loss, test_mae, test_r2 = eval_epoch(model, test_loader, criterion)

    logger.success(f"\n── {label} Test Results ──────────────────")
    logger.success(f"  Test MSE: {test_loss:.4f}")
    logger.success(f"  Test MAE: {test_mae:.4f}")
    logger.success(f"  Test R²:  {test_r2:.4f}")

    # save model
    model_path = OUTPUT_DIR / f"model_{label.lower()}.pt"
    torch.save(model.state_dict(), model_path)
    logger.success(f"  Model saved to {model_path}")

    plot_loss(train_losses, val_losses, label=label)

    return {"label": label, "mse": test_loss, "mae": test_mae, "r2": test_r2}

# ── Baseline ──────────────────────────────────────────────────────────────────
def run_baseline() -> dict:
    """Predict the dataset mean B-factor for every structure."""
    graphs  = build_dataset("residues_dask.csv")
    targets = torch.tensor([g.y.item() for g in graphs])
    mean    = targets.mean()
    preds   = mean.expand(len(targets))

    baseline_mae = mae(preds, targets)
    baseline_r2  = r2(preds, targets)

    logger.info(f"\n── Baseline (predict mean) ──────────────")
    logger.info(f"  MAE: {baseline_mae:.4f}")
    logger.info(f"  R²:  {baseline_r2:.4f}")

    return {"label": "Baseline", "mae": baseline_mae, "r2": baseline_r2}

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    baseline = run_baseline()
    gcn_results = run_training(use_attention=False)
    gat_results = run_training(use_attention=True)

    # summary table
    print("\n── Final Comparison ─────────────────────────────")
    print(f"{'Model':<12} {'MAE':>8} {'R²':>8}")
    print("-" * 32)
    for r in [baseline, gcn_results, gat_results]:
        mae_val = r.get('mae', float('nan'))
        r2_val  = r.get('r2',  float('nan'))
        print(f"{r['label']:<12} {mae_val:>8.4f} {r2_val:>8.4f}")