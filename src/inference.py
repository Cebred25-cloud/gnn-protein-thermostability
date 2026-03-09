import sys
import torch
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.graph_builder import build_graph
from src.model import ProteinGNN

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("output/model_gcn.pt")
RAW_DIR    = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
def load_model() -> ProteinGNN:
    model = ProteinGNN(hidden_channels=64, use_attention=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu",
                                     weights_only=True))
    model.eval()
    logger.info(f"Loaded model from {MODEL_PATH}")
    return model

# ── Download + parse one structure ────────────────────────────────────────────
def fetch_structure(pdb_id: str) -> pd.DataFrame | None:
    pdb_id = pdb_id.upper().strip()
    path   = RAW_DIR / f"{pdb_id}.pdb"

    if not path.exists():
        logger.info(f"Downloading {pdb_id}...")
        r = requests.get(
            f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=15)
        try:
            r.raise_for_status()
        except Exception:
            logger.error(f"{pdb_id} not found on RCSB")
            return None
        if r.text.strip().startswith("<"):
            logger.error(f"{pdb_id} returned invalid data")
            return None
        path.write_text(r.text)
    else:
        logger.info(f"{pdb_id} already cached")

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_id, str(path))

    records = []
    for model in struct:
        for chain in model:
            for res in chain:
                if res.get_id()[0] != ' ':
                    continue
                if 'CA' not in res:
                    continue
                ca = res['CA']
                x, y, z = ca.get_vector().get_array()
                records.append({
                    'pdb_id':   pdb_id,
                    'chain':    chain.get_id(),
                    'res_name': res.get_resname(),
                    'res_seq':  res.get_id()[1],
                    'ca_x':     x,
                    'ca_y':     y,
                    'ca_z':     z,
                    'b_factor': ca.get_bfactor(),
                })

    return pd.DataFrame(records) if records else None

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(pdb_id: str, model: ProteinGNN) -> float | None:
    df = fetch_structure(pdb_id)
    if df is None:
        return None

    graph = build_graph(df)
    if graph is None:
        logger.error(f"{pdb_id} could not be converted to a graph")
        return None

    with torch.no_grad():
        # add batch dimension (single graph = batch of 1)
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        pred  = model(graph.x, graph.edge_index, batch)

    return pred.item()

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # test on a few structures the model has never seen
    TEST_IDS = ["1FKJ", "1G2A", "1HRC", "1L8T", "2ACE"]

    model = load_model()

    print("\n── Predictions ──────────────────────────────")
    print(f"{'PDB ID':<10} {'Predicted B-factor':>20} {'Interpretation':>20}")
    print("-" * 54)

    for pdb_id in TEST_IDS:
        pred = predict(pdb_id, model)
        if pred is None:
            print(f"{pdb_id:<10} {'FAILED':>20}")
            continue

        # interpret: lower B-factor = more rigid = more thermostable
        if pred < 10:
            interp = "Very rigid"
        elif pred < 20:
            interp = "Moderately rigid"
        elif pred < 40:
            interp = "Flexible"
        else:
            interp = "Very flexible"

        print(f"{pdb_id:<10} {pred:>20.4f} {interp:>20}")