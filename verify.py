import torch
import torch_geometric
import pandas as pd
import networkx as nx
from Bio.PDB import PDBParser

print(f"PyTorch:           {torch.__version__}")
print(f"PyTorch Geometric: {torch_geometric.__version__}")
print(f"GPU available:     {torch.cuda.is_available()}")
print(f"Device:            {'CPU (no CUDA)' if not torch.cuda.is_available() else torch.cuda.get_device_name(0)}")
print("All imports OK")