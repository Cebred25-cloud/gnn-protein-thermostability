import requests
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

RAW = Path('data/raw')
RAW.mkdir(parents=True, exist_ok=True)

IDS = ['1TIM','1UBQ','2HHB','1BNA','1CRN','1HTM','3NIR','1MBO','1GFL','2LYZ']
records = []
parser = PDBParser(QUIET=True)

for pid in IDS:
    print(f"Downloading {pid}...")
    r = requests.get(f'https://files.rcsb.org/download/{pid}.pdb', timeout=15)
    path = RAW / f'{pid}.pdb'
    path.write_text(r.text)
    struct = parser.get_structure(pid, str(path))
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
                    'pdb_id':   pid,
                    'chain':    chain.get_id(),
                    'res_name': res.get_resname(),
                    'res_seq':  res.get_id()[1],
                    'ca_x':     x,
                    'ca_y':     y,
                    'ca_z':     z,
                    'b_factor': ca.get_bfactor()
                })

df = pd.DataFrame(records)
df.to_csv('residues_dask.csv', index=False)
print(f'Done — {len(df)} residues from {df["pdb_id"].nunique()} structures')