import requests
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

RAW = Path('data/raw')
RAW.mkdir(parents=True, exist_ok=True)

# 100 well-known X-ray crystal structures
IDS = [
    "1TIM","1UBQ","2HHB","1BNA","1CRN","1HTM","3NIR","1MBO","1GFL","2LYZ",
    "1AKE","1BPI","1CCR","1CDW","1CEW","1CHN","1CID","1CMB","1CMS","1COB",
    "1CSE","1CTJ","1CYO","1DFN","1DGB","1DIG","1DLW","1DNK","1DP7","1DPT",
    "1DRI","1DS1","1DT4","1DU3","1DV1","1DW3","1DXG","1DYP","1E0W","1E1O",
    "1E2A","1E3M","1E4K","1E6M","1E7D","1E8A","1EAH","1EBD","1ECO","1ED8",
    "1EFA","1EFT","1EGJ","1EH2","1EIO","1EJD","1EKL","1ELB","1EMV","1ENH",
    "1EOG","1EPM","1EQK","1EQU","1ER6","1ERK","1ESO","1ETE","1EUL","1EWF",
    "1EXR","1EYS","1EZM","1F13","1F1T","1F2T","1F34","1F3U","1F4N","1F5N",
    "1F6M","1F7L","1F8R","1F9I","1FAW","1FBP","1FCB","1FDS","1FEH","1FFT",
    "1FG9","1FHA","1FIH","1FJ2","1FK5","1FKB","1FKF","1FL2","1FLA","1FLT",
]

records = []
parser = PDBParser(QUIET=True)
failed = []

for pid in tqdm(IDS, desc="Downloading & parsing"):
    try:
        path = RAW / f"{pid}.pdb"
        if not path.exists():
            r = requests.get(
                f'https://files.rcsb.org/download/{pid}.pdb', timeout=15)
            r.raise_for_status()
            if r.text.strip().startswith("<"):
                failed.append(pid)
                continue
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
    except Exception as e:
        failed.append(pid)

df = pd.DataFrame(records)
df.to_csv('residues_dask.csv', index=False)
print(f'Done — {len(df)} residues from {df["pdb_id"].nunique()} structures')
if failed:
    print(f'Failed: {failed}')