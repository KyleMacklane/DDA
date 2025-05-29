import os
import requests
import json
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def search_pdb_structures(query: str, search_by_name=True):
    """Search RCSB for relevant PDB IDs."""
    url = "https://search.rcsb.org/rcsbsearch/v2/query"

    if search_by_name:
        payload = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "operator": "contains_phrase",
                    "value": query,
                    "attribute": "struct.title"
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": 5}
            }
        }
    else:
        payload = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "evalue_cutoff": 0.1,
                    "identity_cutoff": 0,
                    "sequence_type": "protein",
                    "value": query
                }
            },
            "return_type": "polymer_entity",
            "request_options": {
                "paginate": {"start": 0, "rows": 5},
                "results_content_type": ["experimental"]
            }
        }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    results = response.json()

    pdb_ids = list({
        result["identifier"].split("_")[0]
        for result in results.get("result_set", [])
    })

    return pdb_ids


def download_pdb_file(pdb_id:str, save_dir="docking/receptor", save_as="receptor.pdb"):
    """Download receptor PDB file from RCSB."""
    os.makedirs(save_dir, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    response.raise_for_status()
    
    path = os.path.join(save_dir, save_as)
    with open(path, "w") as f:
        f.write(response.text)
    return path


def get_ligand_codes(pdb_id, temp_dir="tmp"):
    """Download mmCIF and extract ligand/component codes."""
    os.makedirs(temp_dir, exist_ok=True)
    cif_path = os.path.join(temp_dir, f"{pdb_id}.cif")

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(url)
    response.raise_for_status()
    
    with open(cif_path, "w") as f:
        f.write(response.text)
    
    cif_dict = MMCIF2Dict(cif_path)
    ligands = cif_dict.get("_chem_comp.id", [])
    types = cif_dict.get("_chem_comp.type", [])

    if isinstance(ligands, str):  # handle single ligand case
        ligands = [ligands]
        types = [types]

    # Filter small molecules only
    ligand_codes = [
        lig for lig, typ in zip(ligands, types)
        if "non-polymer" in typ.lower() or "small molecule" in typ.lower()
    ]

    return list(set(ligand_codes))



def download_ligand_sdf(ligand_code, save_dir="docking/ligand") -> str:
    """Download ligand as SDF from RCSB."""
    os.makedirs(save_dir, exist_ok=True)
    url = f"https://files.rcsb.org/ligands/download/{ligand_code}_ideal.sdf"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"⚠️ Failed to download {ligand_code}")
        return None

    path = os.path.join(save_dir, f"{ligand_code}.sdf")
    with open(path, "w") as f:
        f.write(response.text)
    return path


def prepare_target_and_ligands(query_name):
    """Unified function: gets receptor + all ligands as individual SDFs."""
    print(f"🔍 Searching for '{query_name}'...")
    pdb_ids = search_pdb_structures(query_name)
    if not pdb_ids:
        raise Exception("❌ No PDB IDs found for query.")
    
    pdb_id = pdb_ids[0]
    print(f"✅ Using PDB ID: {pdb_id}")

    receptor_path = download_pdb_file(pdb_id)
    print(f"📥 Receptor downloaded: {receptor_path}")
    
    ligand_codes = get_ligand_codes(pdb_id)
    if not ligand_codes:
        raise Exception("❌ No ligands found in the structure.")

    ligand_paths = []
    for code in ligand_codes:
        path = download_ligand_sdf(code)
        if path:
            print(f"📦 Ligand '{code}' downloaded: {path}")
            ligand_paths.append((code, path))

    return receptor_path, ligand_paths

if __name__ == "__main__":
    receptor_path, ligands = prepare_target_and_ligands("Plasmodium falciparum dihydrofolate reductase")
    print(f"Receptor: {receptor_path}")
    print("Ligands:")
    for code, path in ligands:
        print(f"- {code}: {path}")

