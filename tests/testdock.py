from pdb_utils import prepare_target_and_ligands
from docking.docking import run_docking
import os
from pathlib import Path

# Test parameters
TEST_TARGET = "Plasmodium falciparum dihydrofolate reductase"  # Example malaria target
TEST_SMILES = "O=C(NC1=CC=CC=C1)Nc2ccc(cc2)S(=O)(=O)N"  # Your optimized molecule

def test_full_workflow():
    # Step 1: Get PDB structure and natural ligands
    print("🔄 Fetching PDB structure and ligands...")
    try:
        receptor_path, natural_ligand_paths = prepare_target_and_ligands(TEST_TARGET)
        print(f"✅ Receptor: {receptor_path}")
        print(f"🧪 Natural ligands found: {len(natural_ligand_paths)}")
    except Exception as e:
        print(f"❌ PDB preparation failed: {e}")
        return

    # Step 2: Prepare test molecules (SMILES input)
    test_ligands = [
        TEST_SMILES,  
        "C1=CC=C(C=C1)C=O",  # benzaldehyde
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine as another test case
    ]

    # Step 3: Run docking with direct SMILES input
    print("\n⚛️ Running docking simulation with SMILES input...")
    try:
        results = run_docking(
            pdb_path=receptor_path,
            smiles_list=test_ligands  # Passing SMILES strings directly
        )
        print(f"📊 Docking results:\n{results}")
    except Exception as e:
        print(f"❌ Docking failed: {e}")

    

if __name__ == "__main__":
    # Clean previous test files
    for f in Path("docking").glob("**/test_*"):
        f.unlink()
    
    test_full_workflow()