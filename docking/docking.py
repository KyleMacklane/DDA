import subprocess
import os
import tempfile
from pathlib import Path
from multiprocessing import cpu_count
import csv

# ✅ For ligand conversion using Open Babel
def clean_and_prepare_ligand(input_path: str, output_path: str):
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())
    try:
        subprocess.run([
            "obabel", input_path,
            "-O", output_path,
            "-xh", "--gen3d",
            "--partialcharge", "gasteiger"
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Ligand prepared: {output_path}")
    except subprocess.CalledProcessError as e:
        if os.path.exists(output_path):
          os.remove(output_path)
        raise RuntimeError(f"❌ Open Babel error on ligand:\n{e.stderr}")

# ✅ For receptor prep using MGLTools
def prepare_receptor_with_mgltools(input_pdb: str, output_pdbqt: str):
    try:
        subprocess.run([
            r"C:\Program Files (x86)\MGLTools-1.5.7\python.exe",
            r"C:\Program Files (x86)\MGLTools-1.5.7\Lib\site-packages\AutoDockTools\Utilities24\prepare_receptor4.py",
            "-r", r"C:\Users\macklane4040\Desktop\Drug Discovery Agent\docking\receptor\receptor.pdb",
            "-o", r"C:\Users\macklane4040\Desktop\Drug Discovery Agent\docking\receptor\receptor.pdbqt",
            "-A", "checkhydrogens"
        ], check=True)
        print(f"✅ Receptor prepared with MGLTools: {output_pdbqt}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ Error preparing receptor with MGLTools:\n{e}")

# 🧪 Vina docking function
def run_vina_docking(receptor_pdbqt, ligand_pdbqt, center, size, exhaustiveness=8, num_modes=9):
    output_pdbqt = tempfile.NamedTemporaryFile(delete=False, suffix=".pdbqt").name
    log_txt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name

    command = [
        "vina",
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--cpu", str(cpu_count()),
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(size[0]),
        "--size_y", str(size[1]),
        "--size_z", str(size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--out", output_pdbqt,
        "--log", log_txt
    ]

    try:
        subprocess.run(command, check=True)
        with open(log_txt, "r") as f:
            log_content = f.read()
        return log_content
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ Vina docking failed:\n{e}")

# 📊 Extract top affinity
def extract_best_affinity(log: str) -> float:
    for line in log.splitlines():
        if line.strip().startswith("1 "):
            parts = line.split()
            try:
                return float(parts[1])
            except:
                continue
    raise ValueError("❌ No binding affinity found in log.")

def run_docking(pdb_path: str, smiles_list: list[str]) -> str:
    import tempfile
    from rdkit import Chem
    from rdkit.Chem import AllChem

    ligand_dir = Path("docking/ligand")
    ligand_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Save receptor as pdb and convert to pdbqt
    receptor_input = Path("docking/receptor/receptor.pdb")
    receptor_input.write_text(Path(pdb_path).read_text())  # Copy file
    receptor_pdbqt = "docking/receptor/receptor.pdbqt"
    prepare_receptor_with_mgltools(str(receptor_input), receptor_pdbqt)

    for idx, smi in enumerate(smiles_list):
        ligand_name = f"ligand_{idx}"
        sdf_path = ligand_dir / f"{ligand_name}.sdf"
        pdbqt_path = ligand_dir / f"{ligand_name}.pdbqt"

        # Convert SMILES to SDF
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"❌ Invalid SMILES: {smi}")
            continue
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()

        try:
            # Prepare and dock
            clean_and_prepare_ligand(str(sdf_path), str(pdbqt_path))
            log = run_vina_docking(
                receptor_pdbqt,
                str(pdbqt_path),
                center=(0.0, 0.0, 0.0),
                size=(20.0, 20.0, 20.0)
            )
            affinity = extract_best_affinity(log)

            if affinity > -5.0:
                strength = "🔴 Weak binding"
            elif -7.0 < affinity <= -5.0:
                strength = "🟡 Moderate binding"
            else:
                strength = "🟢 Strong binding"

            results.append(f"{ligand_name}: {affinity:.2f} kcal/mol → {strength}")
        except Exception as e:
            results.append(f"{ligand_name}: ❌ Error → {e}")

    return "\n".join(results)


# 🧪 Example usage
if __name__ == "__main__":
    receptor_input = "docking/receptor/receptor.pdb"
    receptor_pdbqt = "docking/receptor/receptor.pdbqt"

    # ✅ Prepare receptor (once)
    prepare_receptor_with_mgltools(receptor_input, receptor_pdbqt)

    # ✅ Dock each ligand in the folder
    ligand_folder = Path("docking/ligand")
    ligand_files = list(ligand_folder.glob("*.sdf"))

    results = []
    print("🔍 Starting docking process...")

    if not ligand_files:
        print("❌ No ligand SDF files found in docking/ligand/")
    else:
        for ligand_file in ligand_files:
            ligand_name = ligand_file.stem
            ligand_pdbqt = ligand_folder / f"{ligand_name}.pdbqt"

            try:
                # ⚗️ Prepare ligand
                clean_and_prepare_ligand(str(ligand_file), str(ligand_pdbqt))
                if not ligand_pdbqt.exists() or os.path.getsize(ligand_pdbqt) < 100:
                    print(f"⚠️ Skipping {ligand_name} — conversion failed or produced empty PDBQT.")
                    ligand_pdbqt.unlink(missing_ok=True)
                    continue

                # 🚀 Dock
                log = run_vina_docking(
                    receptor_pdbqt,
                    str(ligand_pdbqt),
                    center=(0.0, 0.0, 0.0),
                    size=(20.0, 20.0, 20.0)
                )

                # 📊 Affinity
                affinity = extract_best_affinity(log)
                print(f"\n🔍 Ligand: {ligand_name}")
                print(f"🧪 Binding Affinity: {affinity:.2f} kcal/mol")

                # 🔬 Binding strength interpretation
                if affinity > -5.0:
                    strength = "🔴 Weak binding"
                    print(strength)
                elif -7.0 < affinity <= -5.0:
                    strength = "🟡 Moderate binding"
                    print(strength)
                else:
                    strength= "🟢 Strong binding"
                    print(strength)
                
                print("-" * 50)
                # print("🪄 Docking completed successfully.")
                results.append({
                    "Ligand": ligand_name,
                    "Affinity (kcal/mol)": affinity,
                    "Strength": strength
                })


            except Exception as e:
                print(f"❌ Error processing {ligand_name}: {e}")

    
    # 📊 Sort results by best (lowest) affinity
    results.sort(key=lambda x: x["Affinity (kcal/mol)"])

    # 📁 Write to CSV
    output_csv = "docking/docking_results.csv"
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Ligand", "Affinity (kcal/mol)", "Strength"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📄 Results saved to: {output_csv}")
    print("🏅 Top Ligands by Binding Affinity:")
    for i, r in enumerate(results[:5], 1):  # top 5
        print(f"{i}. {r['Ligand']} → {r['Affinity (kcal/mol)']} kcal/mol ({r['Strength']})")        
         