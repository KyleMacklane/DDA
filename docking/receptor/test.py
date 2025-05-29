import subprocess
import os

def test_prepare_receptor():
    # Paths (make sure to update these!)
    input_pdb = "docking/receptor/receptor.pdb"
    output_pdbqt = "docking/receptor/receptor.pdbqt"

    if not os.path.exists(input_pdb):
        print(f"❌ Input file '{input_pdb}' not found. Place your receptor .pdb in this directory.")
        return

    try:
        print("🔧 Running prepare_receptor4.py...")

        result = subprocess.run(
    [
        r"C:\Program Files (x86)\MGLTools-1.5.7\python.exe",  # or the path to your MGLTools Python interpreter
        r"C:\Program Files (x86)\MGLTools-1.5.7\Lib\site-packages\AutoDockTools\Utilities24\prepare_receptor4.py",
        "-r", r"C:\Users\macklane4040\Desktop\Drug Discovery Agent\docking\receptor\receptor.pdb",
        "-o", r"C:\Users\macklane4040\Desktop\Drug Discovery Agent\docking\receptor\receptor.pdbqt",
        "-A", "checkhydrogens"
    ],
    check=True
)


        print("✅ Script ran successfully.")
        print("📄 Output:", result.stdout)

        if os.path.exists(output_pdbqt):
            print(f"🎉 Output file '{output_pdbqt}' was created successfully.")
        else:
            print(f"⚠️ Script ran but output file '{output_pdbqt}' not found.")

    except subprocess.CalledProcessError as e:
        print("❌ Error running prepare_receptor4.py:")
        print(e.stderr)

if __name__ == "__main__":
    test_prepare_receptor()
