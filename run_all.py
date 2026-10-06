"""Run all reproducible experiments and regenerate repository outputs."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
scripts = [
    ROOT / "src" / "reproduce_figure7.py",
    ROOT / "src" / "synthetic_ldp_experiment.py",
]

for script in scripts:
    print(f"\n=== Running {script.name} ===")
    subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)

print("\nAll outputs regenerated successfully in outputs/.")
