
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from mappings import all_mappings

EPS_MIN = 0.1
EPS_MAX = 5.0
L = 7
K = 1.5

rows = []
for level in range(1, L + 1):
    vals = all_mappings(level, EPS_MIN, EPS_MAX, L, K)
    rows.append({"privacy_level": level, **vals})

df = pd.DataFrame(rows)
out_csv = ROOT / "outputs" / "epsilon_mappings.csv"
df.to_csv(out_csv, index=False)

plt.figure(figsize=(8, 5))
for name in ["linear", "exponential", "sequential", "sigmoid"]:
    plt.plot(df["privacy_level"], df[name], marker="o", label=name.capitalize())
plt.xlabel("Perceived privacy level l (1 = very private, 7 = very safe)")
plt.ylabel("Mapped epsilon")
plt.title("Gaze3P privacy-level to epsilon mappings")
plt.xticks(range(1, L + 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "figure_mapping.png", dpi=200)
plt.close()

print(df.to_string(index=False))
