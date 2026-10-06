
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from mappings import linear_epsilon, exponential_epsilon, sequential_epsilon, sigmoid_epsilon

SEED = 42
N_USERS = 1200
N_RUNS = 300
PREVALENCE = 0.35
EPS_MIN = 0.1
EPS_MAX = 5.0
L = 7
K = 1.5

# This distribution is a controlled synthetic setting, NOT taken from Gaze3P.
# It is intentionally skewed toward the private side to stress-test the mappings.
LEVEL_PROBS = np.array([0.25, 0.20, 0.18, 0.15, 0.10, 0.07, 0.05], dtype=float)
LEVEL_PROBS = LEVEL_PROBS / LEVEL_PROBS.sum()

rng = np.random.default_rng(SEED)

def rr_report(bit, epsilon, rng):
    """Binary epsilon-local-DP randomized response."""
    p_truth = np.exp(epsilon) / (1.0 + np.exp(epsilon))
    if rng.random() < p_truth:
        return bit
    return 1 - bit

def debias_count(reports, epsilons):
    """Unbiased estimator for the count under heterogeneous randomized response."""
    reports = np.asarray(reports, dtype=float)
    epsilons = np.asarray(epsilons, dtype=float)
    p = np.exp(epsilons) / (1.0 + np.exp(epsilons))
    q = 1.0 - p
    denom = p - q

    # Per-user unbiased estimate of the original bit:
    # E[y] = q + (p-q)x  =>  x_hat = (y-q)/(p-q)
    xhat = (reports - q) / denom
    return float(xhat.sum())

def map_eps(levels, method):
    if method == "linear":
        f = linear_epsilon
    elif method == "exponential":
        f = exponential_epsilon
    elif method == "sequential":
        f = sequential_epsilon
    elif method == "sigmoid":
        return np.array([sigmoid_epsilon(int(l), EPS_MIN, EPS_MAX, L, K) for l in levels])
    elif method == "static_min":
        return np.full(len(levels), EPS_MIN, dtype=float)
    elif method == "random":
        return rng.uniform(EPS_MIN, EPS_MAX, size=len(levels))
    else:
        raise ValueError(method)
    return np.array([f(int(l), EPS_MIN, EPS_MAX, L) for l in levels])

methods = ["static_min", "random", "linear", "exponential", "sequential", "sigmoid"]
records = []

for run in range(N_RUNS):
    levels = rng.choice(np.arange(1, L + 1), size=N_USERS, p=LEVEL_PROBS)
    private_bits = rng.binomial(1, PREVALENCE, size=N_USERS)
    true_count = private_bits.sum()

    for method in methods:
        eps = map_eps(levels, method)
        reports = np.array([rr_report(int(b), float(e), rng) for b, e in zip(private_bits, eps)])
        estimated_count = debias_count(reports, eps)
        abs_error = abs(estimated_count - true_count)

        records.append({
            "run": run,
            "method": method,
            "true_count": int(true_count),
            "estimated_count": estimated_count,
            "absolute_error": abs_error,
            "mean_epsilon": float(np.mean(eps)),
            "median_epsilon": float(np.median(eps)),
        })

raw = pd.DataFrame(records)
raw.to_csv(ROOT / "outputs" / "synthetic_ldp_raw.csv", index=False)

summary = raw.groupby("method", as_index=False).agg(
    mean_absolute_error=("absolute_error", "mean"),
    std_absolute_error=("absolute_error", "std"),
    mean_epsilon=("mean_epsilon", "mean"),
    median_epsilon=("median_epsilon", "mean"),
)
summary.to_csv(ROOT / "outputs" / "synthetic_ldp_summary.csv", index=False)

plot_df = summary.sort_values("mean_absolute_error")
plt.figure(figsize=(8, 5))
plt.bar(plot_df["method"], plot_df["mean_absolute_error"])
plt.ylabel("Mean absolute error of estimated count")
plt.xlabel("Method")
plt.title("Controlled extension: heterogeneous randomized response")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "synthetic_ldp_utility.png", dpi=200)
plt.close()

print(summary.to_string(index=False))
