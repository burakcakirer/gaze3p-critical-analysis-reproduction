"""Parametric synthetic checks of the previously implemented Gaze3P epsilon mappings.

IMPORTANT: Generated records are SYNTHETIC; this is not an analysis of Gaze3P
participant records, nor a replication of the paper's personalized DP Table 4.

Scenarios vary a *specified* seven-level probability vector and independent
bounded +/-1 rating perturbations. All four source-paper mappings are used
unchanged. Results consist only of measured metrics, not interpretation.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from mappings import all_mappings

SEED = 20261014
N_USERS = 1200
N_RUNS = 300
PREVALENCE = 0.35
EPS_MIN, EPS_MAX, N_LEVELS, K = .1, 5., 7, 1.5
METHODS = ('static_min', 'random', 'linear', 'exponential', 'sequential', 'sigmoid')
SCENARIOS = {
    'private_skew': [0.25, 0.20, 0.18, 0.15, 0.10, 0.07, 0.05],
    'uniform':      [1 / 7] * 7,
    'safe_skew':    [0.05, 0.07, 0.10, 0.15, 0.18, 0.20, 0.25],
}
RATING_PERTURBATION = (0., .2)

# Preserve *the same* source formulas; precompute mapped values to avoid loops.
EPS_TABLE = {m: np.array([all_mappings(l, EPS_MIN, EPS_MAX, N_LEVELS, K)[m]
                         for l in range(1, N_LEVELS + 1)])
             for m in METHODS if m not in ('static_min', 'random')}


def debiased_count(reports, eps):
    """For heterogeneous binary randomized response, sum unbiased bit estimates."""
    p = 1. / (1. + np.exp(-eps))
    q = 1. - p
    return np.sum((reports - q) / (p - q))


def run_sensitivity():
    rng = np.random.default_rng(SEED)
    results = []
    for scenario, raw_probs in SCENARIOS.items():
        probs = np.asarray(raw_probs, dtype=float)
        assert len(probs) == N_LEVELS and np.isclose(probs.sum(), 1.0)
        for i in range(N_RUNS):
            true_levels = rng.choice(np.arange(1, 8), N_USERS, p=probs)
            true_bits = rng.binomial(1, PREVALENCE, N_USERS)
            true_count = int(np.sum(true_bits))
            # Paired reports' coin tosses across methods and perturbation levels.
            rr_uniforms = rng.random(N_USERS)
            randomized_eps = rng.uniform(EPS_MIN, EPS_MAX, N_USERS)
            flips = rng.choice([-1, 1], size=N_USERS)
            perturb_draws = rng.random(N_USERS)
            for perturbation in RATING_PERTURBATION:
                input_levels = np.clip(
                    true_levels + (perturb_draws < perturbation).astype(int) * flips,
                    1, N_LEVELS
                )
                observed_change = np.mean(input_levels != true_levels)
                for method in METHODS:
                    if method == 'static_min':
                        eps = np.full(N_USERS, EPS_MIN)
                        ideal_eps = eps
                    elif method == 'random':
                        eps = randomized_eps
                        ideal_eps = eps
                    else:
                        eps = EPS_TABLE[method][input_levels - 1]
                        ideal_eps = EPS_TABLE[method][true_levels - 1]
                    truth_prob = 1. / (1. + np.exp(-eps))
                    reports = np.where(rr_uniforms < truth_prob, true_bits, 1 - true_bits)
                    estimate = debiased_count(reports, eps)
                    # This metric is a budget increase relative to the *unperturbed
                    # assigned category*; it is not a formal privacy proof/violation.
                    eps_increase_rate = np.mean(eps > ideal_eps + 1e-12)
                    results.append(dict(
                        scenario=scenario, run=i, perturbation=perturbation,
                        method=method, true_count=true_count,
                        estimated_count=float(estimate),
                        absolute_error=float(abs(estimate - true_count)),
                        mean_epsilon=float(np.mean(eps)),
                        median_epsilon=float(np.median(eps)),
                        budget_increase_fraction=float(eps_increase_rate),
                        changed_rating_fraction=float(observed_change),
                    ))
    out_dir = ROOT / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(results)
    raw.to_csv(out_dir / 'extended_sensitivity_raw.csv', index=False, float_format='%.8f')
    summary = raw.groupby(['scenario', 'perturbation', 'method'], as_index=False).agg(
        n_runs=('absolute_error', 'size'),
        mean_absolute_error=('absolute_error', 'mean'),
        std_absolute_error=('absolute_error', 'std'),
        mean_epsilon=('mean_epsilon', 'mean'),
        median_epsilon=('median_epsilon', 'mean'),
        mean_budget_increase_fraction=('budget_increase_fraction', 'mean'),
        mean_changed_rating_fraction=('changed_rating_fraction', 'mean'),
    )
    summary['mae_ci95_half_width'] = 1.96 * summary['std_absolute_error'] / np.sqrt(summary['n_runs'])
    summary.to_csv(out_dir / 'extended_sensitivity_summary.csv', index=False, float_format='%.8f')
    # A faceted graph: one plot for zero perturbation, one for 20%, clearly labeled.
    for per in RATING_PERTURBATION:
        subset = summary.loc[summary.perturbation == per]
        pv = subset.pivot(index='scenario', columns='method', values='mean_absolute_error')
        pv = pv.loc[['private_skew', 'uniform', 'safe_skew'], list(METHODS)]
        ax = pv.plot(kind='bar', figsize=(10, 5.5), rot=0)
        ax.set_xlabel('Synthetic privacy-level distribution')
        ax.set_ylabel('Mean absolute count error over 300 runs')
        ax.set_title(f'Synthetic sensitivity check; +/-1 rating perturbation probability = {per:.0%}')
        ax.grid(axis='y', alpha=0.2)
        ax.legend(title='Epsilon mapping', ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f'extended_sensitivity_{int(per * 100)}pct.png', dpi=180)
        plt.close()
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    return summary


if __name__ == '__main__':
    run_sensitivity()
