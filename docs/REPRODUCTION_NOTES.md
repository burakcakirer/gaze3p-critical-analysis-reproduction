# Reproduction Notes

## Scope
The repository reproduces the privacy-level-to-epsilon mapping procedure described in Section 6.1 of **Gaze3P: Gaze-Based Prediction of User-Perceived Privacy** and evaluates the same mappings in a controlled synthetic setting.

The controlled experiment is **not** presented as a direct replication of Table 4 in the paper. It uses a different dataset, a heterogeneous randomized-response mechanism, a binary count query, and mean absolute count error as the utility metric.

## Paper-derived configuration

- Privacy levels: `L = 7`
- Minimum epsilon: `epsilon_min = 0.1`
- Maximum epsilon: `epsilon_max = 5.0`
- Sigmoid steepness: `k = 1.5`
- Mapping functions: Linear, Exponential, Sequential, Sigmoid

## Controlled synthetic experiment

- Simulated users per run: `1200`
- Number of independent runs: `300`
- Binary prevalence: `0.35`
- Random seed: `42`
- Privacy mechanism: heterogeneous binary randomized response
- Utility metric: absolute error between the true count and the debiased estimated count

The synthetic privacy-level distribution is explicitly defined in `src/synthetic_ldp_experiment.py`; it is not taken from the Gaze3P dataset.
