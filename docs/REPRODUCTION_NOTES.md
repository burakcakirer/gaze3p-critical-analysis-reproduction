# Reproduction and extension specification

## Provenance

- Assigned paper: Elfares et al., *Gaze3P*, arXiv:2507.00596v2, Sections 5 and 6.1–6.2.
- Paper-reported observations: `outputs/paper_reported_dataset_findings.csv` records **published** Table 1, Table 3, and Section 6.2 values. These were **not recalculated** using the original gaze dataset.
- Four mapping equations: implemented in `src/mappings.py` and plotted in `outputs/figure_mapping.png`.
- Synthetic randomized response: numerical results produced by `src/synthetic_ldp_experiment.py` and *not* the paper's personalized-DP benchmark.
- Sensitivity sweep: numerical results produced by `src/extended_sensitivity_experiment.py`, also synthetic.

## Base synthetic experiment (original repository)

- N users = 1,200; runs = 300; prevalence = 0.35; seed = 42.
- Seven-category synthetic distribution: `[0.25, 0.20, 0.18, 0.15, 0.10, 0.07, 0.05]`.
- Mechanism: heterogeneous binary randomized response.
- Utility: absolute error of a debiased binary-count estimator.
- Config: epsilon minimum 0.1; maximum 5.0; sigmoid steepness 1.5.

## Synthetic sensitivity sweep (new code)

- N users = 1,200, 300 runs per scenario; prevalence = 0.35; seed = 20261014.
- Three explicitly hypothetical rating distributions:
  - private-skewed: `[.25,.20,.18,.15,.10,.07,.05]`
  - uniform: `[1/7] * 7`
  - safe-skewed: `[.05,.07,.10,.15,.18,.20,.25]`
- Perturbation conditions: none or a probability 0.2 of attempting a plus/minus one rating change. Values are clipped at 1 and 7, so **the actual changed-category fraction is below 0.2**.
- Methods: static minimum epsilon, uniformly sampled random epsilon, linear, exponential, sequential, and sigmoid.
- Methods are evaluated on paired binary inputs and paired randomized-response uniform draws per run. Using identical uniforms makes output draws dependent across compared methods; across runs there are independent draws.
- Outputs: per-run absolute count error, mean/median assigned epsilon, a fraction where epsilon increased from its would-have-been value for an unperturbed assigned rating, and confidence-interval half-width computed as `1.96 * standard deviation / sqrt(300)`.
- This is an **epsilon-assignment diagnostic** rather than a measured formal DP violation. No gaze classifier or original gaze feature file is used.

## Regeneration

```bash
python -m pip install -r requirements.txt
python run_all.py
python -m unittest discover -s tests -v
```

The original study's reported learning accuracies cannot be regenerated from these synthetic records.
