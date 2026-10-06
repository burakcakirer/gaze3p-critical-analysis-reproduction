# Critical Analysis and Reproduction of Gaze3P

**Student:** Burak Bekir Çakırer  
**Student ID:** 2211011008  
**Course:** Data Privacy - Academic Paper Analysis  
**Submission date:** _left blank in the report as requested_

This repository contains the academic report and reproducible code for the assigned paper:

> Mayar Elfares, Pascal Reisert, Ralf Küsters, and Andreas Bulling, **"Gaze3P: Gaze-Based Prediction of User-Perceived Privacy,"** arXiv:2507.00596v2, 2025.

## Repository structure

```text
gaze3p-critical-analysis-reproduction/
├── README.md
├── CITATION.cff
├── requirements.txt
├── run_all.py
├── .gitignore
├── report/
│   ├── Gaze3P_Critical_Analysis_Report_Burak_Bekir_Cakirer.docx
│   └── Gaze3P_Critical_Analysis_Report_Burak_Bekir_Cakirer.pdf
├── src/
│   ├── mappings.py
│   ├── reproduce_figure7.py
│   └── synthetic_ldp_experiment.py
├── outputs/
│   ├── epsilon_mappings.csv
│   ├── figure_mapping.png
│   ├── synthetic_ldp_raw.csv
│   ├── synthetic_ldp_summary.csv
│   └── synthetic_ldp_utility.png
├── docs/
│   ├── AI_USAGE_DISCLOSURE.md
│   └── REPRODUCTION_NOTES.md
└── paper/
    └── README.md
```

## Reproduction overview

The repository has two reproducible components:

1. **Paper-derived mapping reproduction**  
   Implements the four perceived-privacy-to-epsilon mappings described in Section 6.1 of Gaze3P:
   - Linear
   - Exponential
   - Sequential
   - Sigmoid

   The scripts use the paper's Figure 7 example configuration:
   `epsilon_min = 0.1`, `epsilon_max = 5`, `L = 7`, and `k = 1.5`.

2. **Controlled extension in a new setting**  
   Applies the four mappings to a synthetic heterogeneous randomized-response experiment. This experiment is intentionally described as an extension rather than a replication of the paper's Table 4.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run everything

```bash
python run_all.py
```

Or run the experiments separately:

```bash
python src/reproduce_figure7.py
python src/synthetic_ldp_experiment.py
```

All regenerated files are written to `outputs/`.

## Main outputs

- `outputs/epsilon_mappings.csv` - reproduced privacy-level-to-epsilon values
- `outputs/figure_mapping.png` - mapping curves
- `outputs/synthetic_ldp_summary.csv` - aggregate controlled-experiment results
- `outputs/synthetic_ldp_raw.csv` - per-run experimental results
- `outputs/synthetic_ldp_utility.png` - utility comparison plot

## Report

The final academic report is provided in both Word and PDF formats under `report/`. The submission date field is intentionally blank.

## Generative AI disclosure

The disclosure required by the assignment is included both in the report appendix and as `docs/AI_USAGE_DISCLOSURE.md`.

## Important methodological note

The synthetic randomized-response experiment is **not a direct reproduction of the paper's Table 4**. The repository reproduces the mapping method and then evaluates it in a transparent, explicitly defined new setting. See `docs/REPRODUCTION_NOTES.md` for details.
