# Leaky vaccine protection on heterogeneous exposure networks

This repository contains the Python code used to construct the survey-derived
weighted exposure network, simulate vaccination strategies under leaky vaccine
protection, perform robustness analyses, analyse node-level infection timing,
and generate the analytical figure components reported in the associated
manuscript.

## Repository scope

The repository provides the scientific simulation, analysis, and plotting
scripts used in the study. It does not contain the household-survey workbook,
because the records include potentially sensitive participant information. A
small synthetic smoke test is included to verify that the primary simulation
pipeline executes successfully without access to the survey data.

The synthetic test is an execution check only. It does not reproduce the
survey-derived network or any numerical result reported in the manuscript.

## Repository structure

```text
analysis/       Infection-timing predictor analysis
simulations/    Primary, disease-parameter, and network-structure experiments
figures/        Main and supplementary figure-component scripts
examples/       Synthetic smoke test and custom-network example generator
data/           Location for authorized survey input; real data are not included
results/        Generated outputs; excluded from version control
docs/           Workflow, release scope, validation, and code inventory
environments/   Workflow-specific package versions
```

## Data availability within the repository

The production workflows expect the authorized household-survey workbook at:

```text
data/20231012.xlsx
```

This workbook is not included. Requests for access to appropriately
de-identified data may be considered on a case-by-case basis by contacting
`xyw980217@163.com`, subject to applicable ethical, privacy, and data-protection
requirements.

The code retains Chinese survey labels only where they are required to identify
source-data columns or categorical responses. All code comments and
public-facing documentation are in English.

## Software environments

The simulation and infection-timing workflows used different recorded Python
environments. Install the workflow-specific environment rather than combining
both pinned requirement files.

Simulation workflow:

```bash
python -m pip install -r environments/requirements-simulation.txt
```

Infection-timing and machine-learning workflow:

```bash
python -m pip install -r environments/requirements-analysis.txt
```

The recorded simulation environment used Python 3.7.10. The infection-timing
workflow was reported under Python 3.12.4. See `environments/README.md` for the
package versions available from the study records.

## Quick execution check

From the repository root, run:

```bash
python examples/run_smoke_test.py
```

or equivalently:

```bash
python simulations/run_primary_vaccination_experiments.py --test-mode
```

The test constructs an in-memory synthetic questionnaire-shaped dataset with
60 nodes and runs a reduced parameter configuration through the production
primary-simulation code path. Expected outputs are written to:

```text
results/smoke_test/
```

A successful run ends with:

```text
[TEST MODE PASSED] Synthetic smoke-test outputs were generated successfully.
```

The test uses one baseline vaccine-efficacy value, one transmissibility value,
two coverage levels, two Monte Carlo repetitions, and a 30-day horizon. These
settings are deliberately small and must not be interpreted as scientific
results.

## Production workflow

Run all commands from the repository root.

### 1. Infection-timing analysis

```bash
python analysis/analyze_infection_timing_predictors.py
```

This script reconstructs the survey-derived exposure network, runs the
unvaccinated reference simulations, and produces the component plots and tables
used for the infection-timing analyses.

### 2. Primary vaccination-strategy experiment

```bash
python simulations/run_primary_vaccination_experiments.py
```

Outputs are written to:

```text
results/primary_simulation/
```

The full parameter sweep is computationally intensive and was executed in
parallel. Cluster-specific submission scripts are not included because
scheduler syntax and resource allocation are system dependent; the Python
entry point above contains the scientific workflow.

### 3. Main and coverage-dependent figure components

```bash
python figures/plot_strategy_comparison_heatmaps.py
python figures/plot_exposure_distribution_mechanism.py
python figures/plot_supplementary_coverage_dependence.py
python figures/plot_supplementary_monte_carlo_uncertainty.py
```

### 4. Disease-parameter robustness

```bash
python simulations/run_disease_parameter_robustness.py
python figures/plot_supplementary_disease_robustness_high_exposure.py
python figures/plot_supplementary_disease_robustness_oldest_first.py
```

### 5. Network-structure robustness

```bash
python simulations/run_network_structure_robustness.py
python figures/plot_supplementary_network_structure_robustness.py
```

Generated figures are written under:

```text
results/figures/
```

## Figure assembly

The repository provides the scripts used to generate analytical panels and
plotting components. Final multi-panel assembly, panel lettering, and minor
layout adjustments were performed manually in Microsoft PowerPoint. Therefore,
the final submission-ready composite figures are not claimed to be regenerated
by a single automated script.

The conceptual framework in Figure 1 was drawn manually and is not a
data-derived output. Its editable source file is not included.

## Custom weighted networks

The production scripts construct the reference network from the authorized
survey workbook. Researchers wishing to adapt the model to another weighted
network should read:

```text
docs/CUSTOM_NETWORK_GUIDE.md
```

A fully synthetic node-and-edge example can be generated with:

```bash
python examples/generate_custom_network_example.py
```

This example documents the expected node attributes and weighted edge fields.
It is an adaptation template, not a replacement for the survey-derived network
used in the manuscript.

## Reproducibility boundaries

- Real questionnaire records are not distributed in this repository.
- The smoke test verifies execution and output generation only.
- Exact manuscript results require the authorized survey workbook and the full
  parameter settings retained in the production scripts.
- Final figure composition includes documented manual layout work.
- Cluster launch scripts and the editable source of the conceptual figure are
  intentionally excluded.
- No software licence is assigned in this release. Reuse rights should be
  clarified after agreement among the authors and the relevant institution.

See `docs/RELEASE_SCOPE.md` and `docs/VALIDATION_REPORT.md` for further details.
