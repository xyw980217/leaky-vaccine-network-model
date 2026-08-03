# Computational workflow

## Execution overview

```text
Authorized survey workbook
        |
        +--> Infection-timing analysis
        |       +--> Figure 2 components
        |       +--> Supplementary benchmark components
        |
        +--> Primary vaccination experiment
        |       +--> Strategy heatmap components
        |       +--> Exposure-mechanism components
        |       +--> Coverage-dependence analysis
        |       +--> Monte Carlo uncertainty analyses
        |
        +--> Disease-parameter robustness experiment
        |       +--> Disease-robustness figures
        |
        +--> Network-structure robustness experiment
                +--> Network-robustness figure
```

## Lightweight smoke test

The production primary-simulation code can be exercised without the survey
workbook by running:

```bash
python examples/run_smoke_test.py
```

The test generates deterministic synthetic questionnaire-shaped records in
memory and uses a reduced parameter configuration. It checks code execution,
network construction, strategy evaluation, result aggregation, and output-file
generation. It is not a numerical reproduction test.

## Production execution order

1. Validate the authorized survey workbook and place it in `data/20231012.xlsx`.
2. Run the infection-timing analysis independently of the vaccination sweep.
3. Run the primary vaccination-strategy experiment.
4. Generate the strategy heatmaps, mechanism components, coverage-dependence
   outputs, and Monte Carlo uncertainty outputs.
5. Run the disease-parameter robustness experiment and its plotting scripts.
6. Run the network-structure robustness experiment and its plotting script.
7. Assemble final composite figures manually in Microsoft PowerPoint where the
   submitted layout includes manual panel lettering or spacing adjustments.

## Output directories

```text
results/smoke_test/
results/infection_timing_analysis/
results/primary_simulation/
results/disease_parameter_robustness/
results/network_structure_robustness/
results/figures/main_strategy_heatmaps/
results/figures/exposure_distribution_mechanism/
results/figures/supplementary/
```

## Primary simulation data retention

The primary experiment retains:

- scenario-level means and Monte Carlo uncertainty summaries;
- exact exposure-bin counts used for the mechanism analysis;
- selected replicate-level outcomes used for numerical-stability analyses;
- node attributes, exposure-bin definitions, and run-configuration metadata.

The random vaccination set is generated once for a given strategy and coverage
within each parameter-setting worker and is then held fixed across Monte Carlo
replicates in that setting.
