# Code inventory and filename mapping

## Manuscript analysis scripts

| Original filename | Reorganized filename | Role |
|---|---|---|
| `weighted_network_edition_experiment_part_6.0.py` | `simulations/run_primary_vaccination_experiments.py` | Primary vaccination-strategy parameter sweep and retained source data; includes an isolated synthetic smoke-test mode |
| `weighted_network_disease_robustness_revised.py` | `simulations/run_disease_parameter_robustness.py` | COVID-like, influenza-like, and RSV-like robustness experiments |
| `weighted_network_structure_robustness_AUDITED.py` | `simulations/run_network_structure_robustness.py` | Survey-derived, perturbed, and idealized network robustness experiments |
| `weighted_network_edition_analysis_part_5.0.py` | `analysis/analyze_infection_timing_predictors.py` | Mean first infection time, random forest, SHAP, Pearson, Lasso, and threshold analyses |
| `Draw_Big_chart_shared_no_vaccine_baseline.py` | `figures/plot_strategy_comparison_heatmaps.py` | Main targeted-versus-random outcome heatmaps |
| `distribution_drawn_abs_revised_exact_bins_full_fig4.py` | `figures/plot_exposure_distribution_mechanism.py` | Exposure-bin mechanism analysis |
| `plot_figureS3.py` | `figures/plot_supplementary_coverage_dependence.py` | Coverage-dependence analysis |
| `plot_figureS4_disease_robustness_high_exposure.py` | `figures/plot_supplementary_disease_robustness_high_exposure.py` | Disease-parameter robustness for high-exposure targeting |
| `plot_figureS5_disease_robustness_oldest_first.py` | `figures/plot_supplementary_disease_robustness_oldest_first.py` | Disease-parameter robustness for oldest-first targeting |
| `plot_figureS6_network_structure_robustness_rename.py` | `figures/plot_supplementary_network_structure_robustness.py` | Network-structure robustness |
| `plot_figureS7_monte_carlo_uncertainty_separate_panels_v3.py` | `figures/plot_supplementary_monte_carlo_uncertainty.py` | Monte Carlo uncertainty component panels and complete uncertainty figure output |

## Reproducibility support scripts

| File | Role |
|---|---|
| `examples/run_smoke_test.py` | Launches the reduced synthetic execution test for the production primary simulation |
| `examples/generate_custom_network_example.py` | Generates and validates a synthetic node-and-edge example for custom-network adaptation |

## Naming principles

- Verbs identify executable actions: `run`, `analyze`, `plot`, and `generate`.
- Filenames describe the scientific purpose rather than historical version
  numbers such as `5.0`, `6.0`, `revised`, `audited`, or `v3`.
- Supplementary figure numbers are retained only where the mapping is stable and
  informative.
- Test and adaptation utilities are separated from manuscript analysis scripts
  under `examples/`.
