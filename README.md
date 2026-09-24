# Vaccination strategies under leaky protection

Simulation and plotting code for the main manuscript and Supplementary Information. Data are supplied separately. This repository does not contain survey records, recipient-level outputs, simulation results or figure data.

## Installation

Use Python 3.12 and install the scientific dependencies:

```sh
python -m pip install -r requirements.txt
```

The tested environment uses Python 3.12.4, NumPy 1.26.4, pandas 2.2.3, SciPy 1.14.1, NetworkX 3.2.1 and Matplotlib 3.10.8. For the original ordering of tied values, retain the tested NumPy version and input row order.

## Input

The simulation input is the consolidated, household-standardised workbook `20231012.xlsx`, first worksheet. Pass its location with `--data`; it does not need to be copied into this repository. The original questionnaire field names are recognised through escaped Unicode strings, so all source files contain ASCII characters only. See [data/README.md](data/README.md).

## Simulations

Run commands from the repository root. Each command requires a new or empty output directory. The defaults retain 100 repetitions, 500 daily time steps and 10 initial infectious individuals.

```sh
python run.py primary --data /path/to/20231012.xlsx --output results/primary
python run.py all-or-nothing --data /path/to/20231012.xlsx --output results/all_or_nothing
python run.py hybrid-leaky --data /path/to/20231012.xlsx --output results/hybrid_leaky
python run.py hybrid-all-or-nothing --data /path/to/20231012.xlsx --output results/hybrid_all_or_nothing
python run.py disease --data /path/to/20231012.xlsx --output results/disease
python run.py network --data /path/to/20231012.xlsx --output results/network
```

Use quotes around paths containing spaces. `--workers` controls parallel processes. The full grids are computationally intensive. A short execution check is:

```sh
python run.py primary --data /path/to/20231012.xlsx --output results/check_primary --workers 1 --runs 2 --days 20 --r0 2 --ve 0.5 --coverage 0 0.5
python run.py hybrid-leaky --data /path/to/20231012.xlsx --output results/check_hybrid --workers 1 --runs 2 --days 20 --r0 1 --alpha 0 0.5 1
```

These short runs are execution checks, not replacements for manuscript results. Changing grid order, coverage levels, run counts or time horizons changes the simulation stream.

| Task | Manuscript use | Default settings |
|---|---|---|
| `primary` | Figures 3, 4, 5A, S5, S9, S10 | 17 R0 values, 5 vaccine efficacies, 20 coverage levels |
| `all-or-nothing` | Figures S1-S3 | Same grid as `primary`, with all-or-nothing protection and waning |
| `hybrid-leaky` | Figures 5B-D and S4 | R0 = 1, 3, 7, 18.6; efficacy = 0.8; 51 quota fractions |
| `hybrid-all-or-nothing` | Figure S4 | R0 = 1, 7, 18.6; efficacy = 0.8; 51 quota fractions |
| `disease` | Figures S6-S7 | COVID-like, influenza-like and RSV-like reference profiles; 3 efficacies |
| `network` | Figure S8 | 7 network types; R0 = 1, 2, 4, 8; 3 efficacies |

Disease profiles use their own reference R0 values (2.6, 1.28 and 1.7), progression and recovery probabilities, waning schedules and age-specific fatality profiles. These are defined in `simulations/disease_sensitivity.py`. Network variants retain the separately validated network-generation, tie-handling and seed rules of the sensitivity experiment.

The principal-grid driver simulates only random, high-exposure and oldest-first vaccination. It advances the legacy random-number stream past discarded experiment slots without simulating those strategies. This preserves the original numerical sequence for the retained strategies under the full default protocol.

## Figures

All panels are exported separately into `with_text`, `without_text` and `legends` subdirectories. Text-free panels contain no titles, axis text, annotations, panel letters, legends or colour bars. PNG and PDF are the defaults; add `--formats png pdf svg` for editable SVG output. Figures are not assembled into composite pages.

```sh
python plot.py 2 --input /path/to/20231012.xlsx --output results/fig2
python plot.py 3 --input results/primary --output results/fig3
python plot.py 4 --input results/primary --output results/fig4
python plot.py 5A --input results/primary --output results/fig5A
python plot.py 5BCD --input results/hybrid_leaky --output results/fig5BCD
python plot.py S1 --input results/all_or_nothing --output results/figS1
python plot.py S2 --input results/all_or_nothing --output results/figS2
python plot.py S3 --input results/all_or_nothing --output results/figS3
python plot.py S4 --input results/hybrid_leaky --other results/hybrid_all_or_nothing --output results/figS4
python plot.py S5 --input results/primary --output results/figS5
python plot.py S6 --input results/disease --output results/figS6
python plot.py S7 --input results/disease --output results/figS7
python plot.py S8 --input results/network --output results/figS8
python plot.py S9 --input results/primary --output results/figS9
python plot.py S10 --input results/primary --output results/figS10
```

Existing retained output directories can be used instead of newly generated results. Figure 2 reconstructs the graph and recipient sets directly from the supplied workbook. Its force-directed layout is not geographical. Figure 1 is a manually assembled conceptual schematic and has no numerical plotting pipeline.

### Figure input files

| Figure | Required files inside the input directory |
|---|---|
| 3, 4, 5A, S1-S3, S5, S9-S10 | `Simulation_Results_All_Scenarios_With_Uncertainty.csv.gz` |
| 5B | `Quota_Mixture_Tradeoff_AllPoint_Summary.csv` |
| 5C-D | `fig5_CD_endpoint_decomposition/Panel_C_FIP_Plot_Ready.csv` and `Panel_D_Peak_Plot_Ready.csv` |
| S4 | `Quota_Mixture_Tradeoff_AllReplicate_Outcomes.csv.gz` from each protection model |
| S6-S7 | `figure_source_data.csv` |
| S8 | `Network_Robustness_Additional_Reduction_Summary.xlsx`, sheet `Parameter_Level_Reduction` |

For summary-based plots, `--input` also accepts the summary CSV itself. S8 additionally accepts the retained parameter-level plotting CSV.

## Calculation conventions

- Exposure strength is internal weighted strength plus external exposure. External non-fixed exposure is external non-fixed activity duration multiplied by an effective contact count of 4. This count is not a contact rate. Internal community encounters retain the separate 0.25-hour edge contribution. This terminology correction does not change the original numerical calculations.
- Beta is calibrated from the internal weighted-network spectral radius after adjustment for internal and prevalence-weighted external exposure.
- Hybrid allocation keeps the shared endpoint recipients vaccinated. The high-exposure-only quota is `floor(alpha * remaining_places + 0.5)`; the oldest-first-only quota fills the remaining places. Each exclusive group follows its original priority ranking. The allocation is quota-based, not a weighted rank score.
- Figure 5A and S3 classify unrounded endpoint means. They do not compute correlations over alpha. Figure 5B displays representative hybrid trajectories.
- Additional reduction is `100 * (random_mean - targeted_mean) / no_vaccination_mean`, in percentage points. The main grid uses the random-strategy zero-coverage mean as the common baseline.
- Figure 5C divides differences in group infection counts by the total population. Figure 5D counts each group's infectious individuals at each strategy's own population peak in each repetition. Group contributions sum to the corresponding total difference. Groups are displayed from low to high exposure.
- S4 uses paired within-repetition changes from alpha = 0. Shading shows pointwise 95% Monte Carlo intervals. Outcomes have separate vertical axes; line crossings and relative slopes are not quantitative comparisons.
- S5 shows the mean and interquartile range across parameter settings. S8 averages over network realisations before plotting parameter-level values. Neither spread is a Monte Carlo confidence interval.
- S9-S10 use independent-sample standard errors for principal-grid strategy differences, treating the no-vaccination mean as fixed.

## Checks

```sh
python tests/check_release.py
python tests/check_release.py --data /path/to/20231012.xlsx
```

The data-dependent check reconstructs the graph, checks all 51 hybrid allocations and verifies group totals against simulated population totals. Source provenance is recorded in [docs/source_manifest.json](docs/source_manifest.json). Generated outputs are ignored by Git; review any additional files before publishing.
