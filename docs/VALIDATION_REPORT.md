# Validation report

## Static checks completed

- All 13 Python scripts in the reorganized repository pass syntax compilation.
- No Chinese-language comments or docstrings remain in the Python files.
- Chinese text remains only in survey-column names, categorical response
  mappings, and input-validation logic required to parse the authorized
  questionnaire workbook.
- No absolute personal-computer paths, email credentials, passwords, tokens, or
  API keys were detected in the code.
- No raw questionnaire records or manuscript simulation outputs are included.

## Synthetic smoke test

The smoke test was executed with:

```bash
python examples/run_smoke_test.py
```

The test completed successfully and ended with:

```text
[TEST MODE PASSED] Synthetic smoke-test outputs were generated successfully.
```

Validated smoke-test properties:

| Check | Result |
|---|---:|
| Synthetic nodes | 60 |
| Synthetic households | 20 |
| Baseline vaccine-efficacy values | 1 |
| Transmissibility values | 1 |
| Coverage levels | 2 |
| Strategies evaluated | 10 |
| Summary rows expected | 20 |
| Summary rows generated | 20 |
| Monte Carlo repetitions per setting | 2 |
| Simulation horizon | 30 days |

The test generated the expected summary, metadata, exact-bin, feature-summary,
and selected replicate-output files. Generated test results were removed from
the release package after validation because the `results/` directory is
intended for local outputs.

The smoke test verifies execution and output schemas only. It does not validate
agreement with manuscript estimates.

## Custom-network example

The example generator was executed with:

```bash
python examples/generate_custom_network_example.py --nodes 30
```

The generated node and edge tables passed the included schema checks:

| Check | Result |
|---|---:|
| Nodes | 30 |
| Weighted undirected edges | 88 |
| Mean internal weighted strength | 13.000 |

Generated example files were removed after validation and are excluded by
`.gitignore`.

## Parameter-preservation check

The production parameter settings remain unchanged unless `--test-mode` is
explicitly supplied. Test-mode overrides are isolated to the reduced synthetic
execution check.

The following production elements were not deliberately changed:

- epidemiological parameter values;
- vaccine waning and natural-immunity functions;
- survey-derived edge-weight construction;
- vaccination ranking rules;
- outcome definitions;
- Monte Carlo summary calculations;
- disease-like and network-structure scenario definitions.

An additional summary-row completeness check was added before the primary
simulation exports its combined result table.

## Checks that still require the authorized data and recorded environments

Before archival release of the exact publication version, the authors should
run the production workflows with the authorized survey workbook and compare:

1. node, household, and edge counts;
2. exposure-strength summaries and spectral radius;
3. scenario-row counts and output-table schemas;
4. selected numerical results against retained manuscript outputs;
5. analytical figure panels against the submitted figures.

The final composite figure layout also requires the documented manual
PowerPoint assembly step.
