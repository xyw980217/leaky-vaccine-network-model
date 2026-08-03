# Reorganization and release notes

## Code organization

1. Removed Chinese-language comments and drafting-oriented comments.
2. Retained Chinese questionnaire labels only where required for input parsing.
3. Replaced historical filenames and version suffixes with descriptive names.
4. Grouped scripts into `simulations`, `analysis`, `figures`, and `examples`.
5. Standardized data and output paths under `data/` and `results/`.
6. Retained the supplied household-priority analytical branches rather than
   silently deleting code not emphasized in the current manuscript.

## Reproducibility additions

1. Added `--test-mode` to the primary vaccination experiment.
2. Added `examples/run_smoke_test.py` as a platform-independent test launcher.
3. Added deterministic in-memory synthetic questionnaire-shaped records for the
   smoke test; no participant data are included.
4. Added an output-row completeness check to the primary summary export.
5. Added a custom weighted-network schema, generator, and adaptation guide.
6. Replaced the previous missing-material checklist with a release-scope
   document that distinguishes required scientific code from intentional
   exclusions.
7. Documented manual PowerPoint assembly of final multi-panel figures.
8. Documented the exclusion of Figure 1 source files and cluster-specific job
   submission scripts.

## Deliberately not changed

- Production epidemiological parameter values.
- Vaccine waning and natural-immunity functions.
- Survey-derived exposure-network equations and edge-weight construction.
- Vaccination ranking rules.
- Outcome definitions and Monte Carlo summary calculations.
- Disease-like and network-structure scenario definitions.

The reduced smoke-test parameters apply only when `--test-mode` is explicitly
provided.
