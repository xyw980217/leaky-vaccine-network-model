# Logic and reproducibility audit

## Checks completed

- All thirteen Python scripts pass a syntax-compilation check.
- Input and output paths were standardized to the repository layout.
- The execution dependencies between simulation, analysis, and plotting scripts
  were identified and documented.
- Chinese comments were removed or translated; Chinese strings required to read
  the original questionnaire schema were retained.
- Editing-oriented markers such as `NEW`, `FIXED`, `MODIFIED`, and similar
  drafting comments were removed.
- The main scientific constants and formulae were not deliberately changed.

## Items that require an author decision before archival release

### 1. Primary-worker random-number reset

`run_primary_vaccination_experiments.py` resets NumPy and Python random seeds to
42 at the beginning of every `(VE_S,0, R0)` worker. This creates reproducible
and potentially common random streams across parameter combinations rather than
fully independent streams between combinations. This is not necessarily wrong,
but it must match the intended Monte Carlo design and should be documented
explicitly. Do not change it after generating the reported results unless the
full experiment is rerun.

### 2. Preprocessing mappings are not identical across workflows

The network-structure robustness script recognizes additional questionnaire
category variants, including full-width inequality symbols and `1-3次`, that
are not present in the primary, disease-robustness, and infection-timing
scripts. It also contains an additional duration category. Confirm whether
these variants occur in the actual workbook. If they do, the reference-network
preprocessing may differ across workflows.

### 3. Household-field cleaning differs

The infection-timing script normalizes whitespace and full-width slash
characters in household fields. The primary and robustness scripts use a
simpler string-strip operation. Confirm that these procedures produce the same
household identifiers in the authorized workbook. A single shared preprocessing
function would reduce the risk of divergence, but refactoring should be done
only after reproducing the currently reported network statistics.

### 4. Spectral-radius fallback in the infection-timing script

If the eigenvalue calculation fails, the script retains the legacy fallback
spectral radius of 10.0. The cleaned version now prints a warning instead of
failing silently, but the fallback itself remains. For a final reproducibility
release, a fail-fast error or a justified fallback should be selected.

### 5. Exploratory ROC analysis

The infection-timing script contains an exploratory ROC classification block
that is not described in the current main text or Supplementary Information.
Either remove it from the manuscript-release branch or move it to a clearly
labelled exploratory script.

### 6. Duplicated network-construction code

Network construction and questionnaire preprocessing are repeated in several
scripts. This preserves the supplied workflows but increases maintenance risk.
A later refactor could extract shared functions into a common module, provided
that regression tests first confirm identical node attributes, edge weights,
spectral radius, and strategy rankings.

### 7. Lightweight test mode

A deterministic synthetic smoke test is now included for the primary
simulation. It checks network construction, strategy execution, result
aggregation, and output generation under a reduced serial configuration. It is
explicitly documented as an execution check rather than a reproduction of
manuscript estimates. The disease-parameter, network-structure, and
infection-timing workflows still require the authorized survey workbook for
meaningful execution.

### 8. Published subset versus simulated strategy set

The primary experiment evaluates additional exploratory and household-priority
strategies beyond the random, high-exposure, and oldest-first strategies used in
the current manuscript. These branches were retained to avoid deleting supplied
analysis logic. The README should make clear which outputs support the paper and
which are auxiliary.

### 9. Partial completion after worker failure

The primary workflow now checks the expected number of summary rows before
exporting the combined result table, so a failed parameter worker cannot be
silently accepted as a complete primary sweep. The disease-robustness workflow
still reports individual worker exceptions and continues collecting successful
tasks; its outputs should therefore be checked for parameter-grid completeness
before interpretation.

### 10. Distinct software environments

The recorded simulation environment and the machine-learning environment use
different Python and package versions. They should remain separate in the
archival repository; combining them into one pinned requirements file could
produce an environment that matches neither workflow.
