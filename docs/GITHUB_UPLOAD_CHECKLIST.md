# GitHub upload checklist

Before making the repository public:

- [ ] Confirm that `data/20231012.xlsx` and all other participant-level files
  are absent.
- [ ] Confirm that `results/` contains only `.gitkeep`.
- [ ] Run `python examples/run_smoke_test.py` from the repository root.
- [ ] Confirm that the smoke test ends with `[TEST MODE PASSED]`.
- [ ] Delete local smoke-test outputs before committing, or rely on the supplied
  `.gitignore` rule for `results/`.
- [ ] Review the author and institutional requirements for a software licence.
- [ ] Add a licence only after co-author and institutional agreement.
- [ ] Add `CITATION.cff` after the manuscript title, author order, repository
  URL, and archival DOI are final.
- [ ] Create a versioned GitHub release for the exact publication code.
- [ ] Archive that release in Zenodo after the public repository has been
  reviewed by the authors.

The editable source of Figure 1, PowerPoint figure-assembly files, real survey
data, and cluster-specific job-submission scripts are intentionally outside the
repository scope.
