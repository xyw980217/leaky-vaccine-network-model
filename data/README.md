# Data directory

The original household-survey workbook is not included because it contains
potentially sensitive participant information. The production simulation and
analysis scripts expect the authorized workbook at:

```text
data/20231012.xlsx
```

The workbook must retain the survey column names used by the scripts. Do not
commit the original workbook, direct identifiers, household addresses, or any
other potentially identifying variables to a public repository.

Requests for access to appropriately de-identified data may be considered on a
case-by-case basis by contacting `xyw980217@163.com`, subject to applicable
ethical, privacy, and data-protection requirements.

No real or de-identified participant-level dataset is required for the included
smoke test. Run:

```bash
python examples/run_smoke_test.py
```

The test creates synthetic questionnaire-shaped records in memory and does not
write or expose participant data.
