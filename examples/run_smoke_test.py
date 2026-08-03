"""Run the lightweight synthetic smoke test for the primary simulation."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SCRIPT = (
    REPOSITORY_ROOT
    / "simulations"
    / "run_primary_vaccination_experiments.py"
)


def main() -> None:
    command = [sys.executable, str(PRIMARY_SCRIPT), "--test-mode"]
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


if __name__ == "__main__":
    main()
