# Software environments

The simulation and infection-timing workflows were run in different recorded
Python environments. They are therefore documented separately rather than
forced into one potentially inconsistent dependency lock file.

## Simulation environment

Recorded cluster runtime: Python 3.7.10.

```bash
python -m pip install -r environments/requirements-simulation.txt
```

## Infection-timing and machine-learning environment

Reported runtime: Python 3.12.4 with the package versions listed in
`requirements-analysis.txt`.

```bash
python -m pip install -r environments/requirements-analysis.txt
```

The exact versions of Matplotlib, Seaborn and OpenPyXL used in the latter
workflow still need to be recovered from the original analysis environment.
