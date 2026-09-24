"""Launch manuscript simulations with portable input and output paths."""
import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = {
    'primary': 'parameter_grid.py',
    'all-or-nothing': 'parameter_grid.py',
    'hybrid-leaky': 'hybrid_leaky.py',
    'hybrid-all-or-nothing': 'hybrid_all_or_nothing.py',
    'disease': 'disease_sensitivity.py',
    'network': 'network_sensitivity.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=TASKS)
    parser.add_argument('--data', type=Path, default=ROOT/'data'/'20231012.xlsx')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=min(20, max(1, (os.cpu_count() or 2)-1)))
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--days', type=int, default=500)
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--r0', type=float, nargs='+')
    parser.add_argument('--ve', type=float, nargs='+')
    parser.add_argument('--coverage', type=float, nargs='+')
    parser.add_argument('--alpha', type=float, nargs='+')
    args = parser.parse_args()
    if min(args.workers, args.runs, args.days, args.seeds) < 1:
        parser.error('workers, runs, days and seeds must be positive')
    for name in ['r0', 've', 'coverage', 'alpha']:
        values = getattr(args, name)
        if values is not None and (any(not math.isfinite(v) for v in values) or len(set(values)) != len(values)):
            parser.error(f'{name} values must be finite and distinct.')
    if args.r0 and any(v <= 0 for v in args.r0):
        parser.error('R0 values must be positive.')
    for name in ['ve', 'coverage', 'alpha']:
        if getattr(args, name) and any(v < 0 or v > 1 for v in getattr(args, name)):
            parser.error(f'{name} values must lie between 0 and 1.')
    if args.alpha and not args.task.startswith('hybrid'):
        parser.error('--alpha applies only to hybrid experiments.')
    if args.task.startswith('hybrid'):
        if args.coverage:
            parser.error('Hybrid allocation uses fixed 50% coverage.')
        if args.alpha and not {0.0, 1.0}.issubset(args.alpha):
            parser.error('Include alpha=0 and alpha=1 for endpoint decomposition.')
    if args.task == 'network' and args.coverage and sorted(args.coverage) != [0, .5]:
        parser.error('Network sensitivity uses coverage levels 0 and 0.5.')
    if args.task == 'disease' and args.coverage and (len(args.coverage) != 1 or not 0 < args.coverage[0] < 1):
        parser.error('Disease sensitivity requires one target coverage strictly between 0 and 1.')
    data = args.data.resolve()
    if not data.is_file():
        parser.error(f'Input workbook not found: {data}')
    output = (args.output or ROOT/'results'/args.task).resolve()
    if output == ROOT or output == data.parent or output in data.parents:
        parser.error('Choose a separate result directory.')
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose a new directory.')
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LEAKY_DATA=str(data), LEAKY_OUTPUT=str(output),
               LEAKY_WORKERS=str(args.workers), LEAKY_RUNS=str(args.runs),
               LEAKY_DAYS=str(args.days), LEAKY_SEEDS=str(args.seeds),
               PYTHONUTF8='1', MPLBACKEND='Agg')
    for name in ['r0','ve','coverage','alpha']:
        value = getattr(args, name)
        if value is not None:
            env['LEAKY_'+name.upper()] = ','.join(map(str, value))
    env['LEAKY_ENGINE'] = '_all_or_nothing' if args.task == 'all-or-nothing' else '_primary'
    command = [sys.executable, str(ROOT/'simulations'/TASKS[args.task])]
    if args.task == 'disease':
        command += ['--data-file', str(data), '--output-dir', str(output), '--workers', str(args.workers),
                    '--mc-runs', str(args.runs), '--simulation-days', str(args.days),
                    '--initial-infectious', str(args.seeds)]
        if args.r0:
            command += ['--r0-mode','common-grid','--r0-values',*map(str,args.r0)]
        if args.ve:
            command += ['--ves-values',*map(str,args.ve)]
        if args.coverage:
            command += ['--coverage',str(args.coverage[0])]
    subprocess.run(command, cwd=output, env=env, check=True)


if __name__ == '__main__':
    main()
