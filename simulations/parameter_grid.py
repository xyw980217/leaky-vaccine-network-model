"""Run the three manuscript strategies for Figures 3-5A and S1-S3, S5, S9-S10."""
import importlib
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGIES = ['Random', 'High Strength (Node)', 'Priority: Oldest First']
METRICS = ['Infection Rate', 'Peak Size', 'Peak Time', 'Expected Deaths']
ENGINE = os.environ.get('LEAKY_ENGINE', '_primary')


def discard_original_stream(engine, slot, coverage):
    # Preserve the published RNG sequence without simulating omitted strategies.
    if slot == 6 and int(coverage * engine.N) > 0:
        np.random.shuffle(np.flatnonzero(engine.is_key_group))
        np.random.shuffle(np.flatnonzero(~engine.is_key_group))
    for _ in range(engine.MC_RUNS_PER_POINT):
        np.random.choice(engine.N, size=engine.NUM_SEEDS_PER_RUN, replace=False)
        remaining = 2 * engine.N * engine.SIM_DAYS_STRATEGY
        while remaining:
            size = min(remaining, 1_000_000)
            np.random.random(size)
            remaining -= size


def recipient_mask(engine, strategy, coverage):
    count = int(coverage * engine.N)
    mask = np.zeros(engine.N, dtype=bool)
    if not count:
        return mask
    if strategy == STRATEGIES[0]:
        order = np.random.choice(engine.N, count, replace=False)
    elif strategy == STRATEGIES[1]:
        order = np.argsort(engine.total_node_strength)[::-1][:count]
    else:
        order = np.argsort(engine.df['age_num'].to_numpy())[::-1][:count]
    mask[order] = True
    return mask


def run_setting(ve, r0):
    engine = importlib.import_module(ENGINE)
    np.random.seed(engine.RNG_SEED)
    random.seed(engine.RNG_SEED)
    internal = engine.strength_internal.sum()
    external = (engine.strength_vir_work[engine.is_key_group].sum() * engine.EXTERNAL_PREVALENCE
                + engine.strength_vir_social[engine.daily_hours_nonfixed > 0].sum() * engine.EXTERNAL_PREVALENCE)
    fraction = internal / (internal + external + 1e-9)
    beta = engine.calibrate_beta(r0 * fraction, engine.PROB_I_TO_S, engine.G)
    summaries, replicates = [], []
    for coverage in engine.COVERAGE_STEPS:
        for slot in range(10):
            strategy = {0: STRATEGIES[0], 1: STRATEGIES[1], 4: STRATEGIES[2]}.get(slot)
            if strategy is None:
                discard_original_stream(engine, slot, coverage)
                continue
            mask = recipient_mask(engine, strategy, coverage)
            outcomes = []
            for replicate in range(1, engine.MC_RUNS_PER_POINT + 1):
                if ENGINE == '_all_or_nothing':
                    _, uniforms, _ = engine.get_aon_protected_mask(mask, ve, r0, coverage, replicate)
                    result = engine.run_strategy_simulation(beta, mask, ve, uniforms)
                else:
                    result = engine.run_strategy_simulation(beta, mask, ve)
                values = [result[0] / engine.N, result[1], result[2], result[3]]
                outcomes.append(values)
                replicates.append(dict(VES_0=ve, R0=r0, Coverage=coverage, Strategy=strategy,
                                       MC_Run=replicate, **dict(zip(METRICS, values))))
            outcome = np.asarray(outcomes, dtype=float)
            row = dict(VES_0=ve, R0=r0, Coverage=coverage, Strategy=strategy)
            row.update({'Actual Coverage': mask.mean(), 'Actual Vaccinated Count': int(mask.sum()),
                        'Calibrated Beta': beta, 'Internal Exposure Ratio': fraction})
            for i, metric in enumerate(METRICS):
                mean = outcome[:, i].mean()
                sd = outcome[:, i].std(ddof=1) if len(outcome) > 1 else np.nan
                se = sd / np.sqrt(len(outcome))
                row.update({metric: mean, metric+' SD': sd, metric+' SE': se,
                            metric+' CI95 Lower': mean-1.96*se, metric+' CI95 Upper': mean+1.96*se})
            summaries.append(row)
        print(f'Completed VE={ve:g}, R0={r0:g}, coverage={coverage:g}', flush=True)
    destination = Path(engine.OUTPUT_ROOT)
    folder = destination / 'scenario_summaries'
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(folder/f'VE{ve:g}_R0{r0:g}.csv', index=False)
    raw = destination / 'replicate_outcomes'
    raw.mkdir(exist_ok=True)
    pd.DataFrame(replicates).to_csv(raw/f'VE{ve:g}_R0{r0:g}.csv.gz', index=False)
    return summaries


def main():
    engine = importlib.import_module(ENGINE)
    output = Path(engine.OUTPUT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    with ProcessPoolExecutor(max_workers=engine.MAX_CORES) as pool:
        tasks = [pool.submit(run_setting, ve, r0) for ve in engine.VES_0_LIST for r0 in engine.R0_LIST]
        for task in as_completed(tasks):
            summaries.extend(task.result())
    frame = pd.DataFrame(summaries).sort_values(['VES_0','R0','Coverage','Strategy'])
    frame.to_csv(output/'Simulation_Results_All_Scenarios_With_Uncertainty.csv.gz', index=False)
    print(f'Saved {len(frame)} summary rows to {output}')


if __name__ == '__main__':
    main()
