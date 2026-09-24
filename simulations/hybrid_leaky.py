"""Quota-based leaky allocation experiments for Figure 5B-D and Figure S4."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import matplotlib
matplotlib.use('Agg')
import math
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.stats import rankdata
current_path = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = '20231012.xlsx'
DATA_PATH = os.environ.get('LEAKY_DATA', str(Path(__file__).resolve().parents[1] / 'data' / '20231012.xlsx'))
SHEET_ID = 0
OUTPUT_ROOT = os.environ.get('LEAKY_OUTPUT', str(Path(__file__).resolve().parents[1] / 'results' / 'hybrid_leaky'))
POINT_RAW_DIR = os.path.join(OUTPUT_ROOT, 'raw_point_data')
POINT_REPLICATE_DIR = os.path.join(OUTPUT_ROOT, 'replicate_outcomes')
POINT_SUMMARY_DIR = os.path.join(OUTPUT_ROOT, 'point_summaries')
METADATA_DIR = os.path.join(OUTPUT_ROOT, 'metadata')
FIGURE_DIR = os.path.join(OUTPUT_ROOT, 'preliminary_figures')
FIG5_CD_DIR = os.path.join(OUTPUT_ROOT, 'fig5_CD_endpoint_decomposition')
for _dir in [OUTPUT_ROOT, POINT_RAW_DIR, POINT_REPLICATE_DIR, POINT_SUMMARY_DIR, METADATA_DIR, FIGURE_DIR, FIG5_CD_DIR]:
    os.makedirs(_dir, exist_ok=True)
MAX_CORES = int(os.environ.get('LEAKY_WORKERS', min(20, max(1, (os.cpu_count() or 2) - 1))))
PROB_E_TO_I = 0.2
PROB_I_TO_S = 0.142
NAT_EFF_MAX = 0.652
NAT_EFF_MIN = 0.247
HH_CONTACT_HOURS = 12.0
SOCIAL_ENCOUNTER_DURATION = 0.25
INTERNAL_SOCIAL_RATIO = 0.3
VIRTUAL_CONTACT_MAPPING = {'a': 3.0, 'b': 10.0, 'c': 20.0, 'd': 30.0}
EXTERNAL_PREVALENCE = 0.005
# Effective contact count, not contacts per hour.
EXTERNAL_SOCIAL_CONTACT_COUNT = 4.0
NUM_SEEDS_PER_RUN = int(os.environ.get('LEAKY_SEEDS', 10))
RNG_SEED = 42
MC_RUNS_PER_POINT = int(os.environ.get('LEAKY_RUNS', 100))
SIM_DAYS_STRATEGY = int(os.environ.get('LEAKY_DAYS', 500))
COVERAGE = 0.5
R0_LIST = [float(x) for x in os.environ['LEAKY_R0'].split(',')] if 'LEAKY_R0' in os.environ else [1.0, 3.0, 7.0, 18.6]
VES_0_LIST = [float(x) for x in os.environ['LEAKY_VE'].split(',')] if 'LEAKY_VE' in os.environ else [0.8]
HYBRID_ALPHA_LIST = [float(x) for x in os.environ['LEAKY_ALPHA'].split(',')] if 'LEAKY_ALPHA' in os.environ else np.round(np.linspace(0.0, 1.0, 51), 2).tolist()
FIG5_CD_R0_LIST = [r for r in [1.0, 3.0, 18.6] if r in R0_LIST]
SAVE_RAW_POINT_DATA = True
MAKE_PRELIMINARY_FIGURE = False
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)
IFR_MAP = {15: 6.95e-05, 24: 0.000309, 34.5: 0.000844, 44.5: 0.00161, 54.5: 0.00595, 64.5: 0.0193, 75: 0.0428}

def _normalize_col(s):
    return str(s).replace('\ufeff', '').strip().replace(' ', '').replace('\uff08', '(').replace('\uff09', ')').replace('/', '')

def pick_column(df, targets):
    norm_map = {col: _normalize_col(col) for col in df.columns}
    wanted_norm = [_normalize_col(t) for t in targets]
    for col, ncol in norm_map.items():
        if ncol in wanted_norm:
            return col
    for col, ncol in norm_map.items():
        if any((w in ncol for w in wanted_norm)) or any((ncol in w for w in wanted_norm)):
            return col
    raise KeyError(f'Column not found: {targets}')
FREQ_MAP = {'\u65e0': 0, '<1\u6b21': 0.5, '1-2\u6b21': 1.5, '3-4\u6b21': 3.5, '\u22655\u6b21': 5.5, 'a': 5.5, 'b': 3.5, 'c': 1.5, 'd': 0.5}
DUR_MAP = {'\u65e0': 0, '2\u5c0f\u65f6\u4ee5\u5185': 1, '2-5\u5c0f\u65f6': 3.5, '5\u5c0f\u65f6\u4ee5\u4e0a': 6, 'a': 1, 'b': 3.5, 'c': 6}

def _map_freq(x):
    s = str(x).strip()
    return FREQ_MAP.get(s) if s in FREQ_MAP else 5.5 if '5\u6b21' in s else 0.5

def _map_dur(x):
    s = str(x).strip()
    return DUR_MAP.get(s) if s in DUR_MAP else 3.5 if '2-5' in s else 0.0

def _infer_employed(s):
    s = str(s).strip()
    return s in ['b', 'B', 'd', 'D'] or '\u5de5' in s or '\u73ed' in s

def _infer_student(s):
    s = str(s).strip()
    return '\u5b66' in s or s in ['a', 'A']

def _map_contact_count(x):
    s = str(x).lower().strip()
    for k, v in VIRTUAL_CONTACT_MAPPING.items():
        if k in s:
            return v
    return 0.0

def calculate_expected_deaths(infected_indices, age_series):
    deaths = 0.0
    ages = age_series.iloc[infected_indices].values
    available_ages = np.array(list(IFR_MAP.keys()))
    for age in ages:
        idx = np.abs(available_ages - age).argmin()
        closest_age = available_ages[idx]
        deaths += IFR_MAP[closest_age]
    return deaths

def summarize_values(values, prefix):
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return {f'{prefix} Mean': np.nan, f'{prefix} SD': np.nan, f'{prefix} SE': np.nan, f'{prefix} CI95 Lower': np.nan, f'{prefix} CI95 Upper': np.nan}
    mean_val = float(np.mean(arr))
    if n > 1:
        sd_val = float(np.std(arr, ddof=1))
        se_val = sd_val / math.sqrt(n)
        ci_low = mean_val - 1.96 * se_val
        ci_high = mean_val + 1.96 * se_val
    else:
        sd_val = np.nan
        se_val = np.nan
        ci_low = np.nan
        ci_high = np.nan
    return {f'{prefix} Mean': mean_val, f'{prefix} SD': sd_val, f'{prefix} SE': se_val, f'{prefix} CI95 Lower': ci_low, f'{prefix} CI95 Upper': ci_high}
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f'Data file not found: {DATA_PATH} Use --data to select the input workbook.')
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_ID)
N = len(df)
print(f'[INFO] Loaded N = {N}', flush=True)
print(f'[INFO] Data file = {DATA_PATH}', flush=True)
COL_COMMUNITY = pick_column(df, ['\u5c0f\u533a\u540d\u79f0', '\u5c0f\u533a', '\u793e\u533a\u540d\u79f0', '\u793e\u533a'])
COL_ADDRESS = pick_column(df, ['\u4f4f\u6237\u8be6\u7ec6\u5730\u5740', '\u4f4f\u5740'])
COL_HH_INDEX = pick_column(df, ['\u672c\u6237\u7b2c\u4efd'])
COL_WORK_STATUS = pick_column(df, ['\u5de5\u4f5c\u72b6\u6001'])
COL_CONTACT_COUNT = pick_column(df, ['\u4eba\u5458\u6570\u91cf', '\u5b66\u4e60\u573a\u6240\u7684\u4eba\u5458\u6570\u91cf'])
COL_TRIP_FREQ = [f'\u9891\u7387{i}' for i in range(1, 6)]
COL_TRIP_DUR = [f'\u65f6\u957f{i}' for i in range(1, 6)]
COL_AGE = pick_column(df, ['\u5e74\u9f84'])
df['_community_clean'] = df[COL_COMMUNITY].astype(str).str.strip()
df['_address_clean'] = df[COL_ADDRESS].astype(str).str.strip()
df['_hh_index_clean'] = df[COL_HH_INDEX].astype(str).str.strip()
df['_household_key'] = df['_community_clean'] + '|' + df['_address_clean']
df['_household_member_key'] = df['_household_key'] + '|' + df['_hh_index_clean']
household_sizes = df.groupby('_household_key').size()
duplicated_member_keys = df['_household_member_key'].duplicated(keep=False)
print(f'[HOUSEHOLD CHECK] Unique households by community + address: {df['_household_key'].nunique()}', flush=True)
print(f'[HOUSEHOLD CHECK] Unique household-member keys after adding household-member index: {df['_household_member_key'].nunique()}', flush=True)
print(f'[HOUSEHOLD CHECK] Households with >=2 respondents: {(household_sizes >= 2).sum()}', flush=True)
if duplicated_member_keys.any():
    print(f'[HOUSEHOLD WARNING] {duplicated_member_keys.sum()} rows have duplicated community + household + member-index keys.', flush=True)
age_map_dict = {'a': 15, 'b': 24, 'c': 34.5, 'd': 44.5, 'e': 54.5, 'f': 64.5, 'g': 75}
df['age_num'] = df[COL_AGE].astype(str).str.strip().map(age_map_dict).fillna(45.0)
daily_hours_nonfixed = np.zeros(N)
for k in range(5):
    if COL_TRIP_FREQ[k] in df.columns:
        fk = df[COL_TRIP_FREQ[k]].map(_map_freq).fillna(0).values
        dk = df[COL_TRIP_DUR[k]].map(_map_dur).fillna(0).values
        daily_hours_nonfixed += fk * dk / 7.0
daily_hours_fixed = np.zeros(N)
is_key_group = np.zeros(N, dtype=bool)
status_arr = df[COL_WORK_STATUS].values
for i in range(N):
    if _infer_employed(status_arr[i]) or _infer_student(status_arr[i]):
        daily_hours_fixed[i] = 8.0
        is_key_group[i] = True
mobility_index = np.clip(daily_hours_fixed + daily_hours_nonfixed, 0, 12.0)
virtual_contact_counts = df[COL_CONTACT_COUNT].map(_map_contact_count).fillna(0).values
G = nx.Graph()
G.add_nodes_from(range(N))
house2members = defaultdict(list)
for i in range(N):
    house2members[df.iloc[i]['_household_key']].append(i)
for members in house2members.values():
    m = len(members)
    if m <= 1:
        continue
    w = HH_CONTACT_HOURS / (m - 1)
    for a in range(m):
        for b in range(a + 1, m):
            G.add_edge(members[a], members[b], weight=w, layer='home')
nodes_indices = np.arange(N)
for i in range(N):
    internal_social_time = daily_hours_nonfixed[i] * INTERNAL_SOCIAL_RATIO
    if internal_social_time > 0:
        n_encounters = int(round(internal_social_time / SOCIAL_ENCOUNTER_DURATION))
        if n_encounters > 0:
            tgts = np.random.choice(nodes_indices[nodes_indices != i], size=min(n_encounters, N - 1), replace=False)
            w = SOCIAL_ENCOUNTER_DURATION
            for t in tgts:
                if G.has_edge(i, t):
                    G[i][t]['weight'] += w
                else:
                    G.add_edge(i, t, weight=w, layer='community_random')
strength_internal = np.array([G.degree(n, weight='weight') for n in G.nodes()])
strength_vir_work = virtual_contact_counts * daily_hours_fixed
external_social_time = daily_hours_nonfixed * (1.0 - INTERNAL_SOCIAL_RATIO)
strength_vir_social = external_social_time * EXTERNAL_SOCIAL_CONTACT_COUNT
strength_virtual = strength_vir_work + strength_vir_social
total_node_strength = strength_internal + strength_virtual
avg_neighbor_strength = np.zeros(N)
for i in range(N):
    neighbors = list(G.neighbors(i))
    if len(neighbors) > 0:
        sum_s = sum((total_node_strength[nbr] for nbr in neighbors))
        avg_neighbor_strength[i] = sum_s / len(neighbors)
print(f'[NETWORK] Built. Avg Total Strength: {total_node_strength.mean():.2f}', flush=True)

def build_sparse_adj_matrix(graph):
    if hasattr(nx, 'to_scipy_sparse_array'):
        return nx.to_scipy_sparse_array(graph, nodelist=range(N), weight='weight', format='csr', dtype=float)
    else:
        return nx.to_scipy_sparse_matrix(graph, nodelist=range(N), weight='weight', format='csr', dtype=float)
adj_matrix = build_sparse_adj_matrix(G)

def calibrate_beta(target_r0, gamma, graph):
    adj_mat = build_sparse_adj_matrix(graph)
    try:
        eigenvalues, _ = eigsh(adj_mat, k=1, which='LA')
        sr = float(eigenvalues[0])
        if sr <= 0:
            raise ValueError(f'Non-positive spectral radius: {sr}')
    except Exception as e:
        print(f'[WARNING] Eigenvalue calculation failed: {e}', flush=True)
        sr = 10.0
    return target_r0 * gamma * 24.0 / sr

def calibrate_scenario(target_r0):
    sum_internal = np.sum(strength_internal)
    work_exposure = np.sum(strength_vir_work[is_key_group]) * EXTERNAL_PREVALENCE
    social_exposure = np.sum(strength_vir_social[daily_hours_nonfixed > 0]) * EXTERNAL_PREVALENCE
    sum_external = work_exposure + social_exposure
    internal_ratio = sum_internal / (sum_internal + sum_external + 1e-09)
    adjusted_target_r0 = target_r0 * internal_ratio
    current_beta = calibrate_beta(adjusted_target_r0, PROB_I_TO_S, G)
    return (current_beta, internal_ratio, adjusted_target_r0)
AGE_VALUES = df['age_num'].values.astype(float)
EXPOSURE_VALUES = total_node_strength.astype(float)
if N > 1:
    AGE_PERCENTILE = (rankdata(AGE_VALUES, method='average') - 1.0) / (N - 1.0)
    EXPOSURE_PERCENTILE = (rankdata(EXPOSURE_VALUES, method='average') - 1.0) / (N - 1.0)
else:
    AGE_PERCENTILE = np.zeros(N)
    EXPOSURE_PERCENTILE = np.zeros(N)

def exact_population_group_ids(values, number_of_groups):
    values = np.asarray(values, dtype=float)
    node_ids = np.arange(len(values), dtype=int)
    order = np.lexsort((node_ids, values))
    group_ids = np.full(len(values), -1, dtype=np.int16)
    for group_id, members in enumerate(np.array_split(order, number_of_groups)):
        group_ids[members] = group_id
    if np.any(group_ids < 0):
        raise RuntimeError('Equal-population grouping failed')
    return group_ids
TOTAL_STRENGTH_DECILE_IDS = exact_population_group_ids(EXPOSURE_VALUES, 10)
TOTAL_STRENGTH_DECILE_COUNTS = np.bincount(TOTAL_STRENGTH_DECILE_IDS, minlength=10).astype(np.int32)

def strength_decile_counts(values):
    return np.bincount(TOTAL_STRENGTH_DECILE_IDS, weights=np.asarray(values, dtype=float), minlength=10)[:10]

def get_quota_mixture_allocation(alpha, coverage):
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError('alpha must be between 0 and 1 inclusive')
    num_vax = int(coverage * N)
    if num_vax == 0:
        empty_mask = np.zeros(N, dtype=bool)
        empty_source = np.full(N, 'not_vaccinated', dtype='<U30')
        diagnostics = {'Shared Endpoint Count': 0, 'Endpoint Discordant Slots': 0, 'High Exposure Exclusive Quota': 0, 'Oldest First Exclusive Quota': 0}
        return (empty_mask, empty_source, diagnostics)
    oldest_order = np.argsort(AGE_VALUES)[::-1]
    high_exposure_order = np.argsort(EXPOSURE_VALUES)[::-1]
    oldest_endpoint = oldest_order[:num_vax]
    high_exposure_endpoint = high_exposure_order[:num_vax]
    oldest_endpoint_mask = np.zeros(N, dtype=bool)
    oldest_endpoint_mask[oldest_endpoint] = True
    high_exposure_endpoint_mask = np.zeros(N, dtype=bool)
    high_exposure_endpoint_mask[high_exposure_endpoint] = True
    shared_mask = oldest_endpoint_mask & high_exposure_endpoint_mask
    shared_indices = np.where(shared_mask)[0]
    oldest_only_order = oldest_order[oldest_endpoint_mask[oldest_order] & ~high_exposure_endpoint_mask[oldest_order]]
    high_exposure_only_order = high_exposure_order[high_exposure_endpoint_mask[high_exposure_order] & ~oldest_endpoint_mask[high_exposure_order]]
    if len(oldest_only_order) != len(high_exposure_only_order):
        raise RuntimeError('Endpoint-exclusive sets must have equal size at fixed coverage')
    discordant_slots = len(oldest_only_order)
    high_exposure_quota = int(np.floor(alpha * discordant_slots + 0.5))
    oldest_first_quota = discordant_slots - high_exposure_quota
    selected_high_exposure_only = high_exposure_only_order[:high_exposure_quota]
    selected_oldest_only = oldest_only_order[:oldest_first_quota]
    mask = np.zeros(N, dtype=bool)
    mask[shared_indices] = True
    mask[selected_high_exposure_only] = True
    mask[selected_oldest_only] = True
    allocation_source = np.full(N, 'not_vaccinated', dtype='<U30')
    allocation_source[shared_indices] = 'shared_endpoint'
    allocation_source[selected_high_exposure_only] = 'high_exposure_only_quota'
    allocation_source[selected_oldest_only] = 'oldest_first_only_quota'
    if int(mask.sum()) != num_vax:
        raise RuntimeError(f'Quota mixture selected {int(mask.sum())} individuals; expected {num_vax}')
    diagnostics = {'Shared Endpoint Count': int(len(shared_indices)), 'Endpoint Discordant Slots': int(discordant_slots), 'High Exposure Exclusive Quota': int(high_exposure_quota), 'Oldest First Exclusive Quota': int(oldest_first_quota)}
    return (mask, allocation_source, diagnostics)

def get_hybrid_vaccinated_mask(alpha, coverage):
    mask, _, _ = get_quota_mixture_allocation(alpha, coverage)
    return mask

def run_strategy_simulation(beta_val, vax_mask, ves_0):
    state = np.zeros(N, dtype=int)
    days_since_rec = np.full(N, -1, dtype=int)
    ever_infected = np.zeros(N, dtype=bool)
    cumulative_exposure = np.zeros(N, dtype=float)
    lambda_external_const = np.where(is_key_group, beta_val * strength_vir_work / 24.0 * EXTERNAL_PREVALENCE, 0.0) + np.where(daily_hours_nonfixed > 0, beta_val * strength_vir_social / 24.0 * EXTERNAL_PREVALENCE, 0.0)
    seeds = np.random.choice(N, size=NUM_SEEDS_PER_RUN, replace=False)
    state[seeds] = 2
    ever_infected[seeds] = True
    daily_infected_counts = []
    recorded_peak_size = -1
    recorded_peak_day = -1
    peak_infectious_by_strength_decile = np.zeros(10, dtype=np.int32)
    for d in range(1, SIM_DAYS_STRATEGY + 1):
        new_state = state.copy()
        r_val = np.random.random(N)
        r_rec = np.random.random(N)
        if d <= 100:
            current_ve = ves_0
        elif d <= 180:
            current_ve = 0.7 * ves_0
        else:
            current_ve = 0.4 * ves_0
        vax_susc_mult = np.where(vax_mask, 1.0 - current_ve, 1.0)
        nat_susc_mult = np.ones(N)
        mask_0_90 = (days_since_rec >= 0) & (days_since_rec <= 90)
        mask_91_365 = (days_since_rec > 90) & (days_since_rec <= 365)
        mask_gt_365 = days_since_rec > 365
        nat_susc_mult[mask_0_90] = 1.0 - NAT_EFF_MAX
        if np.any(mask_91_365):
            decay_ratio = (days_since_rec[mask_91_365] - 90) / (365 - 90)
            current_efficacy = NAT_EFF_MAX - (NAT_EFF_MAX - NAT_EFF_MIN) * decay_ratio
            nat_susc_mult[mask_91_365] = 1.0 - current_efficacy
        nat_susc_mult[mask_gt_365] = 1.0 - NAT_EFF_MIN
        total_susc_mult = vax_susc_mult * nat_susc_mult
        sus_indices = np.where(state == 0)[0]
        if len(sus_indices) > 0:
            I_vector = (state == 2).astype(float)
            lambda_internal = beta_val / 24.0 * adj_matrix.dot(I_vector)
            lambda_total_all = lambda_internal + lambda_external_const
            lambda_sus = lambda_total_all[sus_indices]
            susc_sus = total_susc_mult[sus_indices]
            cumulative_exposure[sus_indices] += lambda_sus
            prob_sus = 1.0 - np.exp(-susc_sus * lambda_sus)
            infected_mask = (prob_sus > 0) & (r_val[sus_indices] < prob_sus)
            newly_infected_indices = sus_indices[infected_mask]
            new_state[newly_infected_indices] = 1
            ever_infected[newly_infected_indices] = True
        exp_indices = np.where(state == 1)[0]
        new_state[exp_indices] = np.where(r_val[exp_indices] < PROB_E_TO_I, 2, 1)
        inf_indices = np.where(state == 2)[0]
        recovered = r_rec[inf_indices] < PROB_I_TO_S
        new_state[inf_indices[recovered]] = 0
        days_since_rec[inf_indices[recovered]] = 0
        mask_increment = (new_state == 0) & (days_since_rec >= 0)
        days_since_rec[mask_increment] += 1
        mask_infected = new_state != 0
        days_since_rec[mask_infected] = -1
        state = new_state
        infectious_mask = state == 2
        current_infectious_count = int(infectious_mask.sum())
        daily_infected_counts.append(current_infectious_count)
        if current_infectious_count > recorded_peak_size:
            recorded_peak_size = current_infectious_count
            recorded_peak_day = d
            peak_infectious_by_strength_decile = strength_decile_counts(infectious_mask).astype(np.int32)
    unique_infected_total = int(np.sum(ever_infected))
    daily_infected_counts = np.array(daily_infected_counts, dtype=np.int32)
    peak_sz = int(np.max(daily_infected_counts)) if len(daily_infected_counts) > 0 else 0
    peak_tm = int(np.argmax(daily_infected_counts)) if len(daily_infected_counts) > 0 else 0
    if recorded_peak_size != peak_sz:
        raise RuntimeError('Peak-size recorder does not match daily curve')
    if recorded_peak_day != peak_tm + 1:
        raise RuntimeError('Peak-day recorder does not match first argmax')
    final_ever_by_strength_decile = strength_decile_counts(ever_infected).astype(np.int32)
    if int(final_ever_by_strength_decile.sum()) != unique_infected_total:
        raise RuntimeError('Final-ever deciles do not sum to total infections')
    if int(peak_infectious_by_strength_decile.sum()) != peak_sz:
        raise RuntimeError('Peak deciles do not sum to peak size')
    infected_indices = np.where(ever_infected)[0]
    expected_deaths = calculate_expected_deaths(infected_indices, df['age_num'])
    return (unique_infected_total, peak_sz, peak_tm, expected_deaths, daily_infected_counts, ever_infected, cumulative_exposure, np.asarray(seeds, dtype=np.int32), final_ever_by_strength_decile, peak_infectious_by_strength_decile)

def point_tag(ves_0, r0, alpha):
    return f'VES{ves_0:.2f}_R0_{r0:g}_Alpha_{alpha:.2f}'.replace('.', 'p')

def task_seed(task_index):
    return int(RNG_SEED)

def run_single_hybrid_point(task_index, current_ves0, target_r0, alpha):
    start_time = time.time()
    seed = task_seed(task_index)
    np.random.seed(seed)
    random.seed(seed)
    print(f'[WORKER START] VES_0={current_ves0}, R0={target_r0}, alpha={alpha:.2f}, seed={seed}', flush=True)
    current_beta, internal_ratio, adjusted_target_r0 = calibrate_scenario(target_r0)
    vax_mask, allocation_source, allocation_diagnostics = get_quota_mixture_allocation(alpha, COVERAGE)
    actual_num_vax = int(np.sum(vax_mask))
    actual_coverage = actual_num_vax / N
    infection_rates = []
    unique_infected_totals = []
    peak_sizes = []
    peak_times = []
    expected_deaths_values = []
    daily_curve_matrix = np.zeros((MC_RUNS_PER_POINT, SIM_DAYS_STRATEGY), dtype=np.int32)
    ever_infected_matrix = np.zeros((MC_RUNS_PER_POINT, N), dtype=bool)
    cumulative_exposure_matrix = np.zeros((MC_RUNS_PER_POINT, N), dtype=np.float64)
    seed_node_indices_matrix = np.zeros((MC_RUNS_PER_POINT, NUM_SEEDS_PER_RUN), dtype=np.int32)
    final_ever_by_strength_decile_matrix = np.zeros((MC_RUNS_PER_POINT, 10), dtype=np.int32)
    peak_infectious_by_strength_decile_matrix = np.zeros((MC_RUNS_PER_POINT, 10), dtype=np.int32)
    for mc_run_id in range(1, MC_RUNS_PER_POINT + 1):
        unique_infected_total, peak_size, peak_time, expected_deaths, daily_curve, ever_infected_mask, cumulative_exposure_arr, seed_node_indices, final_ever_by_strength_decile, peak_infectious_by_strength_decile = run_strategy_simulation(current_beta, vax_mask, current_ves0)
        infection_rate = unique_infected_total / N
        infection_rates.append(infection_rate)
        unique_infected_totals.append(unique_infected_total)
        peak_sizes.append(peak_size)
        peak_times.append(peak_time)
        expected_deaths_values.append(expected_deaths)
        daily_curve_matrix[mc_run_id - 1, :] = daily_curve
        ever_infected_matrix[mc_run_id - 1, :] = ever_infected_mask
        cumulative_exposure_matrix[mc_run_id - 1, :] = cumulative_exposure_arr
        seed_node_indices_matrix[mc_run_id - 1, :] = seed_node_indices
        final_ever_by_strength_decile_matrix[mc_run_id - 1, :] = final_ever_by_strength_decile
        peak_infectious_by_strength_decile_matrix[mc_run_id - 1, :] = peak_infectious_by_strength_decile
    tag = point_tag(current_ves0, target_r0, alpha)
    replicate_df = pd.DataFrame({'VES_0': current_ves0, 'R0': target_r0, 'Coverage': COVERAGE, 'Actual Coverage': actual_coverage, 'Alpha High Exposure': alpha, 'Alpha Oldest First': 1.0 - alpha, 'High Exposure Discordant-Quota Fraction': alpha, 'Oldest First Discordant-Quota Fraction': 1.0 - alpha, 'MC_Run': np.arange(1, MC_RUNS_PER_POINT + 1), 'Simulation Seed': seed, 'Unique Infected Total': unique_infected_totals, 'Final Infection Proportion': infection_rates, 'Peak Size': peak_sizes, 'Peak Time': peak_times, 'Expected Deaths': expected_deaths_values})
    replicate_path = os.path.join(POINT_REPLICATE_DIR, f'Replicate_Outcomes_{tag}.csv.gz')
    replicate_df.to_csv(replicate_path, index=False, compression='gzip')
    raw_path = None
    if SAVE_RAW_POINT_DATA:
        raw_path = os.path.join(POINT_RAW_DIR, f'Raw_Point_Data_{tag}.npz')
        np.savez_compressed(raw_path, VES_0=np.array([current_ves0], dtype=float), R0=np.array([target_r0], dtype=float), Coverage=np.array([COVERAGE], dtype=float), Actual_Coverage=np.array([actual_coverage], dtype=float), Alpha_High_Exposure=np.array([alpha], dtype=float), Alpha_Oldest_First=np.array([1.0 - alpha], dtype=float), High_Exposure_Discordant_Quota_Fraction=np.array([alpha], dtype=float), Oldest_First_Discordant_Quota_Fraction=np.array([1.0 - alpha], dtype=float), Shared_Endpoint_Count=np.array([allocation_diagnostics['Shared Endpoint Count']], dtype=np.int32), Endpoint_Discordant_Slots=np.array([allocation_diagnostics['Endpoint Discordant Slots']], dtype=np.int32), High_Exposure_Exclusive_Quota=np.array([allocation_diagnostics['High Exposure Exclusive Quota']], dtype=np.int32), Oldest_First_Exclusive_Quota=np.array([allocation_diagnostics['Oldest First Exclusive Quota']], dtype=np.int32), Simulation_Seed=np.array([seed], dtype=np.int64), Calibrated_Beta=np.array([current_beta], dtype=float), Internal_Exposure_Ratio=np.array([internal_ratio], dtype=float), Adjusted_Target_R0=np.array([adjusted_target_r0], dtype=float), Vaccinated_Mask=vax_mask, Allocation_Source=allocation_source, Total_Strength_Decile_ID=TOTAL_STRENGTH_DECILE_IDS.astype(np.int16) + 1, Total_Strength_Decile_Node_Counts=TOTAL_STRENGTH_DECILE_COUNTS, Seed_Node_Indices=seed_node_indices_matrix, Daily_Infected_Counts=daily_curve_matrix, Ever_Infected_Matrix=ever_infected_matrix, Final_Ever_Infected_By_Strength_Decile=final_ever_by_strength_decile_matrix, Peak_Infectious_By_Strength_Decile=peak_infectious_by_strength_decile_matrix, Cumulative_Exposure_Matrix=cumulative_exposure_matrix, Unique_Infected_Total=np.asarray(unique_infected_totals, dtype=np.int32), Final_Infection_Proportion=np.asarray(infection_rates, dtype=np.float64), Peak_Size=np.asarray(peak_sizes, dtype=np.int32), Peak_Time=np.asarray(peak_times, dtype=np.int32), Expected_Deaths=np.asarray(expected_deaths_values, dtype=np.float64))
    vaccinated_indices = np.where(vax_mask)[0]
    if len(vaccinated_indices) > 0:
        vaccinated_mean_age = float(np.mean(AGE_VALUES[vaccinated_indices]))
        vaccinated_mean_strength = float(np.mean(EXPOSURE_VALUES[vaccinated_indices]))
    else:
        vaccinated_mean_age = np.nan
        vaccinated_mean_strength = np.nan
    summary = {'VES_0': current_ves0, 'R0': target_r0, 'Coverage': COVERAGE, 'Actual Coverage': actual_coverage, 'Actual Vaccinated Count': actual_num_vax, 'Alpha High Exposure': alpha, 'Alpha Oldest First': 1.0 - alpha, 'High Exposure Discordant-Quota Fraction': alpha, 'Oldest First Discordant-Quota Fraction': 1.0 - alpha, 'Allocation Mode': 'endpoint_set_quota_mixture', 'Shared Endpoint Count': allocation_diagnostics['Shared Endpoint Count'], 'Endpoint Discordant Slots': allocation_diagnostics['Endpoint Discordant Slots'], 'High Exposure Exclusive Quota': allocation_diagnostics['High Exposure Exclusive Quota'], 'Oldest First Exclusive Quota': allocation_diagnostics['Oldest First Exclusive Quota'], 'Simulation Seed': seed, 'MC Runs': MC_RUNS_PER_POINT, 'Simulation Days': SIM_DAYS_STRATEGY, 'Calibrated Beta': current_beta, 'Internal Exposure Ratio': internal_ratio, 'Adjusted Target R0': adjusted_target_r0, 'Vaccinated Mean Age': vaccinated_mean_age, 'Vaccinated Mean Total Strength': vaccinated_mean_strength}
    summary.update(summarize_values(infection_rates, 'Final Infection Proportion'))
    summary.update(summarize_values(peak_sizes, 'Peak Size'))
    summary.update(summarize_values(peak_times, 'Peak Time'))
    summary.update(summarize_values(expected_deaths_values, 'Expected Deaths'))
    point_summary_path = os.path.join(POINT_SUMMARY_DIR, f'Point_Summary_{tag}.csv')
    pd.DataFrame([summary]).to_csv(point_summary_path, index=False)
    print(f'[WORKER DONE] VES_0={current_ves0}, R0={target_r0}, alpha={alpha:.2f}; Elapsed={(time.time() - start_time) / 60:.1f} min', flush=True)
    return {'summary': summary, 'replicate_path': replicate_path, 'raw_path': raw_path}

def paired_metric_summary(values):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return {'mean': np.nan, 'sd': np.nan, 'se': np.nan, 'lower': np.nan, 'upper': np.nan, 'n': 0}
    mean_value = float(valid.mean())
    sd_value = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
    se_value = sd_value / math.sqrt(len(valid))
    return {'mean': mean_value, 'sd': sd_value, 'se': se_value, 'lower': mean_value - 1.96 * se_value, 'upper': mean_value + 1.96 * se_value, 'n': int(len(valid))}

def endpoint_raw_path(r0, alpha):
    tag = point_tag(VES_0_LIST[0], r0, alpha)
    return os.path.join(POINT_RAW_DIR, f'Raw_Point_Data_{tag}.npz')

def build_fig5_cd_endpoint_decomposition():
    metric_rows = []
    net_rows = []
    validation_rows = []
    required_arrays = {'Seed_Node_Indices', 'Final_Ever_Infected_By_Strength_Decile', 'Peak_Infectious_By_Strength_Decile', 'Final_Infection_Proportion', 'Peak_Size', 'Total_Strength_Decile_Node_Counts'}
    for r0 in FIG5_CD_R0_LIST:
        oldest_path = endpoint_raw_path(r0, 0.0)
        high_path = endpoint_raw_path(r0, 1.0)
        if not os.path.exists(oldest_path):
            raise FileNotFoundError(oldest_path)
        if not os.path.exists(high_path):
            raise FileNotFoundError(high_path)
        with np.load(oldest_path, allow_pickle=False) as oldest, np.load(high_path, allow_pickle=False) as high:
            for label, raw in [('oldest', oldest), ('high', high)]:
                missing = required_arrays.difference(raw.files)
                if missing:
                    raise RuntimeError(f'R0={r0:g} {label} raw file is missing: {sorted(missing)}')
            oldest_seeds = np.sort(oldest['Seed_Node_Indices'].astype(np.int32), axis=1)
            high_seeds = np.sort(high['Seed_Node_Indices'].astype(np.int32), axis=1)
            seeds_equal = np.array_equal(oldest_seeds, high_seeds)
            if not seeds_equal:
                raise RuntimeError(f'R0={r0:g}: endpoint seed sets are not paired')
            oldest_final = oldest['Final_Ever_Infected_By_Strength_Decile'].astype(np.int64)
            high_final = high['Final_Ever_Infected_By_Strength_Decile'].astype(np.int64)
            oldest_peak = oldest['Peak_Infectious_By_Strength_Decile'].astype(np.int64)
            high_peak = high['Peak_Infectious_By_Strength_Decile'].astype(np.int64)
            if oldest_final.shape != high_final.shape:
                raise RuntimeError(f'R0={r0:g}: endpoint final-decile arrays disagree')
            if oldest_peak.shape != high_peak.shape:
                raise RuntimeError(f'R0={r0:g}: endpoint peak-decile arrays disagree')
            node_counts = oldest['Total_Strength_Decile_Node_Counts'].astype(np.int32)
            if not np.array_equal(node_counts, high['Total_Strength_Decile_Node_Counts'].astype(np.int32)):
                raise RuntimeError(f'R0={r0:g}: endpoint decile definitions disagree')
            oldest_fip = oldest['Final_Infection_Proportion'].astype(float)
            high_fip = high['Final_Infection_Proportion'].astype(float)
            oldest_peak_total = oldest['Peak_Size'].astype(float)
            high_peak_total = high['Peak_Size'].astype(float)
            fip_contribution = (high_final - oldest_final) / float(N) * 100.0
            peak_contribution = high_peak - oldest_peak
            net_fip = (high_fip - oldest_fip) * 100.0
            net_peak = high_peak_total - oldest_peak_total
            fip_error = float(np.max(np.abs(fip_contribution.sum(axis=1) - net_fip)))
            peak_error = float(np.max(np.abs(peak_contribution.sum(axis=1) - net_peak)))
            if fip_error > 1e-10:
                raise RuntimeError(f'R0={r0:g}: FIP contributions are not additive')
            if peak_error > 1e-10:
                raise RuntimeError(f'R0={r0:g}: peak contributions are not additive')
            for group_id in range(10):
                label = f'{10 * group_id}-{10 * (group_id + 1)}'
                metric_inputs = [('FIP_Contribution_pp', fip_contribution[:, group_id], high_final[:, group_id] / float(N) * 100.0, oldest_final[:, group_id] / float(N) * 100.0), ('Peak_Contribution_People', peak_contribution[:, group_id], high_peak[:, group_id], oldest_peak[:, group_id])]
                for metric, difference, high_values, oldest_values in metric_inputs:
                    stats = paired_metric_summary(difference)
                    metric_rows.append({'R0': r0, 'VES_0': VES_0_LIST[0], 'Coverage': COVERAGE, 'Group_ID': group_id + 1, 'Exposure_Percentile_Range': label, 'Node_Count': int(node_counts[group_id]), 'Metric': metric, 'High_Exposure_Mean': float(np.mean(high_values)), 'Oldest_First_Mean': float(np.mean(oldest_values)), 'High_Minus_Oldest_Mean': stats['mean'], 'Paired_SD': stats['sd'], 'Paired_SE': stats['se'], 'Paired_CI95_Lower': stats['lower'], 'Paired_CI95_Upper': stats['upper'], 'Valid_Paired_Runs': stats['n']})
            for metric, difference, high_values, oldest_values in [('FIP_Contribution_pp', net_fip, high_fip * 100.0, oldest_fip * 100.0), ('Peak_Contribution_People', net_peak, high_peak_total, oldest_peak_total)]:
                stats = paired_metric_summary(difference)
                net_rows.append({'R0': r0, 'VES_0': VES_0_LIST[0], 'Coverage': COVERAGE, 'Metric': metric, 'High_Exposure_Mean': float(np.mean(high_values)), 'Oldest_First_Mean': float(np.mean(oldest_values)), 'Net_Difference_High_Minus_Oldest': stats['mean'], 'Paired_SD': stats['sd'], 'Paired_SE': stats['se'], 'Paired_CI95_Lower': stats['lower'], 'Paired_CI95_Upper': stats['upper'], 'Valid_Paired_Runs': stats['n']})
            validation_rows.append({'R0': r0, 'Paired_Seed_Sets_Equal': seeds_equal, 'FIP_Contribution_Max_Abs_Error_pp': fip_error, 'Peak_Contribution_Max_Abs_Error_People': peak_error, 'Replicates': int(len(net_fip)), 'Deciles': 10, 'Nodes': N})
    metric_df = pd.DataFrame(metric_rows).sort_values(['R0', 'Metric', 'Group_ID'])
    net_df = pd.DataFrame(net_rows).sort_values(['R0', 'Metric'])
    validation_df = pd.DataFrame(validation_rows).sort_values('R0')
    decomposition_path = os.path.join(FIG5_CD_DIR, 'Strength_Decile_Paired_Decomposition.csv')
    net_path = os.path.join(FIG5_CD_DIR, 'Fig5_CD_Net_Differences.csv')
    validation_path = os.path.join(FIG5_CD_DIR, 'Fig5_CD_Validation.csv')
    metric_df.to_csv(decomposition_path, index=False)
    net_df.to_csv(net_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    plot_ready_paths = {}
    for metric, filename in [('FIP_Contribution_pp', 'Panel_C_FIP_Plot_Ready.csv'), ('Peak_Contribution_People', 'Panel_D_Peak_Plot_Ready.csv')]:
        selected = metric_df[metric_df['Metric'] == metric].copy()
        wide = selected.pivot(index='R0', columns='Exposure_Percentile_Range', values='High_Minus_Oldest_Mean')
        ordered_columns = [f'{10 * i}-{10 * (i + 1)}' for i in range(10)]
        wide = wide.reindex(columns=ordered_columns)
        net_selected = net_df[net_df['Metric'] == metric].set_index('R0')
        wide['Net difference'] = net_selected['Net_Difference_High_Minus_Oldest']
        wide = wide.reset_index()
        path = os.path.join(FIG5_CD_DIR, filename)
        wide.to_csv(path, index=False)
        plot_ready_paths[metric] = path
    return {'decomposition': decomposition_path, 'net': net_path, 'validation': validation_path, 'panel_c': plot_ready_paths['FIP_Contribution_pp'], 'panel_d': plot_ready_paths['Peak_Contribution_People']}
if __name__ == '__main__':
    print(f'\n{'=' * 72}', flush=True)
    print('QUOTA-MIXTURE PEAK-vs-FINAL-INFECTION TRADE-OFF ANALYSIS', flush=True)
    print(f'{'=' * 72}', flush=True)
    print(f'[CONFIG] Data path: {DATA_PATH}', flush=True)
    print(f'[CONFIG] Output root: {OUTPUT_ROOT}', flush=True)
    print(f'[CONFIG] R0_LIST = {R0_LIST}', flush=True)
    print(f'[CONFIG] VES_0_LIST = {VES_0_LIST}', flush=True)
    print(f'[CONFIG] COVERAGE = {COVERAGE}', flush=True)
    print(f'[CONFIG] HYBRID_ALPHA_LIST = {HYBRID_ALPHA_LIST}', flush=True)
    print(f'[CONFIG] MC_RUNS_PER_POINT = {MC_RUNS_PER_POINT}', flush=True)
    print(f'[CONFIG] SIM_DAYS_STRATEGY = {SIM_DAYS_STRATEGY}', flush=True)
    print(f'[CONFIG] MAX_CORES = {MAX_CORES}', flush=True)
    household_codes, _ = pd.factorize(df['_household_key'])
    node_attribute_df = pd.DataFrame({'Node_ID': np.arange(N, dtype=int), 'Anonymous_Household_ID': household_codes.astype(int), 'Age': AGE_VALUES, 'Work_School_Activity_Status': is_key_group.astype(int), 'Daily_Activity_Duration': mobility_index, 'Internal_Weighted_Strength': strength_internal, 'External_Virtual_Strength': strength_virtual, 'Total_Weighted_Exposure_Strength': total_node_strength, 'Average_Neighbour_Exposure_Strength': avg_neighbor_strength, 'Total_Strength_Population_Decile': TOTAL_STRENGTH_DECILE_IDS + 1})
    node_attribute_df.to_csv(os.path.join(METADATA_DIR, 'Network_Node_Attributes.csv.gz'), index=False, compression='gzip')
    grouping_rows = []
    for group_id in range(10):
        members = TOTAL_STRENGTH_DECILE_IDS == group_id
        group_values = EXPOSURE_VALUES[members]
        grouping_rows.append({'Grouping': 'Total_Strength_Decile', 'Group_ID_One_Based': group_id + 1, 'Label': f'{10 * group_id}-{10 * (group_id + 1)}', 'Node_Count': int(members.sum()), 'Minimum_Total_Strength': float(group_values.min()), 'Maximum_Total_Strength': float(group_values.max())})
    pd.DataFrame(grouping_rows).to_csv(os.path.join(METADATA_DIR, 'Grouping_Definitions.csv'), index=False)
    allocation_rows = []
    for alpha in HYBRID_ALPHA_LIST:
        mask, allocation_source, allocation_diagnostics = get_quota_mixture_allocation(alpha, COVERAGE)
        for node_id in range(N):
            allocation_rows.append({'Alpha High Exposure': alpha, 'Alpha Oldest First': 1.0 - alpha, 'High Exposure Discordant-Quota Fraction': alpha, 'Oldest First Discordant-Quota Fraction': 1.0 - alpha, 'Allocation Mode': 'endpoint_set_quota_mixture', 'Coverage': COVERAGE, 'Shared Endpoint Count': allocation_diagnostics['Shared Endpoint Count'], 'Endpoint Discordant Slots': allocation_diagnostics['Endpoint Discordant Slots'], 'High Exposure Exclusive Quota': allocation_diagnostics['High Exposure Exclusive Quota'], 'Oldest First Exclusive Quota': allocation_diagnostics['Oldest First Exclusive Quota'], 'Node_ID': node_id, 'Vaccinated': int(mask[node_id]), 'Allocation Source': allocation_source[node_id], 'Age': AGE_VALUES[node_id], 'Total Weighted Exposure Strength': EXPOSURE_VALUES[node_id], 'Age Percentile': AGE_PERCENTILE[node_id], 'Exposure Percentile': EXPOSURE_PERCENTILE[node_id]})
    pd.DataFrame(allocation_rows).to_csv(os.path.join(METADATA_DIR, 'Quota_Mixture_Vaccination_Allocation_By_Alpha.csv.gz'), index=False, compression='gzip')
    configuration_rows = [('N', N), ('DATA_FILE', DATA_FILE), ('SHEET_ID', SHEET_ID), ('MAX_CORES', MAX_CORES), ('RNG_SEED', RNG_SEED), ('PROB_E_TO_I', PROB_E_TO_I), ('PROB_I_TO_S', PROB_I_TO_S), ('NAT_EFF_MAX', NAT_EFF_MAX), ('NAT_EFF_MIN', NAT_EFF_MIN), ('HH_CONTACT_HOURS', HH_CONTACT_HOURS), ('SOCIAL_ENCOUNTER_DURATION', SOCIAL_ENCOUNTER_DURATION), ('INTERNAL_SOCIAL_RATIO', INTERNAL_SOCIAL_RATIO), ('VIRTUAL_CONTACT_MAPPING', repr(VIRTUAL_CONTACT_MAPPING)), ('EXTERNAL_PREVALENCE', EXTERNAL_PREVALENCE), ('EXTERNAL_SOCIAL_CONTACT_COUNT', EXTERNAL_SOCIAL_CONTACT_COUNT), ('NUM_SEEDS_PER_RUN', NUM_SEEDS_PER_RUN), ('MC_RUNS_PER_POINT', MC_RUNS_PER_POINT), ('SIM_DAYS_STRATEGY', SIM_DAYS_STRATEGY), ('COVERAGE', COVERAGE), ('R0_LIST', repr(R0_LIST)), ('VES_0_LIST', repr(VES_0_LIST)), ('HYBRID_ALPHA_LIST', repr(HYBRID_ALPHA_LIST)), ('HYBRID_ALPHA_STEP', 0.02), ('FIG5_CD_R0_LIST', repr(FIG5_CD_R0_LIST)), ('Quota-mixture strategy definition', 'shared endpoint recipients are always vaccinated; among endpoint-discordant slots, alpha is the high-exposure-only quota fraction and 1-alpha is the oldest-first-only quota fraction; alpha=0 exact oldest-first; alpha=1 exact high-exposure'), ('SAVE_RAW_POINT_DATA', SAVE_RAW_POINT_DATA), ('MAKE_PRELIMINARY_FIGURE', MAKE_PRELIMINARY_FIGURE), ('Fig5 C/D retained outputs', 'paired seed node indices; final ever-infected counts by total-strength decile; infectious counts by total-strength decile at each replicate-specific global peak')]
    pd.DataFrame(configuration_rows, columns=['Parameter', 'Value']).to_csv(os.path.join(METADATA_DIR, 'Run_Configuration.csv'), index=False)
    tasks = []
    task_index = 0
    for current_ves0 in VES_0_LIST:
        for target_r0 in R0_LIST:
            for alpha in HYBRID_ALPHA_LIST:
                tasks.append((task_index, current_ves0, target_r0, alpha))
                task_index += 1
    print(f'[INFO] Total quota-mixture points = {len(tasks)}', flush=True)
    print(f'[INFO] Total epidemic simulations = {len(tasks) * MC_RUNS_PER_POINT}', flush=True)
    actual_workers = min(MAX_CORES, len(tasks))
    print(f'[INFO] Parallel workers = {actual_workers}', flush=True)
    start_time = time.time()
    all_summary_rows = []
    all_replicate_frames = []
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = {executor.submit(run_single_hybrid_point, task_idx, ves0, r0, alpha): (task_idx, ves0, r0, alpha) for task_idx, ves0, r0, alpha in tasks}
        completed = 0
        for future in as_completed(futures):
            task_idx, ves0, r0, alpha = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f'[ERROR] task={task_idx}, VES_0={ves0}, R0={r0}, alpha={alpha:.2f}: {exc}', flush=True)
                raise
            all_summary_rows.append(result['summary'])
            all_replicate_frames.append(pd.read_csv(result['replicate_path']))
            completed += 1
            print(f'[PROGRESS] {completed}/{len(tasks)} points completed; elapsed={(time.time() - start_time) / 60:.1f} min', flush=True)
    summary_df = pd.DataFrame(all_summary_rows).sort_values(['VES_0', 'R0', 'Alpha High Exposure']).reset_index(drop=True)
    summary_path = os.path.join(OUTPUT_ROOT, 'Quota_Mixture_Tradeoff_AllPoint_Summary.csv')
    summary_df.to_csv(summary_path, index=False)
    replicate_df = pd.concat(all_replicate_frames, ignore_index=True).sort_values(['VES_0', 'R0', 'Alpha High Exposure', 'MC_Run']).reset_index(drop=True)
    combined_replicate_path = os.path.join(OUTPUT_ROOT, 'Quota_Mixture_Tradeoff_AllReplicate_Outcomes.csv.gz')
    replicate_df.to_csv(combined_replicate_path, index=False, compression='gzip')
    fig5_cd_paths = build_fig5_cd_endpoint_decomposition()
    print(f'\n[DONE] Total elapsed time = {(time.time() - start_time) / 60:.1f} min', flush=True)
    print(f'[SAVED] {summary_path}', flush=True)
    print(f'[SAVED] {combined_replicate_path}', flush=True)
    for output_label, output_path in fig5_cd_paths.items():
        print(f'[SAVED] Fig5 C/D {output_label}: {output_path}', flush=True)
    print(f'[SAVED] Raw point data directory: {POINT_RAW_DIR}', flush=True)
    print(f'[SAVED] Metadata directory: {METADATA_DIR}', flush=True)
