"""All-or-nothing transmission engine for Figures S1-S3."""
from pathlib import Path
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import matplotlib
matplotlib.use('Agg')
import random
import hashlib
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import eigsh
current_path = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = '20231012.xlsx'
DATA_PATH = os.environ.get('LEAKY_DATA', str(Path(__file__).resolve().parents[1] / 'data' / '20231012.xlsx'))
SHEET_ID = 0
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
VES_0_LIST = [float(x) for x in os.environ['LEAKY_VE'].split(',')] if 'LEAKY_VE' in os.environ else [0.2, 0.35, 0.5, 0.65, 0.8]
R0_LIST = [float(x) for x in os.environ['LEAKY_R0'].split(',')] if 'LEAKY_R0' in os.environ else [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.3, 3.5, 4.0, 5.0, 7.0, 8.5, 10.0, 12.0, 14.0, 16.0, 18.6]
MC_RUNS_PER_POINT = int(os.environ.get('LEAKY_RUNS', 100))
SIM_DAYS_STRATEGY = int(os.environ.get('LEAKY_DAYS', 500))
COVERAGE_STEPS = np.array([float(x) for x in os.environ['LEAKY_COVERAGE'].split(',')]) if 'LEAKY_COVERAGE' in os.environ else np.linspace(0, 0.95, 20)
OUTPUT_ROOT = os.environ.get('LEAKY_OUTPUT', str(Path(__file__).resolve().parents[1] / 'results' / 'all_or_nothing'))
VACCINE_FAILURE_MODE = 'All-or-nothing'
AON_TAKE_ASSIGNMENT = 'Independent Bernoulli draw per vaccinated recipient and MC replicate'
AON_PROTECTION_DURATION = 'Waning: VES_0 through day 100; 0.7*VES_0 on days 101-180; 0.4*VES_0 thereafter'
INITIAL_SEED_INTERPRETATION = 'Imported infections present before vaccine protection is applied'

def _stable_uint32_seed(*items):
    payload = '|'.join((repr(item) for item in items)).encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], 'little') % (2 ** 32 - 1)

def get_aon_protected_mask(vax_mask, ves_0, target_r0, coverage, mc_run_id):
    take_seed = _stable_uint32_seed(RNG_SEED, ves_0, target_r0, coverage, mc_run_id, 'AON_take')
    take_rng = np.random.default_rng(take_seed)
    aon_take_uniforms = take_rng.random(N)
    protected_mask = vax_mask & (aon_take_uniforms < ves_0)
    return (protected_mask, aon_take_uniforms, take_seed)
S9_STRATEGIES = {'Random', 'High Strength (Node)', 'Priority: Oldest First'}
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
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f'Data file not found: {DATA_PATH}')
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_ID)
N = len(df)
print(f'[INFO] Loaded N = {N}')
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
print(f'[HOUSEHOLD CHECK] Unique households by community + address: {df['_household_key'].nunique()}')
print(f'[HOUSEHOLD CHECK] Unique household-member keys after adding household-member index: {df['_household_member_key'].nunique()}')
print(f'[HOUSEHOLD CHECK] Households with >=2 respondents: {(household_sizes >= 2).sum()}')
if duplicated_member_keys.any():
    print(f'[HOUSEHOLD WARNING] {duplicated_member_keys.sum()} rows have duplicated community + household + member-index keys. Rows are retained in the supplied consolidated dataset.')
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

def run_strategy_simulation(beta_val, vax_mask, ves_0, aon_take_uniforms):
    state = np.zeros(N, dtype=int)
    days_since_rec = np.full(N, -1, dtype=int)
    ever_infected = np.zeros(N, dtype=bool)
    cumulative_exposure = np.zeros(N, dtype=float)
    lambda_external_const = np.where(is_key_group, beta_val * strength_vir_work / 24.0 * EXTERNAL_PREVALENCE, 0.0) + np.where(daily_hours_nonfixed > 0, beta_val * strength_vir_social / 24.0 * EXTERNAL_PREVALENCE, 0.0)
    seeds = np.random.choice(N, size=NUM_SEEDS_PER_RUN, replace=False)
    state[seeds] = 2
    ever_infected[seeds] = True
    daily_infected_counts = []
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
        current_aon_protected_mask = vax_mask & (aon_take_uniforms < current_ve)
        vax_susc_mult = np.where(current_aon_protected_mask, 0.0, 1.0)
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
        daily_infected_counts.append(np.sum(state == 2))
    unique_infected_total = np.sum(ever_infected)
    daily_infected_counts = np.array(daily_infected_counts)
    peak_sz = np.max(daily_infected_counts) if len(daily_infected_counts) > 0 else 0
    peak_tm = np.argmax(daily_infected_counts) if len(daily_infected_counts) > 0 else 0
    infected_indices = np.where(ever_infected)[0]
    expected_deaths = calculate_expected_deaths(infected_indices, df['age_num'])
    return (unique_infected_total, peak_sz, peak_tm, expected_deaths, daily_infected_counts, ever_infected, cumulative_exposure, np.asarray(seeds, dtype=np.int32))
