"""Disease-parameter sensitivity experiments for Figures S6-S7."""
from __future__ import annotations
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
SCRIPT_DIR = Path(__file__).resolve().parent
HH_CONTACT_HOURS = 12.0
SOCIAL_ENCOUNTER_DURATION = 0.25
INTERNAL_SOCIAL_RATIO = 0.3
VIRTUAL_CONTACT_MAPPING = {'a': 3.0, 'b': 10.0, 'c': 20.0, 'd': 30.0}
EXTERNAL_PREVALENCE = 0.005
# Effective contact count, not contacts per hour.
EXTERNAL_SOCIAL_CONTACT_COUNT = 4.0
DEFAULT_R0_VALUES = [1.0, 2.0, 4.0, 8.0]
DEFAULT_VES_VALUES = [0.2, 0.5, 0.8]
DEFAULT_STRATEGIES = ['Random', 'High Strength (Node)', 'Priority: Oldest First']
OUTCOME_COLUMNS = ['Final Infection Proportion', 'Epidemic Peak Size', 'Expected Deaths']
OUTCOME_LABELS = {'Final Infection Proportion': 'Final infection proportion', 'Epidemic Peak Size': 'Epidemic peak size', 'Expected Deaths': 'Expected deaths (scenario IFR)'}
FREQ_MAP = {'\u65e0': 0.0, '<1\u6b21': 0.5, '1-2\u6b21': 1.5, '3-4\u6b21': 3.5, '\u22655\u6b21': 5.5, 'a': 5.5, 'b': 3.5, 'c': 1.5, 'd': 0.5}
DUR_MAP = {'\u65e0': 0.0, '2\u5c0f\u65f6\u4ee5\u5185': 1.0, '2-5\u5c0f\u65f6': 3.5, '5\u5c0f\u65f6\u4ee5\u4e0a': 6.0, 'a': 1.0, 'b': 3.5, 'c': 6.0}
AGE_CODE_MAP = {'a': 15.0, 'b': 24.0, 'c': 34.5, 'd': 44.5, 'e': 54.5, 'f': 64.5, 'g': 75.0}
DEFAULT_SCENARIO_PARAMETERS: dict[str, dict[str, Any]] = {'Baseline': {'display_name': 'Baseline (SARS-CoV-2-like reference)', 'pathogen_label': 'SARS-CoV-2-like', 'interpretation': 'Reference disease-course and immunity parameterization used in the main analysis.', 'r0_reference': 2.6, 'r0_literature_range': [2.0, 3.6], 'latent_period_days': 5.0, 'infectious_period_days': 7.04, 'prob_e_to_i': 0.2, 'prob_i_to_s': 0.142, 'vax_full_days': 100, 'vax_mid_days': 180, 'vax_mid_multiplier': 0.7, 'vax_long_multiplier': 0.4, 'natural_immunity_initial': 0.652, 'natural_immunity_long_term': 0.247, 'natural_immunity_full_days': 90, 'natural_immunity_wane_end_day': 365, 'parameter_note': 'Baseline values preserve the supplied main-analysis setting.', 'source_keys': ['LAUER_2020_COVID_INCUBATION', 'LIU_2020_COVID_R0_REVIEW', 'VERITY_2020_COVID_IFR', 'COVID_PRIOR_INFECTION_META_2023']}, 'Disease_A': {'display_name': 'Disease A (seasonal-influenza-like)', 'pathogen_label': 'Seasonal influenza-like', 'interpretation': 'Literature-informed short-course respiratory-pathogen robustness profile.', 'r0_reference': 1.28, 'r0_literature_range': [1.19, 1.37], 'latent_period_days': 1.4, 'infectious_period_days': 4.8, 'prob_e_to_i': 0.714286, 'prob_i_to_s': 0.208333, 'vax_full_days': 60, 'vax_mid_days': 120, 'vax_mid_multiplier': 0.7, 'vax_long_multiplier': 0.4, 'natural_immunity_initial': 0.5, 'natural_immunity_long_term': 0.1, 'natural_immunity_full_days': 60, 'natural_immunity_wane_end_day': 180, 'parameter_note': 'Daily transition probabilities are inverse mean durations.', 'source_keys': ['LESSLER_2009_INCUBATION', 'CARRAT_2008_INFLUENZA_TIMELINE', 'BIGGERSTAFF_2014_INFLUENZA_R0', 'MCDONALD_2023_INFLUENZA_CFR']}, 'Disease_B': {'display_name': 'Disease B (RSV-like)', 'pathogen_label': 'RSV-like', 'interpretation': 'Literature-informed RSV-like respiratory-pathogen robustness profile.', 'r0_reference': 1.7, 'r0_literature_range': [1.2, 2.7], 'latent_period_days': 4.4, 'infectious_period_days': 5.0, 'prob_e_to_i': 0.227273, 'prob_i_to_s': 0.2, 'vax_full_days': 90, 'vax_mid_days': 180, 'vax_mid_multiplier': 0.7, 'vax_long_multiplier': 0.4, 'natural_immunity_initial': 0.4, 'natural_immunity_long_term': 0.05, 'natural_immunity_full_days': 60, 'natural_immunity_wane_end_day': 180, 'parameter_note': 'The coarse age structure cannot calibrate infant RSV burden.', 'source_keys': ['LESSLER_2009_INCUBATION', 'REIS_SHAMAN_2016_RSV_MODEL_R0_DURATION', 'RSV_REINFECTION_IMMUNITY', 'RSV_ADULT_BURDEN_IFR_PROXY']}}
DEFAULT_IFR_CONFIG: dict[str, Any] = {'profiles': {'Reference_IFR': {'description': 'Alias retained for backward compatibility; same values as SARS_CoV_2_IFR.', 'values': {'15': 6.95e-05, '24': 0.000309, '34.5': 0.000844, '44.5': 0.00161, '54.5': 0.00595, '64.5': 0.0193, '75': 0.0428}}, 'SARS_CoV_2_IFR': {'description': 'Age-specific COVID-19-like IFR retained from the main analysis.', 'values': {'15': 6.95e-05, '24': 0.000309, '34.5': 0.000844, '44.5': 0.00161, '54.5': 0.00595, '64.5': 0.0193, '75': 0.0428}}, 'Seasonal_Influenza_IFR': {'description': 'Seasonal-influenza-like infection fatality proxy.', 'values': {'15': 6.69e-06, '24': 1.338e-05, '34.5': 2.342e-05, '44.5': 4.683e-05, '54.5': 0.00012042, '64.5': 0.00049841, '75': 0.0032781}}, 'RSV_IFR_Proxy': {'description': 'RSV-like infection fatality proxy for the supplied coarse age groups.', 'values': {'15': 1e-06, '24': 3e-06, '34.5': 1e-05, '44.5': 3e-05, '54.5': 0.00015, '64.5': 0.0066, '75': 0.012}}}, 'scenario_profile': {'Baseline': 'SARS_CoV_2_IFR', 'Disease_A': 'Seasonal_Influenza_IFR', 'Disease_B': 'RSV_IFR_Proxy'}}

@dataclass
class ModelData:
    n: int
    adjacency: csr_matrix
    spectral_radius: float
    internal_exposure_ratio: float
    age_values: np.ndarray
    age_groups: np.ndarray
    is_key_group: np.ndarray
    daily_hours_nonfixed: np.ndarray
    mobility_index: np.ndarray
    strength_internal: np.ndarray
    strength_virtual_work: np.ndarray
    strength_virtual_social: np.ndarray
    strength_virtual: np.ndarray
    total_strength: np.ndarray
    average_neighbor_strength: np.ndarray
    node_metadata: pd.DataFrame
_WORKER_MODEL: ModelData | None = None
_WORKER_SCENARIOS: dict[str, dict[str, Any]] | None = None
_WORKER_SETTINGS: dict[str, Any] | None = None
_WORKER_VAX_MASKS: dict[str, np.ndarray] | None = None
_WORKER_IFR_BY_SCENARIO: dict[str, np.ndarray] | None = None

def normalize_column(value: Any) -> str:
    return str(value).replace('\ufeff', '').strip().replace(' ', '').replace('\uff08', '(').replace('\uff09', ')').replace('/', '')

def pick_column(frame: pd.DataFrame, targets: Iterable[str]) -> str:
    normalized = {column: normalize_column(column) for column in frame.columns}
    wanted = [normalize_column(target) for target in targets]
    for column, candidate in normalized.items():
        if candidate in wanted:
            return str(column)
    for column, candidate in normalized.items():
        if any((value in candidate for value in wanted)) or any((candidate in value for value in wanted)):
            return str(column)
    raise KeyError(f'Column not found. Tried: {list(targets)}')

def map_frequency(value: Any) -> float:
    text = str(value).strip()
    if text in FREQ_MAP:
        return FREQ_MAP[text]
    if '5\u6b21' in text:
        return 5.5
    return 0.5

def map_duration(value: Any) -> float:
    text = str(value).strip()
    if text in DUR_MAP:
        return DUR_MAP[text]
    if '2-5' in text:
        return 3.5
    return 0.0

def is_employed(value: Any) -> bool:
    text = str(value).strip()
    return text in {'b', 'B', 'd', 'D'} or '\u5de5' in text or '\u73ed' in text

def is_student(value: Any) -> bool:
    text = str(value).strip()
    return '\u5b66' in text or text in {'a', 'A'}

def map_virtual_contacts(value: Any) -> float:
    text = str(value).lower().strip()
    for code, count in VIRTUAL_CONTACT_MAPPING.items():
        if code in text:
            return count
    return 0.0

def stable_seed(*items: Any, base_seed: int) -> int:
    key = '|'.join(map(str, items))
    digest = hashlib.sha256(key.encode('utf-8')).hexdigest()
    return int((base_seed + int(digest[:12], 16)) % (2 ** 32 - 1))

def age_label(age: float) -> str:
    return str(int(age)) if float(age).is_integer() else str(float(age))

def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def resolve_existing_path(path: Path, description: str) -> Path:
    if path.is_absolute():
        resolved = path.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f'{description} not found: {resolved}')
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    script_candidate = (SCRIPT_DIR / path).resolve()
    if script_candidate.exists():
        return script_candidate
    raise FileNotFoundError(f'{description} not found. Tried: {cwd_candidate} and {script_candidate}')

def load_json_or_default(path: Path | None, description: str, default_payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if path is None:
        return (default_payload, 'built-in defaults')
    try:
        resolved = resolve_existing_path(path, description)
    except FileNotFoundError as exc:
        print(f'[WARN] {exc}', flush=True)
        print(f'[WARN] Using built-in default {description.lower()}.', flush=True)
        return (default_payload, 'built-in defaults')
    return (load_json(resolved), str(resolved))

def load_ifr_profiles_from_payload(payload: dict[str, Any]) -> tuple[dict[str, dict[float, float]], dict[str, str], dict[str, str]]:
    profiles = payload.get('profiles', {})
    if not profiles:
        raise ValueError('IFR configuration contains no profiles')
    parsed_profiles: dict[str, dict[float, float]] = {}
    descriptions: dict[str, str] = {}
    for profile_name, profile in profiles.items():
        values = profile.get('values', profile)
        parsed_profiles[str(profile_name)] = {float(age): float(value) for age, value in values.items()}
        descriptions[str(profile_name)] = str(profile.get('description', ''))
    scenario_profile = {str(key): str(value) for key, value in payload.get('scenario_profile', {}).items()}
    return (parsed_profiles, descriptions, scenario_profile)

def resolve_scenario_ifr_profiles(scenarios: dict[str, dict[str, Any]], profiles: dict[str, dict[float, float]], scenario_profile: dict[str, str], fallback_profile: str, common_profile: str | None) -> dict[str, str]:
    if common_profile:
        if common_profile not in profiles:
            raise KeyError(f"Common IFR profile '{common_profile}' not found")
        return {scenario_name: common_profile for scenario_name in scenarios}
    if fallback_profile not in profiles:
        raise KeyError(f"Fallback IFR profile '{fallback_profile}' not found")
    resolved = {}
    for scenario_name in scenarios:
        profile_name = scenario_profile.get(scenario_name, fallback_profile)
        if profile_name not in profiles:
            raise KeyError(f"Scenario {scenario_name} requests missing IFR profile '{profile_name}'")
        resolved[scenario_name] = profile_name
    return resolved

def ifr_mapping_for_age_groups(age_groups: np.ndarray, ifr_map: dict[float, float]) -> dict[str, float]:
    available_ifr_ages = np.array(sorted(ifr_map), dtype=float)
    mapped: dict[str, float] = {}
    for age in age_groups:
        nearest_age = available_ifr_ages[np.argmin(np.abs(available_ifr_ages - float(age)))]
        mapped[age_label(float(age))] = ifr_map[float(nearest_age)]
    return mapped

def make_ifr_by_node(age_values: np.ndarray, ifr_map: dict[float, float]) -> np.ndarray:
    available_ifr_ages = np.array(sorted(ifr_map), dtype=float)
    ifr_by_node = np.empty(len(age_values), dtype=float)
    for index, age in enumerate(age_values):
        nearest_age = available_ifr_ages[np.argmin(np.abs(available_ifr_ages - float(age)))]
        ifr_by_node[index] = ifr_map[float(nearest_age)]
    return ifr_by_node

def validate_scenarios(scenarios: dict[str, dict[str, Any]]) -> None:
    required = {'display_name', 'interpretation', 'r0_reference', 'prob_e_to_i', 'prob_i_to_s', 'vax_full_days', 'vax_mid_days', 'vax_mid_multiplier', 'vax_long_multiplier', 'natural_immunity_initial', 'natural_immunity_long_term', 'natural_immunity_full_days', 'natural_immunity_wane_end_day'}
    for name, params in scenarios.items():
        missing = sorted(required - set(params))
        if missing:
            raise ValueError(f'Scenario {name} is missing parameters: {missing}')
        probabilities = ['prob_e_to_i', 'prob_i_to_s', 'vax_mid_multiplier', 'vax_long_multiplier', 'natural_immunity_initial', 'natural_immunity_long_term']
        for key in probabilities:
            value = float(params[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'Scenario {name}: {key} must be in [0, 1]')
        if float(params['r0_reference']) <= 0.0:
            raise ValueError(f'Scenario {name}: r0_reference must be positive')
        if int(params['vax_mid_days']) < int(params['vax_full_days']):
            raise ValueError(f'Scenario {name}: vax_mid_days precedes vax_full_days')
        if int(params['natural_immunity_wane_end_day']) < int(params['natural_immunity_full_days']):
            raise ValueError(f'Scenario {name}: natural immunity waning interval is invalid')

def build_model(data_file: Path, sheet: str | int, seed: int) -> ModelData:
    if not data_file.exists():
        raise FileNotFoundError(f'Input workbook not found: {data_file}')
    try:
        sheet_value: str | int = int(sheet)
    except (TypeError, ValueError):
        sheet_value = sheet
    frame = pd.read_excel(data_file, sheet_name=sheet_value)
    n = len(frame)
    if n == 0:
        raise ValueError('Input workbook contains no rows')
    col_community = pick_column(frame, ['\u5c0f\u533a\u540d\u79f0', '\u5c0f\u533a', '\u793e\u533a\u540d\u79f0', '\u793e\u533a'])
    col_address = pick_column(frame, ['\u4f4f\u6237\u8be6\u7ec6\u5730\u5740', '\u4f4f\u5740'])
    col_work_status = pick_column(frame, ['\u5de5\u4f5c\u72b6\u6001'])
    col_contact_count = pick_column(frame, ['\u4eba\u5458\u6570\u91cf', '\u5b66\u4e60\u573a\u6240\u7684\u4eba\u5458\u6570\u91cf'])
    col_age = pick_column(frame, ['\u5e74\u9f84'])
    trip_frequency_columns = [f'\u9891\u7387{i}' for i in range(1, 6)]
    trip_duration_columns = [f'\u65f6\u957f{i}' for i in range(1, 6)]
    community_clean = frame[col_community].astype(str).str.strip()
    address_clean = frame[col_address].astype(str).str.strip()
    household_key = community_clean + '|' + address_clean
    age_values = frame[col_age].astype(str).str.strip().str.lower().map(AGE_CODE_MAP).fillna(45.0).to_numpy(dtype=float)
    daily_hours_nonfixed = np.zeros(n, dtype=float)
    for frequency_column, duration_column in zip(trip_frequency_columns, trip_duration_columns):
        if frequency_column in frame.columns and duration_column in frame.columns:
            frequency = frame[frequency_column].map(map_frequency).fillna(0.0).to_numpy(dtype=float)
            duration = frame[duration_column].map(map_duration).fillna(0.0).to_numpy(dtype=float)
            daily_hours_nonfixed += frequency * duration / 7.0
    daily_hours_fixed = np.zeros(n, dtype=float)
    is_key_group = np.zeros(n, dtype=bool)
    for index, status in enumerate(frame[col_work_status].to_numpy()):
        if is_employed(status) or is_student(status):
            daily_hours_fixed[index] = 8.0
            is_key_group[index] = True
    mobility_index = np.clip(daily_hours_fixed + daily_hours_nonfixed, 0.0, 12.0)
    virtual_contact_counts = frame[col_contact_count].map(map_virtual_contacts).fillna(0.0).to_numpy(dtype=float)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    household_members: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(household_key.to_numpy()):
        household_members[str(key)].append(index)
    for members in household_members.values():
        size = len(members)
        if size <= 1:
            continue
        weight = HH_CONTACT_HOURS / (size - 1)
        for left in range(size):
            for right in range(left + 1, size):
                graph.add_edge(members[left], members[right], weight=weight, layer='home')
    network_rng = np.random.RandomState(seed)
    all_nodes = np.arange(n)
    for index in range(n):
        internal_social_time = daily_hours_nonfixed[index] * INTERNAL_SOCIAL_RATIO
        if internal_social_time <= 0:
            continue
        encounter_count = int(round(internal_social_time / SOCIAL_ENCOUNTER_DURATION))
        if encounter_count <= 0:
            continue
        candidates = all_nodes[all_nodes != index]
        targets = network_rng.choice(candidates, size=min(encounter_count, n - 1), replace=False)
        for target in targets:
            target_int = int(target)
            if graph.has_edge(index, target_int):
                graph[index][target_int]['weight'] += SOCIAL_ENCOUNTER_DURATION
            else:
                graph.add_edge(index, target_int, weight=SOCIAL_ENCOUNTER_DURATION, layer='community_random')
    if hasattr(nx, 'to_scipy_sparse_array'):
        adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(n), weight='weight', format='csr', dtype=float)
        adjacency = csr_matrix(adjacency)
    else:
        adjacency = nx.to_scipy_sparse_matrix(graph, nodelist=range(n), weight='weight', format='csr', dtype=float)
    eigenvalues, _ = eigsh(adjacency, k=1, which='LA')
    spectral_radius = float(eigenvalues[0])
    if spectral_radius <= 0:
        raise ValueError(f'Non-positive network spectral radius: {spectral_radius}')
    strength_internal = np.asarray(adjacency.sum(axis=1)).ravel()
    strength_virtual_work = virtual_contact_counts * daily_hours_fixed
    external_social_time = daily_hours_nonfixed * (1.0 - INTERNAL_SOCIAL_RATIO)
    strength_virtual_social = external_social_time * EXTERNAL_SOCIAL_CONTACT_COUNT
    strength_virtual = strength_virtual_work + strength_virtual_social
    total_strength = strength_internal + strength_virtual
    average_neighbor_strength = np.zeros(n, dtype=float)
    for index in range(n):
        neighbors = list(graph.neighbors(index))
        if neighbors:
            average_neighbor_strength[index] = float(np.mean(total_strength[neighbors]))
    sum_internal = float(np.sum(strength_internal))
    sum_external = float(np.sum(strength_virtual_work[is_key_group]) * EXTERNAL_PREVALENCE + np.sum(strength_virtual_social[daily_hours_nonfixed > 0]) * EXTERNAL_PREVALENCE)
    internal_exposure_ratio = sum_internal / (sum_internal + sum_external + 1e-09)
    household_codes, _ = pd.factorize(household_key)
    node_metadata = pd.DataFrame({'Node_ID': np.arange(n, dtype=int), 'Anonymous_Household_ID': household_codes.astype(int), 'Age_Group_Value': age_values, 'Work_School_Activity_Status': is_key_group.astype(int), 'Daily_Activity_Duration': mobility_index, 'Internal_Weighted_Strength': strength_internal, 'External_Virtual_Strength': strength_virtual, 'Total_Weighted_Exposure_Strength': total_strength, 'Average_Neighbour_Exposure_Strength': average_neighbor_strength})
    return ModelData(n=n, adjacency=adjacency, spectral_radius=spectral_radius, internal_exposure_ratio=internal_exposure_ratio, age_values=age_values, age_groups=np.array(sorted(np.unique(age_values)), dtype=float), is_key_group=is_key_group, daily_hours_nonfixed=daily_hours_nonfixed, mobility_index=mobility_index, strength_internal=strength_internal, strength_virtual_work=strength_virtual_work, strength_virtual_social=strength_virtual_social, strength_virtual=strength_virtual, total_strength=total_strength, average_neighbor_strength=average_neighbor_strength, node_metadata=node_metadata)

def make_vaccination_masks(model: ModelData, coverage: float, seed: int) -> dict[str, np.ndarray]:
    number_vaccinated = int(coverage * model.n)
    masks: dict[str, np.ndarray] = {}
    random_rng = np.random.RandomState(seed)
    random_indices = random_rng.choice(model.n, number_vaccinated, replace=False)
    high_strength_indices = np.argsort(model.total_strength)[::-1][:number_vaccinated]
    oldest_indices = np.argsort(model.age_values)[::-1][:number_vaccinated]
    for strategy, indices in {'Random': random_indices, 'High Strength (Node)': high_strength_indices, 'Priority: Oldest First': oldest_indices}.items():
        mask = np.zeros(model.n, dtype=bool)
        mask[np.asarray(indices, dtype=int)] = True
        masks[strategy] = mask
    masks['No vaccination'] = np.zeros(model.n, dtype=bool)
    return masks

def calibrate_beta(target_r0: float, recovery_probability: float, model: ModelData) -> float:
    adjusted_r0 = target_r0 * model.internal_exposure_ratio
    return adjusted_r0 * recovery_probability * 24.0 / model.spectral_radius

def current_vaccine_efficacy(day: int, ves_0: float, scenario: dict[str, Any]) -> float:
    if day <= int(scenario['vax_full_days']):
        return ves_0
    if day <= int(scenario['vax_mid_days']):
        return float(scenario['vax_mid_multiplier']) * ves_0
    return float(scenario['vax_long_multiplier']) * ves_0

def simulate_replicate(model: ModelData, scenario: dict[str, Any], ifr_by_node: np.ndarray, beta: float, vaccinated_mask: np.ndarray, ves_0: float, simulation_days: int, initial_infectious: int, run_seed: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rng = np.random.RandomState(run_seed)
    state = np.zeros(model.n, dtype=np.int8)
    days_since_recovery = np.full(model.n, -1, dtype=np.int32)
    ever_infected = np.zeros(model.n, dtype=bool)
    infection_events = np.zeros(model.n, dtype=np.int32)
    seeds = rng.choice(model.n, size=initial_infectious, replace=False)
    state[seeds] = 2
    ever_infected[seeds] = True
    infection_events[seeds] += 1
    external_force = np.where(model.is_key_group, beta * model.strength_virtual_work / 24.0 * EXTERNAL_PREVALENCE, 0.0) + np.where(model.daily_hours_nonfixed > 0, beta * model.strength_virtual_social / 24.0 * EXTERNAL_PREVALENCE, 0.0)
    daily_infectious = np.empty(simulation_days, dtype=np.int32)
    for day_index in range(simulation_days):
        day = day_index + 1
        new_state = state.copy()
        infection_progression_draw = rng.random_sample(model.n)
        recovery_draw = rng.random_sample(model.n)
        vaccine_efficacy = current_vaccine_efficacy(day, ves_0, scenario)
        vaccine_multiplier = np.where(vaccinated_mask, 1.0 - vaccine_efficacy, 1.0)
        natural_multiplier = np.ones(model.n, dtype=float)
        full_days = int(scenario['natural_immunity_full_days'])
        wane_end_day = int(scenario['natural_immunity_wane_end_day'])
        initial_natural_efficacy = float(scenario['natural_immunity_initial'])
        long_term_natural_efficacy = float(scenario['natural_immunity_long_term'])
        full_mask = (days_since_recovery >= 0) & (days_since_recovery <= full_days)
        waning_mask = (days_since_recovery > full_days) & (days_since_recovery <= wane_end_day)
        long_mask = days_since_recovery > wane_end_day
        natural_multiplier[full_mask] = 1.0 - initial_natural_efficacy
        if np.any(waning_mask):
            denominator = max(wane_end_day - full_days, 1)
            fraction = (days_since_recovery[waning_mask] - full_days) / denominator
            current_natural_efficacy = initial_natural_efficacy - (initial_natural_efficacy - long_term_natural_efficacy) * fraction
            natural_multiplier[waning_mask] = 1.0 - current_natural_efficacy
        natural_multiplier[long_mask] = 1.0 - long_term_natural_efficacy
        susceptible_indices = np.flatnonzero(state == 0)
        if susceptible_indices.size:
            infectious_vector = (state == 2).astype(float)
            internal_force = beta / 24.0 * model.adjacency.dot(infectious_vector)
            total_force = internal_force + external_force
            susceptibility = vaccine_multiplier * natural_multiplier
            probability = 1.0 - np.exp(-susceptibility[susceptible_indices] * total_force[susceptible_indices])
            newly_infected_mask = infection_progression_draw[susceptible_indices] < probability
            newly_infected = susceptible_indices[newly_infected_mask]
            new_state[newly_infected] = 1
            ever_infected[newly_infected] = True
            infection_events[newly_infected] += 1
        exposed_indices = np.flatnonzero(state == 1)
        if exposed_indices.size:
            progress = infection_progression_draw[exposed_indices] < float(scenario['prob_e_to_i'])
            new_state[exposed_indices] = np.where(progress, 2, 1)
        infectious_indices = np.flatnonzero(state == 2)
        if infectious_indices.size:
            recovered = recovery_draw[infectious_indices] < float(scenario['prob_i_to_s'])
            recovered_indices = infectious_indices[recovered]
            new_state[recovered_indices] = 0
            days_since_recovery[recovered_indices] = 0
        increment_mask = (new_state == 0) & (days_since_recovery >= 0)
        days_since_recovery[increment_mask] += 1
        days_since_recovery[new_state != 0] = -1
        state = new_state
        daily_infectious[day_index] = int(np.sum(state == 2))
    ever_infected_count = int(np.sum(ever_infected))
    peak_size = int(np.max(daily_infectious)) if simulation_days else 0
    peak_time = int(np.argmax(daily_infectious) + 1) if simulation_days else 0
    expected_deaths = float(np.sum(ifr_by_node[ever_infected]))
    total_infection_events = int(np.sum(infection_events))
    age_rows: list[dict[str, Any]] = []
    for age in model.age_groups:
        group_mask = np.isclose(model.age_values, age)
        population_count = int(np.sum(group_mask))
        for vaccination_status, status_mask in (('All', np.ones(model.n, dtype=bool)), ('Vaccinated', vaccinated_mask), ('Unvaccinated', ~vaccinated_mask)):
            combined = group_mask & status_mask
            age_rows.append({'Age_Group_Value': float(age), 'Age_Group_Label': age_label(float(age)), 'Vaccination_Status': vaccination_status, 'Population_Count': int(np.sum(combined)), 'Ever_Infected_Count': int(np.sum(ever_infected & combined)), 'Infection_Event_Count': int(np.sum(infection_events[combined])), 'Total_Age_Group_Population': population_count})
    all_status_rows = [row for row in age_rows if row['Vaccination_Status'] == 'All']
    if sum((row['Ever_Infected_Count'] for row in all_status_rows)) != ever_infected_count:
        raise AssertionError('Age-group ever-infected counts do not sum to the replicate total')
    if sum((row['Infection_Event_Count'] for row in all_status_rows)) != total_infection_events:
        raise AssertionError('Age-group infection-event counts do not sum to the replicate total')
    metrics = {'Ever Infected Count': float(ever_infected_count), 'Final Infection Proportion': float(ever_infected_count / model.n), 'Epidemic Peak Size': float(peak_size), 'Peak Time': float(peak_time), 'Expected Deaths': expected_deaths, 'Total Infection Events': float(total_infection_events)}
    return (metrics, age_rows)

def summarize_values(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    count = len(array)
    mean_value = float(np.mean(array)) if count else math.nan
    sd_value = float(np.std(array, ddof=1)) if count > 1 else 0.0
    se_value = sd_value / math.sqrt(count) if count else math.nan
    return {prefix: mean_value, f'{prefix} SD': sd_value, f'{prefix} SE': se_value, f'{prefix} CI95 Lower': mean_value - 1.96 * se_value, f'{prefix} CI95 Upper': mean_value + 1.96 * se_value}

def atomic_to_csv(frame: pd.DataFrame, path: Path, compression: str | None=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)

def atomic_to_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)

def task_slug(scenario_name: str, ves_0: float, r0: float) -> str:
    return f'{scenario_name}__VES_{ves_0:g}__R0_{r0:g}'.replace('.', 'p')

def initialize_worker(model: ModelData, scenarios: dict[str, dict[str, Any]], settings: dict[str, Any], vaccination_masks: dict[str, np.ndarray], ifr_by_scenario: dict[str, np.ndarray]) -> None:
    global _WORKER_MODEL, _WORKER_SCENARIOS, _WORKER_SETTINGS, _WORKER_VAX_MASKS, _WORKER_IFR_BY_SCENARIO
    _WORKER_MODEL = model
    _WORKER_SCENARIOS = scenarios
    _WORKER_SETTINGS = settings
    _WORKER_VAX_MASKS = vaccination_masks
    _WORKER_IFR_BY_SCENARIO = ifr_by_scenario

def run_task(task: tuple[str, float, float]) -> dict[str, str]:
    if _WORKER_MODEL is None or _WORKER_SCENARIOS is None or _WORKER_SETTINGS is None or (_WORKER_VAX_MASKS is None) or (_WORKER_IFR_BY_SCENARIO is None):
        raise RuntimeError('Worker was not initialized')
    model = _WORKER_MODEL
    settings = _WORKER_SETTINGS
    scenario_name, ves_0, r0 = task
    scenario = _WORKER_SCENARIOS[scenario_name]
    ifr_by_node = _WORKER_IFR_BY_SCENARIO[scenario_name]
    scenario_ifr_profile = str(settings['scenario_ifr_profiles'][scenario_name])
    slug = task_slug(scenario_name, ves_0, r0)
    checkpoint_dir = Path(settings['checkpoint_dir'])
    summary_path = checkpoint_dir / f'{slug}__summary.csv'
    replicate_path = checkpoint_dir / f'{slug}__replicates.csv.gz'
    age_path = checkpoint_dir / f'{slug}__age_counts.csv.gz'
    manifest_path = checkpoint_dir / f'{slug}__done.json'
    signature_payload = {'scenario_name': scenario_name, 'scenario_parameters': scenario, 'ves_0': float(ves_0), 'r0': float(r0), 'coverage': float(settings['coverage']), 'mc_runs': int(settings['mc_runs']), 'simulation_days': int(settings['simulation_days']), 'initial_infectious': int(settings['initial_infectious']), 'seed': int(settings['seed']), 'ifr_profile': scenario_ifr_profile, 'ifr_values': settings['scenario_ifr_values'][scenario_name], 'population_size': model.n, 'spectral_radius': model.spectral_radius}
    signature = hashlib.sha256(json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    if settings['resume'] and manifest_path.exists() and summary_path.exists() and replicate_path.exists() and age_path.exists():
        existing_manifest = load_json(manifest_path)
        if existing_manifest.get('task_signature') == signature:
            return {'summary': str(summary_path), 'replicates': str(replicate_path), 'age': str(age_path), 'manifest': str(manifest_path), 'status': 'reused'}
    beta = calibrate_beta(r0, float(scenario['prob_i_to_s']), model)
    summary_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    age_count_rows: list[dict[str, Any]] = []
    start_time = time.time()
    coverage_strategy_pairs = [(0.0, 'No vaccination')]
    coverage_strategy_pairs.extend(((float(settings['coverage']), strategy) for strategy in DEFAULT_STRATEGIES))
    for coverage, strategy in coverage_strategy_pairs:
        vaccinated_mask = _WORKER_VAX_MASKS[strategy]
        actual_vaccinated = int(np.sum(vaccinated_mask))
        actual_coverage = actual_vaccinated / model.n
        metric_storage: dict[str, list[float]] = defaultdict(list)
        for mc_run in range(1, int(settings['mc_runs']) + 1):
            run_seed = stable_seed('epidemic', r0, mc_run, base_seed=int(settings['seed']))
            metrics, age_rows = simulate_replicate(model=model, scenario=scenario, ifr_by_node=ifr_by_node, beta=beta, vaccinated_mask=vaccinated_mask, ves_0=ves_0, simulation_days=int(settings['simulation_days']), initial_infectious=int(settings['initial_infectious']), run_seed=run_seed)
            for metric_name, value in metrics.items():
                metric_storage[metric_name].append(float(value))
            identifier = {'Scenario': scenario_name, 'Scenario_Display_Name': scenario['display_name'], 'VES_0': float(ves_0), 'R0': float(r0), 'Coverage': float(coverage), 'Actual_Coverage': float(actual_coverage), 'Actual_Vaccinated_Count': actual_vaccinated, 'Strategy': strategy, 'MC_Run': mc_run, 'Paired_Random_Seed': run_seed}
            replicate_rows.append({**identifier, **metrics, 'IFR_Profile': scenario_ifr_profile})
            for row in age_rows:
                age_count_rows.append({**identifier, **row})
        summary_row: dict[str, Any] = {'Scenario': scenario_name, 'Scenario_Display_Name': scenario['display_name'], 'Scenario_Interpretation': scenario['interpretation'], 'VES_0': float(ves_0), 'R0': float(r0), 'Coverage': float(coverage), 'Actual_Coverage': float(actual_coverage), 'Actual_Vaccinated_Count': actual_vaccinated, 'Strategy': strategy, 'Beta': beta, 'IFR_Profile': scenario_ifr_profile, 'MC_Runs': int(settings['mc_runs']), 'Simulation_Days': int(settings['simulation_days']), 'Initial_Infectious_Nodes': int(settings['initial_infectious'])}
        for metric_name, values in metric_storage.items():
            summary_row.update(summarize_values(values, metric_name))
        summary_rows.append(summary_row)
    summary_frame = pd.DataFrame(summary_rows)
    replicate_frame = pd.DataFrame(replicate_rows)
    age_frame = pd.DataFrame(age_count_rows)
    atomic_to_csv(summary_frame, summary_path)
    atomic_to_csv(replicate_frame, replicate_path, compression='gzip')
    atomic_to_csv(age_frame, age_path, compression='gzip')
    atomic_to_json({'task': {'scenario': scenario_name, 'ves_0': ves_0, 'r0': r0}, 'task_signature': signature, 'summary_rows': len(summary_frame), 'replicate_rows': len(replicate_frame), 'age_rows': len(age_frame), 'elapsed_seconds': time.time() - start_time, 'completed': True}, manifest_path)
    return {'summary': str(summary_path), 'replicates': str(replicate_path), 'age': str(age_path), 'manifest': str(manifest_path), 'status': 'computed'}

def compute_figure_source(summary: pd.DataFrame, target_coverage: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ['Scenario', 'Scenario_Display_Name', 'VES_0', 'R0']
    for keys, group in summary.groupby(group_columns, sort=False):
        key_values = dict(zip(group_columns, keys))
        no_vaccination = group[np.isclose(group['Coverage'], 0.0)]
        target = group[np.isclose(group['Coverage'], target_coverage)]
        random_rows = target[target['Strategy'] == 'Random']
        if no_vaccination.empty or random_rows.empty:
            continue
        y0_row = no_vaccination.iloc[0]
        random_row = random_rows.iloc[0]
        for _, strategy_row in target.iterrows():
            strategy = str(strategy_row['Strategy'])
            if strategy == 'Random':
                continue
            for outcome in OUTCOME_COLUMNS:
                y0 = float(y0_row[outcome])
                y_random = float(random_row[outcome])
                y_strategy = float(strategy_row[outcome])
                delta = math.nan if y0 == 0 else 100.0 * (y_random - y_strategy) / y0
                rows.append({**key_values, 'Coverage': target_coverage, 'Strategy': strategy, 'Outcome': outcome, 'Outcome_Display_Name': OUTCOME_LABELS[outcome], 'Y0_No_Vaccination': y0, 'Y_Random': y_random, 'Y_Strategy': y_strategy, 'Raw_Difference_Random_Minus_Strategy': y_random - y_strategy, 'Additional_Reduction_Percent': delta, 'Improves_Random': bool(delta > 0) if not math.isnan(delta) else None})
    return pd.DataFrame(rows)

def compute_paired_differences(replicates: pd.DataFrame, summary: pd.DataFrame, target_coverage: float) -> pd.DataFrame:
    y0_lookup: dict[tuple[str, float, float, str], float] = {}
    for _, row in summary[np.isclose(summary['Coverage'], 0.0)].iterrows():
        for outcome in OUTCOME_COLUMNS:
            y0_lookup[str(row['Scenario']), float(row['VES_0']), float(row['R0']), outcome] = float(row[outcome])
    target = replicates[np.isclose(replicates['Coverage'], target_coverage)].copy()
    identifier_columns = ['Scenario', 'Scenario_Display_Name', 'VES_0', 'R0', 'MC_Run']
    rows: list[dict[str, Any]] = []
    for keys, group in target.groupby(identifier_columns, sort=False):
        identifier = dict(zip(identifier_columns, keys))
        random_rows = group[group['Strategy'] == 'Random']
        if random_rows.empty:
            continue
        random_row = random_rows.iloc[0]
        for _, strategy_row in group.iterrows():
            strategy = str(strategy_row['Strategy'])
            if strategy == 'Random':
                continue
            for outcome in OUTCOME_COLUMNS:
                y_random = float(random_row[outcome])
                y_strategy = float(strategy_row[outcome])
                y0 = y0_lookup[str(identifier['Scenario']), float(identifier['VES_0']), float(identifier['R0']), outcome]
                rows.append({**identifier, 'Coverage': target_coverage, 'Strategy': strategy, 'Outcome': outcome, 'Y0_Mean': y0, 'Y_Random_Replicate': y_random, 'Y_Strategy_Replicate': y_strategy, 'Paired_Raw_Difference': y_random - y_strategy, 'Paired_Additional_Reduction_Percent': math.nan if y0 == 0 else 100.0 * (y_random - y_strategy) / y0})
    return pd.DataFrame(rows)

def save_metadata(output_dir: Path, model: ModelData, scenarios: dict[str, dict[str, Any]], settings: dict[str, Any], vaccination_masks: dict[str, np.ndarray], ifr_profiles: dict[str, dict[float, float]], ifr_descriptions: dict[str, str], scenario_ifr_profiles: dict[str, str], data_file: Path) -> None:
    metadata_dir = output_dir / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    atomic_to_csv(model.node_metadata, metadata_dir / 'network_node_attributes.csv.gz', compression='gzip')
    vaccination_rows = []
    for strategy in DEFAULT_STRATEGIES:
        mask = vaccination_masks[strategy]
        for node_id in np.flatnonzero(mask):
            vaccination_rows.append({'Node_ID': int(node_id), 'Coverage': float(settings['coverage']), 'Strategy': strategy, 'Vaccinated': 1})
    atomic_to_csv(pd.DataFrame(vaccination_rows), metadata_dir / 'vaccination_assignments.csv.gz', compression='gzip')
    scenario_rows = []
    for name, parameters in scenarios.items():
        scenario_rows.append({'Scenario': name, **parameters})
    atomic_to_csv(pd.DataFrame(scenario_rows), metadata_dir / 'scenario_parameters.csv')
    age_population_rows = []
    for scenario_name, profile_name in scenario_ifr_profiles.items():
        ifr_map = ifr_profiles[profile_name]
        mapped_ifr = ifr_mapping_for_age_groups(model.age_groups, ifr_map)
        for age in model.age_groups:
            age_population_rows.append({'Scenario': scenario_name, 'IFR_Profile': profile_name, 'Age_Group_Value': float(age), 'Age_Group_Label': age_label(float(age)), 'Population_Count': int(np.sum(np.isclose(model.age_values, age))), 'Scenario_IFR': mapped_ifr[age_label(float(age))]})
    atomic_to_csv(pd.DataFrame(age_population_rows), metadata_dir / 'age_group_population_and_ifr.csv')
    ifr_source_rows = []
    for profile_name, ifr_map in ifr_profiles.items():
        for age, value in sorted(ifr_map.items()):
            ifr_source_rows.append({'IFR_Profile': profile_name, 'Age_Group_Value': float(age), 'Age_Group_Label': age_label(float(age)), 'IFR': float(value), 'Description': ifr_descriptions.get(profile_name, '')})
    atomic_to_csv(pd.DataFrame(ifr_source_rows), metadata_dir / 'all_ifr_profiles.csv')
    data_fingerprint = hashlib.sha256(data_file.read_bytes()).hexdigest()
    run_configuration = {**settings, 'data_file_name': data_file.name, 'data_file_sha256': data_fingerprint, 'population_size': model.n, 'spectral_radius': model.spectral_radius, 'internal_exposure_ratio': model.internal_exposure_ratio, 'external_prevalence': EXTERNAL_PREVALENCE, 'strategies': DEFAULT_STRATEGIES, 'scenario_ifr_profiles': scenario_ifr_profiles, 'ifr_descriptions': ifr_descriptions, 'model_scope_note': 'Disease A and Disease B vary disease-course and immunity parameters only. The exposure network, age grouping, allocation rules, external prevalence, and vaccination strategies are held fixed. Expected deaths use the scenario-specific IFR profile recorded for each scenario.'}
    for key, value in list(run_configuration.items()):
        if isinstance(value, Path):
            run_configuration[key] = str(value)
    atomic_to_json(run_configuration, metadata_dir / 'run_configuration.json')

def merge_results(task_outputs: list[dict[str, str]], output_dir: Path, coverage: float) -> None:
    summaries = [pd.read_csv(item['summary']) for item in task_outputs]
    replicates = [pd.read_csv(item['replicates']) for item in task_outputs]
    age_counts = [pd.read_csv(item['age']) for item in task_outputs]
    summary = pd.concat(summaries, ignore_index=True).sort_values(['Scenario', 'VES_0', 'R0', 'Coverage', 'Strategy'])
    replicate = pd.concat(replicates, ignore_index=True).sort_values(['Scenario', 'VES_0', 'R0', 'Coverage', 'Strategy', 'MC_Run'])
    age = pd.concat(age_counts, ignore_index=True).sort_values(['Scenario', 'VES_0', 'R0', 'Coverage', 'Strategy', 'MC_Run', 'Age_Group_Value', 'Vaccination_Status'])
    figure_source = compute_figure_source(summary, target_coverage=coverage)
    paired_differences = compute_paired_differences(replicate, summary, target_coverage=coverage)
    atomic_to_csv(summary, output_dir / 'scenario_summary.csv.gz', compression='gzip')
    atomic_to_csv(replicate, output_dir / 'replicate_outcomes.csv.gz', compression='gzip')
    atomic_to_csv(age, output_dir / 'age_group_infections.csv.gz', compression='gzip')
    atomic_to_csv(figure_source, output_dir / 'figure_source_data.csv')
    atomic_to_csv(paired_differences, output_dir / 'paired_strategy_differences.csv.gz', compression='gzip')
    with pd.ExcelWriter(output_dir / 'disease_robustness_summary.xlsx') as writer:
        summary.to_excel(writer, sheet_name='Scenario_Summary', index=False)
        figure_source.to_excel(writer, sheet_name='Figure_Source_Data', index=False)
        qualitative = figure_source.groupby(['Scenario', 'Strategy', 'Outcome'], as_index=False).agg(Parameter_Combinations=('Additional_Reduction_Percent', 'count'), Mean_Additional_Reduction=('Additional_Reduction_Percent', 'mean'), Median_Additional_Reduction=('Additional_Reduction_Percent', 'median'), Fraction_Improves_Random=('Improves_Random', 'mean'))
        qualitative.to_excel(writer, sheet_name='Qualitative_Summary', index=False)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file', type=Path, default=SCRIPT_DIR.parent / 'data' / '20231012.xlsx', help='Path to 20231012.xlsx; default: data/20231012.xlsx.')
    parser.add_argument('--sheet', default='0', help='Excel sheet index or name; default: 0')
    parser.add_argument('--scenario-config', type=Path, default=None)
    parser.add_argument('--ifr-config', type=Path, default=None)
    parser.add_argument('--ifr-profile', default='Reference_IFR')
    parser.add_argument('--common-ifr-profile', default=None, help='Optional override: use one IFR profile for all scenarios.')
    parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR.parent / 'results' / 'disease_sensitivity')
    parser.add_argument('--scenarios', nargs='+', default=['Baseline', 'Disease_A', 'Disease_B'])
    parser.add_argument('--r0-mode', choices=['common-grid', 'scenario-reference'], default='scenario-reference', help="Use the shared R0 grid or each scenario's literature reference R0.")
    parser.add_argument('--r0-values', nargs='+', type=float, default=DEFAULT_R0_VALUES)
    parser.add_argument('--ves-values', nargs='+', type=float, default=DEFAULT_VES_VALUES)
    parser.add_argument('--coverage', type=float, default=0.5)
    parser.add_argument('--mc-runs', type=int, default=100)
    parser.add_argument('--simulation-days', type=int, default=500)
    parser.add_argument('--initial-infectious', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    default_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', min(20, max(1, (os.cpu_count() or 2) - 1))))
    parser.add_argument('--workers', type=int, default=default_workers)
    parser.add_argument('--no-resume', action='store_true', help='Recompute task checkpoints even when completed files exist.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if not 0.0 < args.coverage < 1.0:
        raise ValueError('--coverage must be between 0 and 1')
    if args.mc_runs < 1 or args.simulation_days < 1 or args.initial_infectious < 1:
        raise ValueError('MC runs, simulation days, and initial infectious count must be positive')
    data_file = resolve_existing_path(args.data_file, 'Input workbook')
    all_scenarios, scenario_config_source = load_json_or_default(args.scenario_config, 'Scenario configuration', DEFAULT_SCENARIO_PARAMETERS)
    missing_scenarios = sorted(set(args.scenarios) - set(all_scenarios))
    if missing_scenarios:
        raise KeyError(f'Scenarios not found in configuration: {missing_scenarios}')
    scenarios = {name: all_scenarios[name] for name in args.scenarios}
    validate_scenarios(scenarios)
    ifr_payload, ifr_config_source = load_json_or_default(args.ifr_config, 'IFR configuration', DEFAULT_IFR_CONFIG)
    ifr_profiles, ifr_descriptions, scenario_profile = load_ifr_profiles_from_payload(ifr_payload)
    scenario_ifr_profiles = resolve_scenario_ifr_profiles(scenarios=scenarios, profiles=ifr_profiles, scenario_profile=scenario_profile, fallback_profile=args.ifr_profile, common_profile=args.common_ifr_profile)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f'[INFO] Building survey-derived network from {data_file}', flush=True)
    model = build_model(data_file, args.sheet, args.seed)
    if args.initial_infectious > model.n:
        raise ValueError('Initial infectious count exceeds population size')
    vaccination_masks = make_vaccination_masks(model, args.coverage, args.seed)
    ifr_by_scenario = {scenario_name: make_ifr_by_node(model.age_values, ifr_profiles[profile_name]) for scenario_name, profile_name in scenario_ifr_profiles.items()}
    scenario_ifr_values = {scenario_name: ifr_mapping_for_age_groups(model.age_groups, ifr_profiles[profile_name]) for scenario_name, profile_name in scenario_ifr_profiles.items()}
    if args.r0_mode == 'common-grid':
        r0_values_by_scenario = {scenario_name: [float(value) for value in args.r0_values] for scenario_name in scenarios}
    else:
        r0_values_by_scenario = {scenario_name: [float(scenario['r0_reference'])] for scenario_name, scenario in scenarios.items()}
    settings = {'coverage': float(args.coverage), 'mc_runs': int(args.mc_runs), 'simulation_days': int(args.simulation_days), 'initial_infectious': int(args.initial_infectious), 'seed': int(args.seed), 'workers': int(args.workers), 'r0_mode': args.r0_mode, 'r0_values': [float(value) for value in args.r0_values], 'r0_values_by_scenario': r0_values_by_scenario, 'ves_values': [float(value) for value in args.ves_values], 'scenario_names': list(scenarios), 'fallback_ifr_profile': str(args.ifr_profile), 'common_ifr_profile': args.common_ifr_profile, 'scenario_ifr_profiles': scenario_ifr_profiles, 'scenario_ifr_values': scenario_ifr_values, 'scenario_config_source': scenario_config_source, 'ifr_config_source': ifr_config_source, 'checkpoint_dir': str(checkpoint_dir), 'resume': not args.no_resume}
    save_metadata(output_dir=output_dir, model=model, scenarios=scenarios, settings=settings, vaccination_masks=vaccination_masks, ifr_profiles=ifr_profiles, ifr_descriptions=ifr_descriptions, scenario_ifr_profiles=scenario_ifr_profiles, data_file=data_file)
    tasks = [(scenario, float(ves_0), float(r0)) for scenario in scenarios for ves_0 in args.ves_values for r0 in r0_values_by_scenario[scenario]]
    print(f'[INFO] Population={model.n}; tasks={len(tasks)}; MC/task/strategy={args.mc_runs}; days={args.simulation_days}; workers={args.workers}', flush=True)
    task_outputs: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    start_time = time.time()
    if args.workers == 1:
        initialize_worker(model, scenarios, settings, vaccination_masks, ifr_by_scenario)
        for index, task in enumerate(tasks, start=1):
            try:
                result = run_task(task)
                task_outputs.append(result)
                print(f'[DONE {index}/{len(tasks)}] {task} ({result['status']})', flush=True)
            except Exception as exc:
                failures.append({'task': repr(task), 'error': repr(exc)})
                print(f'[ERROR] {task}: {exc}', file=sys.stderr, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)), initializer=initialize_worker, initargs=(model, scenarios, settings, vaccination_masks, ifr_by_scenario)) as executor:
            futures = {executor.submit(run_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    result = future.result()
                    task_outputs.append(result)
                    print(f'[DONE {index}/{len(tasks)}] {task} ({result['status']})', flush=True)
                except Exception as exc:
                    failures.append({'task': repr(task), 'error': repr(exc)})
                    print(f'[ERROR] {task}: {exc}', file=sys.stderr, flush=True)
    atomic_to_json({'completed_tasks': len(task_outputs), 'failed_tasks': failures, 'expected_tasks': len(tasks), 'elapsed_seconds': time.time() - start_time}, output_dir / 'run_status.json')
    if failures:
        raise RuntimeError(f'{len(failures)} tasks failed. Check {output_dir / 'run_status.json'} and rerun to resume.')
    merge_results(task_outputs, output_dir, coverage=args.coverage)
    print(f'[COMPLETE] Results saved to {output_dir}', flush=True)
if __name__ == '__main__':
    main()
