# Adapting the simulation to a custom weighted network

## Purpose and scope

The manuscript analyses use a survey-derived weighted exposure network. The
production scripts construct that network directly from the authorized survey
workbook. This guide describes the minimum network and node-level information
needed to adapt the primary simulation to another weighted network.

A custom network changes the model input and is therefore a new analysis. Its
results should not be presented as reproducing the manuscript estimates.

## Example generator

Run:

```bash
python examples/generate_custom_network_example.py
```

The script creates two synthetic files under:

```text
examples/custom_network_example/
```

- `nodes.csv`
- `edges.csv`

The generated records contain no participant data.

## Node table

The example node table contains one row per node and the following fields:

| Column | Meaning |
|---|---|
| `node_id` | Consecutive integer identifier from 0 to N-1 |
| `age` | Numeric age or age-group representative value used for expected deaths |
| `household_id` | Anonymous household identifier used by household-priority strategies |
| `work_school_activity_status` | Binary indicator for work/school activity |
| `daily_activity_duration` | Daily activity duration used by mobility-based ranking |
| `external_work_strength` | External work/school exposure contribution |
| `external_social_strength` | External social exposure contribution |

## Edge table

The edge table contains one row per undirected internal-network edge:

| Column | Meaning |
|---|---|
| `source` | Source node identifier |
| `target` | Target node identifier |
| `weight` | Positive weighted exposure value |
| `layer` | Descriptive layer label, such as `household` or `community` |

Self-loops should be removed unless they have an explicit interpretation in a
new model. Duplicate undirected edges should be aggregated before simulation.

## Internal objects required by the primary simulation

Before the calibration and strategy functions are called,
`simulations/run_primary_vaccination_experiments.py` expects the following
objects:

```python
N                       # number of nodes
G                       # undirected NetworkX graph with positive edge weights
df["age_num"]           # numeric ages
df["_household_key"]    # anonymous household identifiers
is_key_group            # Boolean work/school indicator
mobility_index          # daily activity duration
daily_hours_nonfixed    # non-fixed out-of-home activity duration
strength_vir_work       # external work/school exposure strength
strength_vir_social     # external social exposure strength
strength_virtual        # strength_vir_work + strength_vir_social
strength_internal       # weighted degree in G
total_node_strength     # strength_internal + strength_virtual
avg_neighbor_strength   # mean total strength of internal-network neighbours
```

The following loading pattern illustrates how the synthetic `nodes.csv` and
`edges.csv` files can be converted into those objects:

```python
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

input_dir = Path("examples/custom_network_example")
nodes = pd.read_csv(input_dir / "nodes.csv").sort_values("node_id")
edges = pd.read_csv(input_dir / "edges.csv")

expected_ids = np.arange(len(nodes), dtype=int)
if not np.array_equal(nodes["node_id"].to_numpy(dtype=int), expected_ids):
    raise ValueError("node_id values must be consecutive integers from 0 to N-1")

G = nx.from_pandas_edgelist(
    edges,
    source="source",
    target="target",
    edge_attr=["weight", "layer"],
    create_using=nx.Graph,
)
G.add_nodes_from(expected_ids.tolist())

N = len(nodes)
df = pd.DataFrame(
    {
        "age_num": nodes["age"].to_numpy(dtype=float),
        "_household_key": nodes["household_id"].astype(str),
    }
)
is_key_group = nodes["work_school_activity_status"].astype(bool).to_numpy()
mobility_index = nodes["daily_activity_duration"].to_numpy(dtype=float)
strength_vir_work = nodes["external_work_strength"].to_numpy(dtype=float)
strength_vir_social = nodes["external_social_strength"].to_numpy(dtype=float)
strength_virtual = strength_vir_work + strength_vir_social
strength_internal = np.array(
    [G.degree(node_id, weight="weight") for node_id in range(N)],
    dtype=float,
)
total_node_strength = strength_internal + strength_virtual
daily_hours_nonfixed = (strength_vir_social > 0).astype(float)

avg_neighbor_strength = np.zeros(N, dtype=float)
for node_id in range(N):
    neighbours = list(G.neighbors(node_id))
    if neighbours:
        avg_neighbor_strength[node_id] = np.mean(
            total_node_strength[np.asarray(neighbours, dtype=int)]
        )
```

## Integration procedure

1. Copy the primary simulation script to a clearly named adaptation script,
   such as `simulations/run_custom_network_experiment.py`.
2. Replace only the survey loading and survey-based network-construction block
   with the loading pattern above.
3. Retain the calibration, vaccine-protection, transmission, strategy, outcome,
   and output functions unchanged unless the new scientific question requires a
   documented model change.
4. Record the custom network provenance, edge-weight units, node-attribute
   definitions, and any modifications to external exposure strength.
5. Use a distinct output directory so that custom-network results cannot be
   confused with the manuscript results.
6. Validate node count, edge count, weighted-strength distribution, connected
   components, and spectral radius before running the full parameter sweep.

## Important methodological caution

Edge weights in this project are interpreted as exposure-duration equivalents.
A custom network whose weights represent counts, probabilities, distances, or
unnormalized similarities must not be inserted without a defensible conversion
or recalibration. The spectral-radius calibration does not by itself guarantee
that differently scaled edge weights have the same epidemiological meaning.
