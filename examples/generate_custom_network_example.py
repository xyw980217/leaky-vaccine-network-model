"""Generate a small weighted custom-network example and validate its schema.

The files produced by this script are synthetic and contain no participant data.
They are intended to illustrate the node and edge information required when
adapting the simulation workflow to a user-supplied weighted network.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import networkx as nx
import numpy as np
import pandas as pd


REQUIRED_NODE_COLUMNS = {
    "node_id",
    "age",
    "household_id",
    "work_school_activity_status",
    "daily_activity_duration",
    "external_work_strength",
    "external_social_strength",
}
REQUIRED_EDGE_COLUMNS = {"source", "target", "weight", "layer"}


def generate_example(number_of_nodes: int = 30, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if number_of_nodes < 12:
        raise ValueError("number_of_nodes must be at least 12.")

    rng = np.random.default_rng(seed)
    node_ids = np.arange(number_of_nodes, dtype=int)
    household_ids = node_ids // 3
    ages = np.resize(np.array([15.0, 24.0, 34.5, 44.5, 54.5, 64.5, 75.0]), number_of_nodes)
    key_status = (node_ids % 3 != 2).astype(int)
    activity_duration = np.clip(4.0 + (node_ids % 6) * 1.2, 0.0, 12.0)

    nodes = pd.DataFrame(
        {
            "node_id": node_ids,
            "age": ages,
            "household_id": household_ids,
            "work_school_activity_status": key_status,
            "daily_activity_duration": activity_duration,
            "external_work_strength": key_status * (6.0 + node_ids % 5),
            "external_social_strength": 2.0 + (node_ids % 4),
        }
    )

    graph = nx.Graph()
    graph.add_nodes_from(node_ids.tolist())

    for household_id, members in nodes.groupby("household_id")["node_id"]:
        member_list = members.tolist()
        if len(member_list) > 1:
            weight = 12.0 / (len(member_list) - 1)
            for i, source in enumerate(member_list):
                for target in member_list[i + 1 :]:
                    graph.add_edge(source, target, weight=weight, layer="household")

    for source in node_ids:
        candidate_targets = node_ids[node_ids != source]
        targets = rng.choice(candidate_targets, size=min(2, len(candidate_targets)), replace=False)
        for target in targets:
            source_int = int(source)
            target_int = int(target)
            if graph.has_edge(source_int, target_int):
                graph[source_int][target_int]["weight"] += 0.25
            else:
                graph.add_edge(source_int, target_int, weight=0.25, layer="community")

    edges = nx.to_pandas_edgelist(graph)[["source", "target", "weight", "layer"]]
    edges = edges.sort_values(["source", "target"]).reset_index(drop=True)
    return nodes, edges


def validate_schema(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    missing_node_columns = REQUIRED_NODE_COLUMNS.difference(nodes.columns)
    missing_edge_columns = REQUIRED_EDGE_COLUMNS.difference(edges.columns)
    if missing_node_columns:
        raise ValueError(f"Missing node columns: {sorted(missing_node_columns)}")
    if missing_edge_columns:
        raise ValueError(f"Missing edge columns: {sorted(missing_edge_columns)}")

    node_ids = nodes["node_id"].astype(int).to_numpy()
    expected_ids = np.arange(len(nodes), dtype=int)
    if not np.array_equal(np.sort(node_ids), expected_ids):
        raise ValueError("node_id values must be consecutive integers from 0 to N-1.")

    valid_ids = set(node_ids.tolist())
    edge_ids = set(edges["source"].astype(int)).union(set(edges["target"].astype(int)))
    if not edge_ids.issubset(valid_ids):
        raise ValueError("The edge list contains node identifiers absent from nodes.csv.")
    if (pd.to_numeric(edges["weight"], errors="coerce") <= 0).any():
        raise ValueError("All edge weights must be positive.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic custom-network example."
    )
    parser.add_argument("--nodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "custom_network_example",
    )
    args = parser.parse_args()

    nodes, edges = generate_example(args.nodes, args.seed)
    validate_schema(nodes, edges)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = args.output_dir / "nodes.csv"
    edges_path = args.output_dir / "edges.csv"
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)

    graph = nx.from_pandas_edgelist(
        edges,
        source="source",
        target="target",
        edge_attr=["weight", "layer"],
        create_using=nx.Graph,
    )
    weighted_strength = np.array(
        [graph.degree(node_id, weight="weight") for node_id in range(len(nodes))],
        dtype=float,
    )

    print(f"[SAVED] {nodes_path}")
    print(f"[SAVED] {edges_path}")
    print(f"[CHECK] Nodes: {len(nodes)}")
    print(f"[CHECK] Edges: {len(edges)}")
    print(f"[CHECK] Mean internal weighted strength: {weighted_strength.mean():.3f}")


if __name__ == "__main__":
    main()
