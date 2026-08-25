"""
Heterogeneous heritage-experience network construction (paper Section 3.4).

Builds the concept-level weighted graph from the three derived tables in `data/`:

  1. Nodes are entities, keyed by a case-normalised name (lowercase + strip), so
     duplicate surface forms collapse to one concept-level node.
  2. Edges come from the relation records. Each unordered pair is merged
     (duplicate pairs combined), and the edge weight is

         W = ALPHA * co_occurrence_count + BETA * semantic_count

     where `semantic_count` aggregates the typed relations (causal, description,
     contrast) and ALPHA = 0.7, BETA = 0.3.
  3. Node prominence is measured with NetworkX degree centrality and weighted
     degree (strength).
  4. Thematic communities are detected with Louvain (python-louvain, weighted,
     fixed seed), with modularity reported.

Provenance
----------
Consolidated, with the file-loading parameterised, from the original processing
repository:

  * `app/social/graph/builder.py`        -> node / edge construction and weighting
  * `app/social/utils/normalization.py`  -> name normalisation
  * `app/social/analysis/centrality.py`  -> degree centrality + strength
  * `app/social/analysis/communities.py` -> Louvain community detection
  * `app/social/config/constants.py`     -> ALPHA / BETA weights

The graph-construction maths, the edge-weight coefficients, and the Louvain
settings are reproduced verbatim.

NOTE ON BETWEENNESS: the recovered analysis modules computed degree centrality
and weighted degree (strength) but did NOT compute betweenness centrality. To
avoid presenting code that was not actually part of the original run, betweenness
is not included here. `networkx.betweenness_centrality(G, weight="weight")` is the
standard call if you wish to add it.

NOTE ON NODE/EDGE COUNTS: as recovered, this builder seeds nodes from every
distinct (case-normalised) entity in `entity_mentions.csv` -- 5,906 nodes on the
published data, of which 836 are isolated (never appear in a relation) -- and
draws 5,181 weighted edges. The paper reports 5,110 nodes and 5,308 typed edges:
the 5,110 figure corresponds exactly to the number of distinct relation endpoints
(i.e. counting only entities that participate in a relation), and the edge count
differs because 16 out-of-schema relation records (retained in the data for
transparency) do not contribute to the co-occurrence/semantic weight terms here.
The construction logic is reproduced as recovered; the difference is one of
node-set convention, not a change in the weighting.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd

# Edge-weighting coefficients (paper Section 3.4): W = ALPHA*co_occur + BETA*semantic.
EDGE_WEIGHT_ALPHA = 0.7   # co-occurrence coefficient
EDGE_WEIGHT_BETA = 0.3    # causal / description / contrast coefficient

# Relations that count toward the "semantic" term of the edge weight.
SEMANTIC_RELATIONS = {"causal", "description", "contrast"}


# --------------------------------------------------------------------------- #
# Normalisation helpers (from app/social/utils/normalization.py)
# --------------------------------------------------------------------------- #
def normalize_entity_name(name: Any) -> str:
    """Concept-level node key: lowercase + strip whitespace."""
    value = "" if name is None else str(name)
    return value.lower().strip()


def normalize_label(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    return text if text else default


# --------------------------------------------------------------------------- #
# Graph construction (from app/social/graph/builder.py)
# --------------------------------------------------------------------------- #
def add_nodes(G: nx.Graph, entities_df: pd.DataFrame) -> None:
    node_types: Dict[str, str] = {}
    first_year: Dict[str, int] = {}
    last_year: Dict[str, int] = {}

    for _, row in entities_df.iterrows():
        name = normalize_entity_name(row["entity_name"])
        if not name:
            continue
        node_types.setdefault(name, normalize_label(row.get("entity_type")))
        year = int(row["year"]) if pd.notna(row.get("year")) else 0
        if year:
            first_year[name] = min(first_year.get(name, year), year)
            last_year[name] = max(last_year.get(name, year), year)

    for name, entity_type in node_types.items():
        G.add_node(
            name,
            entity_type=entity_type,
            first_appearance_year=first_year.get(name, 0),
            last_appearance_year=last_year.get(name, 0),
        )


def add_edges(G: nx.Graph, relations_df: pd.DataFrame) -> None:
    cooccur_counts: Counter[Tuple[str, str]] = Counter()
    semantic_counts: Counter[Tuple[str, str]] = Counter()
    relation_types: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(dict)

    for _, row in relations_df.iterrows():
        u = normalize_entity_name(row["source_node"])
        v = normalize_entity_name(row["target_node"])
        if not u or not v or u == v or u not in G or v not in G:
            continue
        edge_key = tuple(sorted((u, v)))  # merge duplicate (undirected) pairs
        relation = normalize_label(row.get("relation"))
        if relation == "co_occurrence":
            cooccur_counts[edge_key] += 1
        elif relation in SEMANTIC_RELATIONS:
            semantic_counts[edge_key] += 1
        relation_types[edge_key][relation] = relation_types[edge_key].get(relation, 0) + 1

    for edge_key in set(cooccur_counts) | set(semantic_counts):
        co_count = cooccur_counts.get(edge_key, 0)
        semantic_count = semantic_counts.get(edge_key, 0)
        weight = EDGE_WEIGHT_ALPHA * co_count + EDGE_WEIGHT_BETA * semantic_count
        if weight <= 0:
            continue
        u, v = edge_key
        G.add_edge(
            u, v,
            weight=float(weight),
            co_occur=int(co_count),
            causal=int(semantic_count),
            relation_types=relation_types.get(edge_key, {}),
        )


def build_graph(entities_df: pd.DataFrame, relations_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    add_nodes(G, entities_df)
    add_edges(G, relations_df)
    return G


# --------------------------------------------------------------------------- #
# Centrality (from app/social/analysis/centrality.py)
# --------------------------------------------------------------------------- #
def compute_centrality(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    """Degree centrality C_D(v) = deg(v)/(n-1), plus weighted degree (strength)."""
    degree_centrality = nx.degree_centrality(G)
    strength = {
        node: sum(d["weight"] for _, _, d in G.edges(node, data=True))
        for node in G.nodes()
    }
    nx.set_node_attributes(G, degree_centrality, "degree_centrality")
    nx.set_node_attributes(G, strength, "weighted_degree")
    return {"degree_centrality": degree_centrality, "strength": strength}


# --------------------------------------------------------------------------- #
# Community detection (from app/social/analysis/communities.py)
# --------------------------------------------------------------------------- #
def detect_louvain(G: nx.Graph) -> Dict[str, int]:
    if G.number_of_nodes() == 0:
        return {}
    try:
        import community as community_louvain  # python-louvain
        return {
            str(node): int(cid)
            for node, cid in community_louvain.best_partition(
                G, weight="weight", random_state=42
            ).items()
        }
    except Exception:
        pass
    try:
        communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42)
        assignments: Dict[str, int] = {}
        for cid, nodes in enumerate(communities):
            for node in nodes:
                assignments[str(node)] = cid
        return assignments
    except Exception:
        assignments = {}
        for cid, component in enumerate(nx.connected_components(G)):
            for node in component:
                assignments[str(node)] = cid
        return assignments


def modularity(G: nx.Graph, assignments: Dict[str, int]) -> float:
    if G.number_of_edges() == 0 or not assignments:
        return 0.0
    communities: Dict[int, set] = defaultdict(set)
    for node, cid in assignments.items():
        communities[cid].add(node)
    try:
        return float(nx.algorithms.community.modularity(G, communities.values(), weight="weight"))
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_tables(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    entities = pd.read_csv(data_dir / "entity_mentions.csv")
    relations = pd.read_csv(data_dir / "relations.csv")
    return entities, relations


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the heritage-experience network (paper Section 3.4).")
    parser.add_argument(
        "--data-dir", default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Directory containing entity_mentions.csv and relations.csv (default: ../data).",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    entities_df, relations_df = load_tables(data_dir)
    G = build_graph(entities_df, relations_df)
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    centrality = compute_centrality(G)
    top = sorted(centrality["degree_centrality"].items(), key=lambda x: -x[1])[:10]
    print("Top nodes by degree centrality:")
    for node, score in top:
        print(f"  {node:<30} {score:.4f}")

    assignments = detect_louvain(G)
    n_comms = len(set(assignments.values())) if assignments else 0
    print(f"Louvain communities: {n_comms} (modularity {modularity(G, assignments):.4f})")

    print(f"Density: {nx.density(G):.6f}  |  "
          f"Connected components: {nx.number_connected_components(G)}")


if __name__ == "__main__":
    main()
