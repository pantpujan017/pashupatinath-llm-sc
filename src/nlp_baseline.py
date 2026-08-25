"""
Traditional NLP baseline (paper Section 3.2).

Reproduces the non-LLM comparison baseline:

  * Named-entity recognition with NLTK's MaxEnt chunker (`ne_chunk`).
  * Document-level sentiment with VADER, using the standard +/-0.05 compound
    thresholds to assign positive / neutral / negative.
  * A co-occurrence graph in which every pair of entities found in the same
    review is linked (edge weight = number of co-occurring reviews).

Provenance
----------
This module is a faithful consolidation of the baseline cells of the original
`nlp-vs-llm-comparison` notebook (the same notebook is shipped under
`notebooks/nlp_vs_llm_comparison.ipynb`). The extraction, sentiment and
graph-construction logic are reproduced verbatim; only the file-loading has been
parameterised so that no path points at redistributed raw review text.

Raw review text is NOT included in this repository (see the repository README and
the data provenance note). To run the NER + VADER half of the baseline you must
supply your own file of review text via `--reviews-csv`; it needs a text column
(e.g. `text_clean`) and, ideally, a `review_id` column. The LLM side of the
comparison uses the derived tables published in `data/`.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import entropy as scipy_entropy

# VADER compound-score thresholds for document-level polarity (paper Section 3.2).
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

# Candidate names for the free-text column holding review text.
TEXT_COLUMN_CANDIDATES = [
    "text_clean", "text", "review", "review_text",
    "content", "comment", "review_body", "body",
]


def ensure_nltk_data() -> None:
    """Download the NLTK resources the baseline relies on (idempotent)."""
    for pkg in [
        "vader_lexicon", "stopwords", "punkt", "punkt_tab",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker", "maxent_ne_chunker_tab", "words",
    ]:
        nltk.download(pkg, quiet=True)


def detect_text_column(df: pd.DataFrame) -> str:
    """Pick the best text column for NER: longest average, most word-like."""
    for col in TEXT_COLUMN_CANDIDATES:
        if col in df.columns:
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > 20:
                return col
    obj_cols = [
        (c, df[c].dropna().astype(str).str.len().mean())
        for c in df.columns if df[c].dtype == object
    ]
    return max(obj_cols, key=lambda x: x[1])[0] if obj_cols else df.columns[0]


def extract_ner_entities(text: str):
    """NLTK MaxEnt named-entity recognition for one document."""
    try:
        tokens = word_tokenize(str(text))
        tagged = pos_tag(tokens)
        tree = ne_chunk(tagged, binary=False)
        return [
            (" ".join(t for t, _ in sub.leaves()), sub.label())
            for sub in tree if hasattr(sub, "label")
        ]
    except Exception:
        return []


def vader_label(score: float) -> str:
    """Map a VADER compound score to a polarity label (+/-0.05 thresholds)."""
    if score >= VADER_POS_THRESHOLD:
        return "positive"
    if score <= VADER_NEG_THRESHOLD:
        return "negative"
    return "neutral"


def run_ner(raw: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Run NLTK NER over every review; return a long entity table."""
    raw["nlp_entities"] = raw[text_col].apply(extract_ner_entities)
    rows = []
    for idx, row in raw.iterrows():
        for ename, etype in row["nlp_entities"]:
            rows.append({
                "review_id": row.get("review_id", idx),
                "entity_name": ename,
                "entity_type": etype,
            })
    return pd.DataFrame(rows)


def run_vader(raw: pd.DataFrame, text_col: str, sia: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """Score each review with document-level VADER sentiment."""
    raw["vader_compound"] = raw[text_col].apply(
        lambda t: sia.polarity_scores(str(t))["compound"]
    )
    raw["vader_label"] = raw["vader_compound"].apply(vader_label)
    return raw


def build_cooccurrence_graph(nlp_ent: pd.DataFrame, stop_words: set) -> nx.Graph:
    """Every pair of entities appearing in the same review is linked.

    Edge weight accumulates the number of reviews in which the pair co-occurs.
    """
    G = nx.Graph()
    for _rid, group in nlp_ent.groupby("review_id"):
        tokens = list({
            e.lower().strip()
            for e in group["entity_name"]
            if len(e.strip()) > 2 and e.lower().strip() not in stop_words
        })
        for a, b in itertools.combinations(tokens, 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)
    return G


def network_metrics(G: nx.Graph, label: str = "Graph") -> dict:
    """Topology summary for a graph (nodes, edges, degree centrality, entropy)."""
    G_un = G.to_undirected() if G.is_directed() else G
    degrees = [d for _, d in G_un.degree()]
    degree_cent = nx.degree_centrality(G_un)
    top_node = max(degree_cent, key=degree_cent.get) if degree_cent else None
    deg_counts = np.array(list(Counter(degrees).values()), dtype=float)
    shannon = scipy_entropy(deg_counts / deg_counts.sum(), base=2) if deg_counts.size else 0.0
    largest_cc = max(nx.connected_components(G_un), key=len) if G_un.number_of_nodes() else set()
    avg_clust = nx.average_clustering(G_un.subgraph(largest_cc)) if largest_cc else 0.0
    return {
        "label": label,
        "Nodes": G_un.number_of_nodes(),
        "Edges": G_un.number_of_edges(),
        "Core node": top_node,
        "Core degree centrality": round(degree_cent.get(top_node, 0.0), 4) if top_node else 0.0,
        "Shannon entropy (bits)": round(float(shannon), 4),
        "Avg clustering coeff": round(float(avg_clust), 4),
        "Network density": round(nx.density(G_un), 6),
        "Avg degree": round(float(np.mean(degrees)), 2) if degrees else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Traditional NLP baseline (paper Section 3.2).")
    parser.add_argument(
        "--reviews-csv", required=True,
        help="CSV of review text (NOT redistributed here; supply your own). "
             "Needs a free-text column such as 'text_clean' and ideally a 'review_id'.",
    )
    parser.add_argument("--out-csv", default="nlp_baseline_metrics.csv")
    args = parser.parse_args()

    ensure_nltk_data()
    sia = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words("english"))

    raw = pd.read_csv(args.reviews_csv)
    text_col = detect_text_column(raw)
    print(f"Using text column: {text_col!r} over {len(raw):,} reviews")

    raw = run_vader(raw, text_col, sia)
    print("Document-level sentiment (VADER):")
    print(raw["vader_label"].value_counts().to_string())

    nlp_ent = run_ner(raw, text_col)
    print(f"NLTK NER extracted {len(nlp_ent):,} entity mentions "
          f"across {nlp_ent['entity_type'].nunique()} generic types")

    G_nlp = build_cooccurrence_graph(nlp_ent, stop_words)
    metrics = network_metrics(G_nlp, "NLP co-occurrence")
    pd.DataFrame([metrics]).to_csv(args.out_csv, index=False)
    print("Co-occurrence network metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
