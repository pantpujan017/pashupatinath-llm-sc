# Prompt-based knowledge acquisition for living sacred heritage

Taxonomy, extraction prompt, derived outputs, and analysis code accompanying:

> Pant, P., Newpane, S., Sitaula, H., Sakhakarmy, M. **Prompt-based knowledge acquisition for living sacred heritage: a domain-adaptive entity taxonomy and semantic network from visitor reviews.** Submitted to *Data & Knowledge Engineering*.

## What this is

A domain-adaptive knowledge-acquisition pipeline for a domain with **no annotated corpus and no reference ontology**. A 12-type entity taxonomy is specified declaratively in an extraction prompt — no labelled training data, no fine-tuning — and applied with GPT-4o to visitor reviews of Pashupatinath Temple, Kathmandu, a living UNESCO sacred heritage site. The output is a typed, queryable semantic network of visitor experience.

Eight of the twelve entity types are new to sacred heritage and account for 64.8% of extracted entities. The taxonomy introduces a `dual_valence` class for experiences described with simultaneous grief and transcendence at cremation-active sites.

## Contents

```
taxonomy/
  taxonomy_specification.md   12 entity types, constraints, relation schema, aspect list
  extraction_prompt.txt       verbatim prompt (Appendix A of the paper)
data/
  entity_mentions.csv         11,286 extracted entities
  relations.csv               5,590 relation records -> 5,110 nodes / 5,308 typed edges
  aspect_sentiments.csv       6,232 aspect-level sentiment records
  README.md                   column dictionary and provenance notes
notebooks/
  nlp_vs_llm_comparison.ipynb traditional NLP baseline vs. LLM-SC comparison
src/
  nlp_baseline.py             traditional NLP baseline: NLTK ne_chunk NER, VADER, co-occurrence graph (§3.2)
  network_construction.py     graph construction, edge weighting (α=0.7, β=0.3), degree centrality, Louvain communities (§3.4)
  extraction_pipeline.py      OpenAI Batch API extraction — documented skeleton (§3.3; see note below)
```

`src/nlp_baseline.py` and `src/network_construction.py` are the analysis code used
for the paper. `src/extraction_pipeline.py` documents the Section 3.3 extraction
procedure and parameters but is a reconstructed skeleton, not a runnable
reproduction: the original OpenAI Batch API driver was run outside version control
and is not available. The authoritative extraction prompt is
`taxonomy/extraction_prompt.txt`. The recovered network code computes degree
centrality and weighted degree; betweenness centrality is not included because it
was not part of the recovered analysis modules.

## Data provenance and what is deliberately excluded

**Raw review text is not redistributed**, in line with the source platform's terms of use. This repository contains only derived extraction outputs: entity mentions, typed relations, and aspect-level sentiment records, each with a short supporting span (typically under 20 words) rather than full review text.

**Review identifiers are pseudonymised.** Platform-assigned numeric review IDs were used during processing for de-duplication only. They have been replaced here with stable surrogate keys (`R00001` …) that are consistent across all three files, so within-review joins still work, but no record can be traced back to a source review or reviewer profile.

No usernames, profile information, or other personal identifiers were ever collected. No individual reviewer is identifiable from any file in this repository.

## Corpus summary

| | |
|---|---|
| Full corpus | 3,939 TripAdvisor reviews, 14 Oct 2008 – 16 Apr 2026, English |
| Analysed sample | 2,500 randomly drawn reviews |
| Valid entity extraction | 2,485 reviews |
| Valid aspect-sentiment extraction | 2,487 reviews |
| Contributing relation records | 2,433 reviews |
| Review length | mean 68.3 words (SD 60.0), median 49, max 612 |
| Ratings | 5★ 58.4% · 4★ 27.8% · 3★ 9.2% · 2★ 2.8% · 1★ 1.8% |
| Trip types | Friends 26.0% · Solo 21.0% · Couples 20.7% · Family 16.0% · Business 5.6% · Unspecified 10.7% |

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook notebooks/nlp_vs_llm_comparison.ipynb
```

The comparison notebook runs the traditional NLP baseline (NLTK `ne_chunk` + VADER) live and compares it against the pre-computed LLM outputs in `data/`. Point `DATA_DIR` at `data/`.

Note that the notebook's NLP baseline requires raw review text, which is not redistributed here. To re-run that half of the comparison you will need to collect reviews yourself from the source listing.

Extraction used `gpt-4o-2026-05-28` via the OpenAI Batch API, temperature 0.1, max_tokens 1200, in chunks of 45 requests with a 24-hour completion window.

## Citation

See `CITATION.cff`, or cite the paper above.

## Licence

Code is released under the MIT Licence (`LICENSE`). The derived data files in `data/` and the taxonomy specification are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — reuse freely with attribution to the paper.
