# Data dictionary

Three derived output files from the LLM-SC extraction pipeline. All are joinable on `review_id`.

## Provenance and exclusions

These files contain **derived extraction outputs only**. Raw review text is not redistributed, in line with the source platform's terms of use.

`review_id` values are **pseudonymous surrogate keys** (`R00001` … `R02491`), stable and consistent across all three files. The original platform-assigned identifiers were used during processing for de-duplication only and have been removed, so no record can be traced back to a source review or reviewer profile.

The `quote`, `description`, and `evidence` columns contain short supporting spans extracted by the model (typically 4–25 words), not full review text.

---

## `entity_mentions.csv` — 11,286 rows

One row per extracted entity mention.

| Column | Type | Description |
|---|---|---|
| `review_id` | string | Pseudonymous review key |
| `year` | int | Year of review publication (2008–2026) |
| `period` | string | Temporal period: `early`, `growth`, `pre_covid_peak`, `covid_onset`, `covid_deep`, `recovery_early`, `recovery_late`, `post_recovery` |
| `trip_type` | string | `solo`, `couples`, `friends`, `family`, `business`, or blank if unspecified |
| `entity_name` | string | Surface form of the extracted entity |
| `entity_type` | string | One of the 12 taxonomy types (see `../taxonomy/`) |
| `quote` | string | Supporting span from the review text |
| `rating` | int | Star rating 1–5 |

## `relations.csv` — 5,590 rows

One row per extracted relation record. After case-normalising node names and de-duplicating pairs, these yield a graph of **5,110 nodes** and **5,308 typed edges**.

| Column | Type | Description |
|---|---|---|
| `review_id` | string | Pseudonymous review key |
| `year` | int | Year of review publication |
| `source_node` | string | Source entity name |
| `target_node` | string | Target entity name |
| `relation` | string | `co_occurrence`, `description`, `causal`, `contrast`, or a residual out-of-schema label |
| `description` | string | Short natural-language gloss of the relation |

Note: 16 records carry out-of-schema relation labels (prompt rule violations) and are retained as `other` in the paper's analysis.

## `aspect_sentiments.csv` — 6,232 rows

One row per aspect-level sentiment judgement (mean 2.51 per review).

| Column | Type | Description |
|---|---|---|
| `review_id` | string | Pseudonymous review key |
| `year` | int | Year of review publication |
| `trip_type` | string | As above |
| `aspect` | string | One of nine: `service`, `facility`, `environment`, `ritual_experience`, `spiritual_authenticity`, `access_fairness`, `sacred_atmosphere`, `crowd_management`, `cultural_sensitivity` |
| `sentiment` | string | `positive`, `neutral`, `negative` (11 records carry out-of-schema labels) |
| `score` | float | 0.0–1.0 calibrated sentiment score |
| `evidence` | string | Supporting span from the review text |
| `rating` | int | Star rating 1–5 |

Note: 7 records carry out-of-schema aspect labels and are excluded from Table 2 of the paper, giving 6,225 of the 6,232 total.

---

## Redactions

One extracted entity contained a third party's personal name and telephone number, quoted
verbatim by a reviewer. Both have been replaced with `[NAME REDACTED]` and
`[PHONE REDACTED]` (10 fields affected, 1 entity). All other values are as produced by the
extraction pipeline.

The full dataset was screened for telephone numbers, email addresses, URLs, and social
handles; this was the only match.

## Known schema violations

A small fraction of model outputs violated the prompt's controlled vocabularies and are retained here as produced, for transparency:

| Violation | Count |
|---|---|
| Out-of-schema sentiment labels (`mixed`, `dual_valence`) | 11 |
| Out-of-schema aspect labels | 7 |
| Out-of-schema relation labels | 16 |

These are reported in §6.4 of the paper and excluded from the affected analyses.
