# 12-Type Entity Taxonomy for Living Sacred Heritage

Specification accompanying *Prompt-based knowledge acquisition for living sacred heritage: a domain-adaptive entity taxonomy and semantic network from visitor reviews.*

The taxonomy is applied **declaratively** — it is specified in the extraction prompt (`extraction_prompt.txt`), not learned from labelled training data. Four types are carried over from the original LLM-SC framework; eight are introduced here for sacred heritage and account for 64.8% of all extracted entities.

## Entity classes

| Type | Definition | Examples | Count | % |
|---|---|---|---|---|
| `ritual` † | Religious ceremonies and acts | cremation, aarti, darshan, puja | 1,741 | 15.4 |
| `scenic_spot` | Physical heritage structures, architectural features | pagoda, stone carvings, courtyard | 1,642 | 14.5 |
| `spiritual_emotion` † | Internal emotional/spiritual states felt **by the reviewer** | awe, peace, humility, overwhelmed | 1,462 | 13.0 |
| `sacred_space` † | Religiously significant physical locations | Bagmati River, ghat, inner sanctum | 1,425 | 12.6 |
| `problem` | Visitor complaints, management failures | entry restriction, pollution, overcrowding | 1,270 | 11.3 |
| `cultural_rule` † | Access norms, restrictions, cultural protocols | non-Hindu entry ban, dress code, photo ban | 1,064 | 9.4 |
| `facility_service` | Physical facilities and operational services | ticketing, signage, restroom, guide | 876 | 7.8 |
| `religious_actor` † | Individuals in religious or devotional roles | sadhu, priest, pilgrim, devotee | 798 | 7.1 |
| `sacred_object` † | Religiously significant physical items | Shiva lingam, prasad, incense, diya | 493 | 4.4 |
| `dual_valence` † | Experiences eliciting simultaneous +/− emotion | cremation ceremony (grief + transcendence) | 220 | 1.9 |
| `general_sentiment` | Overall emotional tone not tied to a specific entity | wonderful, peaceful, spiritual | 190 | 1.7 |
| `festival_event` † | Named religious festivals and ceremonies | Maha Shivaratri, Teej, Bala Chaturdashi | 105 | 0.9 |
| **Total** | **12 types** | | **11,286** | **100.0** |

† = new to sacred heritage (8 types, 7,308 entities, 64.8%)

## Constraints

Two classes carry constraints that must be enforced explicitly, because pilot testing showed models violate them by default:

**`dual_valence`** — the reviewer must express **both** a positive and a negative emotion about the **same** referent within the same textual span.
- Qualifies: *"watching the cremation was both heartbreaking and transcendent"* → `cremation ceremony`
- Does **not** qualify: *"dead bodies were visible"* — one-sided observation, classify as `sacred_space` or `ritual`
- Does **not** qualify: anything only positive or only negative

**`spiritual_emotion`** — the state must be felt by the reviewer personally, not observed in others or attributed to the site in the abstract.
- Qualifies: *"I was moved"*, *"overwhelmed with emotions"*, *"a profound sense of peace"*, *"humbled by what I witnessed"*
- Does **not** qualify: *"pilgrims were praying devoutly"* (third-party), *"the atmosphere felt sacred"* (ambient quality)

**Disjointness with aspect labels** — `sacred_atmosphere` and `spiritual_authenticity` are sentiment **aspect** labels, not entity types. An entity about the sacred feeling of a place is `spiritual_emotion`.

**Exclusions** — the review subject itself is never extracted: Pashupatinath Temple, "the temple", Kathmandu, Nepal, India. Generic nouns (place, site, visit, experience) are also excluded.

## Relation types

Five relation classes connect entity pairs in the semantic network:

| Relation | Records | Directed |
|---|---|---|
| `co_occurrence` | 1,962 | no |
| `description` | 1,420 | yes |
| `causal` | 1,124 | yes |
| `contrast` | 1,068 | no |
| `other` (residual, out-of-schema) | 16 | — |

Edge weight: `W = α × co_occurrence_weight + β × causal_weight`, with α = 0.7, β = 0.3.

## Sentiment aspects

Nine aspects, each scored 0.0–1.0 with a supporting evidence span:
`service` · `facility` · `environment` · `ritual_experience` · `spiritual_authenticity` · `access_fairness` · `sacred_atmosphere` · `crowd_management` · `cultural_sensitivity`

Labels are constrained to `positive` | `neutral` | `negative`. Exactly 0.5 is prohibited unless the text contains genuinely equal positive and negative evidence for that aspect.

## Reuse

To apply this taxonomy at another sacred heritage site, edit the site-specific exclusions in rule 1 of the prompt and the illustrative examples attached to each type. No retraining is required.
