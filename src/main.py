"""
batch_run.py  —  Pashupatinath review extraction via OpenAI Batch API
Run from your terminal:
    pip install openai pandas tqdm
    python batch_run.py
"""

import json, os, re, time, math
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ════════════════════════════════════════════════════════════════
# CONFIG  — edit these
# ════════════════════════════════════════════════════════════════
API_KEY          = "API_KEY"
INPUT_FILE       = "2500_random.csv"
OUTPUT_RAW       = "llm_raw_outputs.jsonl"
CHECKPOINT_FILE  = "extraction_checkpoint.json"
CHUNK_SIZE       = 45       # safe under 90k enqueued token limit
TEMPERATURE      = 0.1
MAX_NEW_TOKENS   = 900
POLL_INTERVAL    = 30       # seconds between status checks

client = OpenAI(api_key=API_KEY)

# ════════════════════════════════════════════════════════════════
# PROMPT
# ════════════════════════════════════════════════════════════════
def build_prompt(text):
    return f"""You are analyzing tourist reviews of Pashupatinath Temple, Kathmandu — a living UNESCO sacred heritage site.
Extract entities, relationships, and aspect-level sentiments. Output STRICT JSON only — no preamble, no explanation, no markdown fences.

ENTITY TYPES (use exactly these labels):
  "scenic_spot"       : Physical heritage structures, architectural features, monuments, carvings, pagodas, spires, courtyards
  "problem"           : Complaints, negative experiences, management failures, pollution, aggression, poor facilities
  "facility_service"  : Staff behavior, ticketing, guides, restrooms, food stalls, signage, parking, entry process
  "general_sentiment" : Overall emotional tone not tied to any specific aspect
  "ritual"            : Religious ceremonies and acts: aarti, puja, cremation ceremony, darshan, parikrama, tarpan, abhishek, havan
  "religious_actor"   : Specific people in religious roles: sadhu, priest, pujari, pilgrim, devotee, monk, yogi — NOT generic tourists
  "sacred_space"      : Religiously significant locations: ghat, Bagmati River, inner sanctum, cremation ground, lingam shrine, pagoda roof
  "spiritual_emotion" : Deep internal emotional/spiritual states explicitly felt by the reviewer: awe, transcendence, devotion, reverence, peace, grief, humility, being moved, overwhelmed, blessed, spiritual connection, sacred feeling, sense of mortality — MUST be felt by the reviewer, not just described
  "festival_event"    : Named religious events: Maha Shivaratri, Teej festival, Bala Chaturdashi, evening aarati ceremony
  "cultural_rule"     : Access norms and restrictions: non-Hindu entry ban, dress code requirement, no photography zones, entry fee for foreigners
  "sacred_object"     : Religiously significant physical items: prasad, Shiva lingam, incense sticks, flower offerings, trident (trishul), marigold garlands, diya lamp, rudraksha
  "dual_valence"      : An experience the reviewer describes with SIMULTANEOUS positive AND negative emotion — grief AND awe together, beautiful AND sad, moving AND disturbing IN THE SAME SENTENCE about the SAME thing

RELATIONSHIP TYPES: co_occurrence | causal | description | contrast

SENTIMENT ASPECTS: service | facility | environment | ritual_experience | spiritual_authenticity | access_fairness | sacred_atmosphere | crowd_management | cultural_sensitivity

CRITICAL RULES — read carefully:
1. DO NOT extract "Pashupatinath Temple", "the temple", "Kathmandu", "Nepal", "India" as entities — these are the review subject, not extractable entities
2. DO NOT extract generic nouns ("place", "site", "visit", "experience") as entities
3. spiritual_emotion MUST be an emotion the REVIEWER personally felt — phrases like "I was moved", "I felt awe", "overwhelmed with emotions", "sense of peace", "humbled", "deeply spiritual moment", "moved to tears" → ALL are spiritual_emotion. Extract the emotion name as the entity ("awe", "sense of peace", "overwhelmed emotions"). Also classify as spiritual_emotion: 'other worldly fascination', 'exquisite cultural experience', 'strangely changed', 'sense of mortality' — these are reviewer-felt internal states."
4. dual_valence — STRICT definition with examples:
   ✓ CORRECT: "watching the cremation was both heartbreaking and transcendent" → dual_valence entity = "cremation ceremony"
   ✓ CORRECT: "the funeral pyres were deeply sad yet strangely beautiful" → dual_valence entity = "funeral pyres"
   ✗ WRONG: "dead bodies" alone → NOT dual_valence, just sacred_space or ritual
   ✗ WRONG: "monkeys", "crowd", "smoke" → NOT dual_valence
   ✗ WRONG: anything that is only negative or only positive → NOT dual_valence
   Only use dual_valence if the reviewer explicitly expresses BOTH a positive feeling AND a negative feeling about the EXACT same thing
5. Sentiment scores — use the FULL range:
   0.0–0.2 = strongly negative ("terrible", "disgusting", "ruined")
   0.3–0.4 = mildly negative ("disappointing", "could be better")
   0.6–0.7 = mildly positive ("nice", "worth visiting", "good")
   0.8–0.9 = strongly positive ("amazing", "life-changing", "beautiful")
   1.0 = superlative ("best experience of my life", "absolutely perfect")
   NEVER use exactly 0.5 unless the text contains genuinely equal positive and negative evidence for that specific aspect. NEVER use 0.0 unless the text is explicitly hateful or disgusted.
6. "sacred_atmosphere" and "spiritual_authenticity" are ASPECT labels for sentiment analysis — they are NOT entity types. If you want to extract an entity about the sacred feeling or authenticity of a place, use "spiritual_emotion" instead.
7. sentiment value in aspect_sentiments MUST be exactly one of: "positive", "neutral", or "negative" — no other values, no qualifiers like "mildly negative".
8. Extract ALL entities present — be thorough — minimum 4 entities per review
9. Return ONLY valid JSON — no markdown, no backticks, no extra text

OUTPUT FORMAT:
{{
  "entities": [
    {{"id": 1, "name": "<exact phrase from text, not generic>", "type": "<entity_type>", "quote": "<2-6 word supporting quote from text>"}}
  ],
  "relations": [
    {{"from_id": 1, "to_id": 2, "type": "<relation_type>", "description": "<brief reason>"}}
  ],
  "aspect_sentiments": [
    {{"aspect": "<aspect>", "sentiment": "positive|neutral|negative", "score": 0.85, "evidence": "<direct quote from review>"}}
  ],
  "review_type": "devotional|experiential|critical|mixed",
  "dominant_experience": "<one phrase: what is this review primarily about>"
}}

REVIEW:
{text}"""

# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f).get("processed_ids", []))
    return set()


def save_checkpoint(processed_ids):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed_ids": list(processed_ids)}, f)


def wait_for_batch(batch_id):
    print(f"  Polling batch {batch_id} every {POLL_INTERVAL}s ...")
    while True:
        b = client.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"    status={b.status}  completed={c.completed}  "
              f"failed={c.failed}  total={c.total}")
        if b.status == "completed":
            return b
        elif b.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} ended with status: {b.status}\n"
                               f"Errors: {b.errors}")
        time.sleep(POLL_INTERVAL)


def submit_chunk(lines, chunk_index):
    """Write a temp .jsonl, upload it, submit, return the batch object."""
    tmp_file = f"_chunk_{chunk_index}.jsonl"
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    with open(tmp_file, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    os.remove(tmp_file)
    return batch


def parse_and_save(completed_batch, meta, processed_ids):
    """Download results, parse JSON, append to OUTPUT_RAW."""
    result_content = client.files.content(completed_batch.output_file_id).text
    lines = [l for l in result_content.strip().split("\n") if l]
    new_ids = []

    with open(OUTPUT_RAW, "a", encoding="utf-8") as out_f:
        for line in lines:
            result   = json.loads(line)
            rid      = result["custom_id"]
            row_meta = meta.get(rid, {})

            try:
                raw_out = result["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                raw_out = ""

            try:
                json_match = re.search(r"\{.*\}", raw_out, re.DOTALL)
                parsed = json.loads(json_match.group() if json_match else raw_out)
                valid  = True
            except Exception:
                parsed = {}
                valid  = False

            record = {
                "review_id":  rid,
                "rating":     row_meta.get("rating"),
                "raw_output": raw_out,
                "valid_json": valid,
                "parsed":     parsed,
                "year":       row_meta.get("year"),
                "period":     row_meta.get("period"),
                "trip_type":  row_meta.get("trip_type"),
                "word_count": row_meta.get("word_count"),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            new_ids.append(rid)

    processed_ids.update(new_ids)
    save_checkpoint(processed_ids)

    if completed_batch.error_file_id:
        print(f"  ⚠️  Some requests failed — error file: {completed_batch.error_file_id}")

    return len(new_ids)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    # ── Load data ────────────────────────────────────────────────
    df = pd.read_csv(INPUT_FILE)
    print(f"✓ Loaded {len(df)} reviews from {INPUT_FILE}")

    meta = df.set_index(df["review_id"].astype(str)).to_dict("index")

    # ── Resume support ───────────────────────────────────────────
    processed_ids = load_checkpoint()
    if processed_ids:
        print(f"✓ Resuming — already processed: {len(processed_ids)}")

    # ── Build request lines for unprocessed reviews ───────────────
    all_lines = []
    for _, row in df.iterrows():
        rid = str(row["review_id"])
        if rid in processed_ids:
            continue
        request = {
            "custom_id": rid,
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model":       "gpt-4o",
                "temperature": TEMPERATURE,
                "max_tokens":  MAX_NEW_TOKENS,
                "messages":    [{"role": "user", "content": build_prompt(row["text_clean"])}]
            }
        }
        all_lines.append(json.dumps(request, ensure_ascii=False) + "\n")

    if not all_lines:
        print("✅ Nothing new to process — all reviews already in checkpoint.")
        return

    # ── Split into chunks ────────────────────────────────────────
    chunks = [all_lines[i:i+CHUNK_SIZE] for i in range(0, len(all_lines), CHUNK_SIZE)]
    total_chunks = len(chunks)
    print(f"✓ {len(all_lines)} reviews → {total_chunks} chunks of up to {CHUNK_SIZE}\n")

    # ── Process chunks sequentially ──────────────────────────────
    for i, chunk in enumerate(chunks, start=1):
        print(f"{'='*55}")
        print(f"Chunk {i}/{total_chunks}  ({len(chunk)} requests)")

        batch = submit_chunk(chunk, i)
        print(f"  Submitted: {batch.id}")

        completed = wait_for_batch(batch.id)

        saved = parse_and_save(completed, meta, processed_ids)
        print(f"  ✓ Saved {saved} records  (total processed: {len(processed_ids)})")

    print(f"\n{'='*55}")
    print(f"✅ All done! {len(processed_ids)} reviews saved to {OUTPUT_RAW}")
    print("   Run your analysis cells (or script) on that file now.")


if __name__ == "__main__":
    main()