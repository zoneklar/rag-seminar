"""
Station 09: Retrieval-Evaluation mit Goldfragen

Lernziel:
Überprüfen, ob die erwarteten Chunks unter den
ersten k Suchergebnissen gefunden werden.

Metrik:
Hit@k

Voraussetzung:
uv run python -m stations.05_chroma_filter --index

Start:
uv run python -m stations.09_evaluation
"""

from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# 1. Vorbereitung

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_05"

COLLECTION_NAME = "teralune_metadata"
EMBEDDING_MODEL = "text-embedding-3-small"

TOP_K = 3

GOLD_QUESTIONS = [
    {
        "question": "How long does a cursor remain valid?",
        "gold_chunk": "D022_2",
    },
    {
        "question": "How long is an Orbit 3 access token valid?",
        "gold_chunk": "D002_1",
    },
    {
        "question": "What is the next_cursor field used for?",
        "gold_chunk": "D022_1",
    },
]

load_dotenv(PROJECT_DIR / ".env")

client = OpenAI()

chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    collection = chroma.get_collection(name=COLLECTION_NAME)
except chromadb.errors.NotFoundError:
    raise SystemExit("Collection fehlt. Zuerst Station 05 indexieren.")


# 2. Goldfragen auswerten

hits = 0
reciprocal_ranks = []

for item in GOLD_QUESTIONS:
    question = item["question"]
    gold_chunk = item["gold_chunk"]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )

    question_vector = response.data[0].embedding

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=TOP_K,
    )

    found_ids = results["ids"][0]

    

    hit = gold_chunk in found_ids
    rr = 0.0
    if hit:
        hits += 1
        rank = found_ids.index(gold_chunk) + 1
        rr = 1 / rank
    
    reciprocal_ranks.append(rr)

    print("\nFRAGE:", question)
    print("Erwartet:", gold_chunk)
    print("Gefunden:", found_ids)
    print(f"Hit@{TOP_K}:", int(hit))
    print(f"RR@{TOP_K}:", rr)



# 3. Gesamtergebnis

print("\n" + "=" * 60)
print("RETRIEVAL-EVALUATION")
print("=" * 60)

print("Goldfragen:", len(GOLD_QUESTIONS))
hit_rate = hits / len(GOLD_QUESTIONS)
mrr = sum(reciprocal_ranks) / len(GOLD_QUESTIONS)

# print("Erfolgreich:", hits)
print(f"Hit-Range@{TOP_K}: {hit_rate:.1%}")
print(f"MRR@{TOP_K}: {mrr:.3f}")

