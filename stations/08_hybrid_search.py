"""
Station 08: Hybrid Search mit BM25, Dense Retrieval und RRF

Lernziel:
Lexikalische und semantische Suche kombinieren.

BM25 liefert ein Ranking anhand von Suchbegriffen.
Dense Retrieval liefert ein Ranking anhand von Embeddings.
RRF kombiniert beide Rankings anhand ihrer Positionen.

Voraussetzung:
uv run python -m stations.05_chroma_filter --index

Start:
uv run python -m stations.08_hybrid
"""

import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_05"

COLLECTION_NAME = "teralune_metadata"
EMBEDDING_MODEL = "text-embedding-3-small"

QUESTIONS = [
    "How long does a cursor remain valid?",
    "next_cursor",
]

TOP_K = 5
RRF_K = 60

load_dotenv(PROJECT_DIR / ".env")


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(rankings, k=60):
    """
    Kombiniert Rankings anhand ihrer Rangpositionen.

    Anders als in Station 07 enthalten die Treffer
    jetzt Dictionaries mit ID, Text und Score.

    RRF benötigt daraus ausschließlich die Chunk-ID.
    """

    scores = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit["id"]

            contribution = 1 / (k + rank)

            scores[chunk_id] = scores.get(chunk_id, 0) + contribution

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return ranked


def dense_search(collection, client, question, top_k=5):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )

    question_vector = response.data[0].embedding

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=top_k,
    )

    hits = []

    for chunk_id, document, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "id": chunk_id,
                "text": document,
                "distance": float(distance),
            }
        )

    return hits


def bm25_search(collection, question, top_k=5):

    stored = collection.get(include=["documents"])

    documents = stored["documents"]
    ids = stored["ids"]

    if not documents:
        return []

    tokenized_documents = [tokenize(doc) for doc in documents]

    bm25 = BM25Okapi(tokenized_documents)
    scores = bm25.get_scores(tokenize(question))

    ranked = sorted(
        zip(ids, documents, scores),
        key=lambda item: (-item[2], item[0]),
    )

    results = []

    for chunk_id, document, score in ranked[:top_k]:
        if score <= 0:
            continue

        results.append(
            {
                "id": chunk_id,
                "text": document,
                "bm25": float(score),
            }
        )

    return results



def main():

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = chroma.get_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        raise SystemExit("Collection fehlt. Zuerst Station 05 indexieren.")

    client = OpenAI()

    print("HYBRID SEARCH")
    print("Gespeicherte Chunks:", collection.count())

    for question in QUESTIONS:
        lexical_result = bm25_search(collection, question, top_k=TOP_K)
        dense_result = dense_search(collection, client, question, top_k=TOP_K)
        fused = reciprocal_rank_fusion([lexical_result, dense_result])

        print("\n" + "=" * 60)
        print("FRAGE:", question)
        print("=" * 60)

        print("\nBM25:")

        for rank, hit in enumerate(lexical_result[:3], start=1):
            print(f"  {rank}. {hit['id']:<12} Score: {hit['bm25']:.3f}")

        print("\nDense Retrieval:")

        for rank, hit in enumerate(dense_result[:3], start=1):
            print(f"  {rank}. {hit['id']:<12} Distanz: {hit['distance']:.3f}")

        print("\nHybrid Search (RRF):")

        for rank, (chunk_id, score) in enumerate(fused[:3], start=1):
            print(f"  {rank}. {chunk_id:<12} RRF: {score:.5f}")


main()