"""
Station 06: Lexikalische Suche mit BM25

Lernziel:
Dokumente anhand ihrer Suchbegriffe durchsuchen.

Voraussetzung:
uv run python -m stations.05_chroma_filter --index

Start:
uv run python -m stations.06_bm25
"""

import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_05"
COLLECTION_NAME = "teralune_metadata"

QUESTIONS = [
    "How long does a cursor remain valid?",
    "next_cursor",
]


def tokenize(text):
    """Zerlegt Text in kleingeschriebene Suchbegriffe."""

    return re.findall(r"\w+", text.lower())


def bm25_search(collection, question, top_k=5):

    stored = collection.get(include=["documents"])
    documents = stored["documents"]
    ids = stored["ids"]

    if not documents:
        return []


    tokenized_documents = [tokenize(document) for document in documents]

    bm25 = BM25Okapi(tokenized_documents)
    question_tokens = tokenize(question)
    # Berechne die Scores für die tokenisierte Suchfrage
    scores = bm25.get_scores(question_tokens)

    results = []
    ranked = sorted(
        zip(ids, documents, scores),
        key=lambda item: (-item[2], item[0]),
    )

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

    print("BM25-SUCHE")
    print("Gespeicherte Chunks:", collection.count())

    for question in QUESTIONS:
        results = bm25_search(collection, question, top_k=5)

        print("\n" + "=" * 60)
        print("FRAGE:", question)

        if not results:
            print("Noch keine Treffer. TODOs bearbeiten.")
            continue

        for rank, hit in enumerate(results, start=1):
            print(f"\nRang {rank}")
            print("Chunk:", hit["id"])
            print("BM25-Score:", round(hit["bm25"], 3))
            print("Text:", hit["text"][:300])


if __name__ == "__main__":
    main()
