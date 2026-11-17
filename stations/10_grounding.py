"""
Station 10: Grounding und Quellenbezug

Lernziel:
Gefundene Chunks als Kontext an ein LLM übergeben.

Experiment:
1. Eine beantwortbare Frage.
2. Eine Frage mit fehlenden Informationen.

Voraussetzung:
uv run python -m stations.05_chroma_filter --index

Start:
uv run python -m stations.10_grounding
"""

from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# 1. Konfiguration

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_05"

COLLECTION_NAME = "teralune_metadata"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

TOP_K = 3

QUESTIONS = [
    "Wie lange versucht Orbit 3.0 fehlgeschlagene Webhooks erneut?",
    "Welche offiziellen Rabatte bekommen Universitäten?",
]

INSTRUCTIONS = """
Beantworte die Frage ausschließlich anhand des bereitgestellten Kontexts.

Wenn der Kontext keine ausreichenden Informationen enthält,
sage ausdrücklich, dass du die Frage nicht beantworten kannst.

Erfinde keine Informationen.

Gib für jede wesentliche Aussage die Chunk-ID der verwendeten Quelle an,
beispielsweise [D022_2]. Verwende ausschließlich IDs aus dem Kontext.

Antworte auf Deutsch.
"""

load_dotenv(PROJECT_DIR / ".env")


# 2. Dense Retrieval


def dense_search(collection, client, question, top_k=3):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )

    results = collection.query(
        query_embeddings=[response.data[0].embedding],
        n_results=top_k,
        where={
            "$and": [
                {"version": "3.0"},
                {"status": "current"},
            ]
        },
    )

    return [
        {"id": chunk_id, "text": text, "metadata": metadata}
        for chunk_id, text, metadata in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
        )
    ]


def build_context(hits):
    texte = []

    for hit in hits:
        chunk_id = hit["id"]
        source = hit["metadata"]["source"]
        text = hit["text"]

        texte.append(
            f"Chunk-ID: {chunk_id}\n"
            f"Quelle: {source}\n"
            f"{text}\n"
        )

    
    return "\n\n".join(texte)


# 4. Antwort generieren


def answer(client, question, context):
    """
    Übergibt Kontext und Frage an das Sprachmodell.

    Das Modell erhält zwei unterschiedliche Informationen:

    instructions:
        Regeln für die Antwortgenerierung.

    input:
        Den gefundenen Kontext und die Benutzerfrage.

    Entscheidend:
    Das Modell besitzt weiterhin sein vortrainiertes Wissen.

    Wir fordern es auf, ausschließlich den Kontext zu
    verwenden. Technisch verhindern können wir die
    Verwendung anderen Wissens damit nicht.

    Außerdem kann das Modell Informationen im Kontext
    falsch interpretieren oder unzulässige Schlüsse ziehen.
    """

    response = client.responses.create(
        model=CHAT_MODEL,
        instructions=INSTRUCTIONS,
        input=f"KONTEXT:\n{context}\n\nFRAGE:\n{question}"
    )

    return response.output_text


# 5. Hauptprogramm


def main():

    client = OpenAI()

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = chroma.get_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        raise SystemExit("Collection fehlt. Zuerst Station 05 indexieren.")

    if collection.count() == 0:
        raise SystemExit("Collection ist leer. Zuerst Station 05 indexieren.")

    for question in QUESTIONS:
        hits = dense_search(
            collection,
            client,
            question,
            top_k=TOP_K,
        )

        print("\n" + "=" * 60)
        print("FRAGE:", question)

        print("\nQUELLEN:")
        for hit in hits:
            print(" ", hit["metadata"]["source"])

        context = build_context(hits)
        print("\nKONTEXT:")
        print(context)

        result = answer(client, question, context)
        print("\nANTWORT:")
        print(result)

    print("\n" + "=" * 60)
    print("Ein Grounding-Prompt garantiert keine Faktentreue.")


if __name__ == "__main__":
    main()
