"""
Station 04: Markdown-Dokumente in Chroma speichern

Lernziel:
Mehrere Markdown-Dateien einlesen, in Chunks zerlegen
und in einer Vektordatenbank speichern.

Index erstellen:
uv run python -m stations.04_chroma_documents --index

Suche durchführen:
uv run python -m stations.04_chroma_documents
"""

import argparse
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCUMENT_DIR = PROJECT_DIR / "data" / "teralune-mini" / "corpus" / "public"
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_02"
COLLECTION_NAME = "teralune_documents"

QUESTION = "How long is an Orbit 3 access token valid?"

load_dotenv(PROJECT_DIR / ".env")
client = OpenAI()
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

parser = argparse.ArgumentParser()
parser.add_argument("--index", action="store_true")
args = parser.parse_args()


# Structure Chunking aus Station 02
def structure_chunks(text) -> list:
    """Zerlegt Markdown-Text anhand der Überschriften."""

    chunks = []
    current_chunk = ""

    for line in text.splitlines():
        if line.startswith(("# ", "## ", "### ")):
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# 1. Dokumente indexieren
if args.index:
    documents = []
    ids = []

    files = sorted(DOCUMENT_DIR.glob("*.md"))

    print("Gefundene Dateien:", len(files))
    for file in files:
        print("\nDatei:", file.name)
        text = file.read_text(encoding="utf-8")
        chunks = structure_chunks(text)
        print("Erzeugte Chunks:", len(chunks))

        # IDS erzeugen und chunks in documents hängen
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            ids.append(f"{file.name}_{i}")

    print("Erzeugte Chunks:", len(documents))
    print("Erzeugte IDS:", len(ids))

    # Solange keine Chunks vorhanden sind, beenden wir
    # das Programm ohne API-Aufruf.
    if not documents:
        raise SystemExit("Noch keine Chunks vorhanden. TODO 1 bearbeiten.")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=documents,
    )

    vectors = [item.embedding for item in response.data]
    print("Erzeugte Embeddings:", len(vectors))

    collection = chroma.get_or_create_collection(name=COLLECTION_NAME)
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=vectors,
    )

    print("Indexierung fertig.")
    print("Gespeicherte Chunks:", collection.count())



else:
    # Prüfen, ob die Collection bereits existiert.
    # So entsteht kein Fehler beim ersten Programmstart.
    try:
        collection = chroma.get_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        raise SystemExit("Collection fehlt. Zuerst mit --index indexieren.")

    print("Gespeicherte Chunks:", collection.count())

    if collection.count() == 0:
        raise SystemExit("Collection ist leer. TODO 2 bearbeiten.")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=QUESTION,
    )

    question_vector = response.data[0].embedding

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=3,
    )

    print("\nFrage:", QUESTION)
    print("\nGefundene Chunks:")

    for document, distance in zip(
        results["documents"][0],
        results["distances"][0],
    ):
        print(f"\nDistanz: {distance:.3f}")
        print(document[:500])
        print("=" * 60)
