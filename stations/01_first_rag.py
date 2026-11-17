"""
Station 01: Unsere erste RAG-Pipeline

Lernziel:
Dokumente mit Embeddings durchsuchen und eine Frage
mithilfe eines Sprachmodells beantworten.

Start:
uv run python -m stations.01_first_rag
"""
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


documents = [
    "Orbit retries failed webhook deliveries for 72 hours.",
    "Orbit access tokens are valid for 60 minutes.",
    "Orbit retains telemetry data for 90 days.",
    "Orbit limits API clients to 1000 requests per minute.",
    "Orbit exports telemetry data in CSV and JSON format.",
]

question = "How long does Orbit retry a failed webhook delivery?"
question = "What's the surname of Mr. Ed"

print("Frage:", question)

# 1. Embeddings erzeugen

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=documents + [question],
)

vectors = [item.embedding for item in response.data]
# Nach diesem Schritt:
print("Anzahl Vektoren:", len(vectors))  # Anzahl der erstellen Vektoren
print("Dimensionen:", len(vectors[0]))


# 2. Ähnlichkeit berechnen

document_vectors = np.array(vectors[:-1])
question_vector = np.array(vectors[-1])

# TODO: Cosine Similarity zwischen Frage und Dokumenten berechnen.
# Bei normierten Embeddings genügt das Skalarprodukt.

similarities = document_vectors @ question_vector

for i in range(len(documents)):
    print(f"{similarities[i]:.4f} | {documents[i]}")

# 3. Retrieval

best_index = np.argmax(similarities)
best_document = documents[best_index]

print("\nGefundenes Dokument:")
print(best_document)

# 10 Minuten Pause 10:40
# 4. Antwort generieren
response = client.responses.create(
    model="gpt-4.1-mini",
    instructions=(
        "Beantworte die Frage ausschließlich anhand des Kontexts. "
        "Falls die Information fehlt, sage das. "
        "Antworte auf Deutsch."
    ),
    input=f"Frage: {question}\n\nKontext: {best_document}",
)

print("Frage:", question)
print("\nAntwort:")
print(response.output_text)