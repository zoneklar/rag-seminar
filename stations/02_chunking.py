"""
Station 02: Dokumente in Chunks zerlegen

Lernziel:
Drei Chunking-Strategien kennenlernen:
1. Fixed Size
2. Overlap
3. Structure Chunking

Start:
uv run python -m stations.02_chunking
"""

from pathlib import Path

# 1. Dokument laden
PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE = PROJECT_DIR / "data" / "teralune-mini" / "corpus" / "public" / "D022.md"

text = FILE.read_text(encoding="utf-8")
print("Dokument:", FILE.name)
print("Anzahl Zeichen:", len(text))


# 2. Fixed Size Chunking
CHUNK_SIZE = 500

chunks = []

for start in range(0, len(text), CHUNK_SIZE):
    # Wir schneiden einen Teil des Strings heraus.
    chunk = text[start : start + CHUNK_SIZE]

    # Den Chunk speichern wir in einer Liste.
    chunks.append(chunk)

print("\nFIXED SIZE")
print("Anzahl Chunks:", len(chunks))
print("Erster Chunk:\n", chunks[0])
print("*******")
print("Zweiter Chunk:\n", chunks[1])

# 3. Chunking mit Overlap

OVERLAP = 100

# TODO:
# Erzeuge Chunks mit jeweils 100 Zeichen Überlappung.
#
step = CHUNK_SIZE - OVERLAP
overlap_chunks = []

for start in range(0, len(text), step):
    chunk = text[start : start + CHUNK_SIZE]

    overlap_chunks.append(chunk)

print("\nMIT OVERLAP")
print("Anzahl Chunks:", len(overlap_chunks))
print("Erster Chunk:\n", overlap_chunks[0])
print("*******")
print("Zweiter Chunk:\n", overlap_chunks[1])


# 4. Structure Chunking (simplified)

structure_chunks = []

# Hier sammeln wir die Zeilen des aktuellen Abschnitts.
current_chunk = ""

# Eine neue Überschrift beginnt einen neuen Chunk.
# Erkenne Überschriften mit #, ## und ###.
for line in text.splitlines():
    # Prüfen, ob die Zeile eine Markdown-Überschrift ist.
    if line.startswith(("# ", "## ", "### ")):
        # Den bisherigen Abschnitt speichern,
        # sofern er nicht leer ist.
        if current_chunk.strip():
            structure_chunks.append(current_chunk.strip())

        # Einen neuen Abschnitt mit der Überschrift beginnen.
        current_chunk = line + "\n"

    else:
        # Normale Textzeilen an den aktuellen Chunk anhängen.
        current_chunk += line + "\n"


if current_chunk.strip():
    structure_chunks.append(current_chunk.strip())

print("\nSTRUCTURE CHUNKING")
print("Anzahl Chunks:", len(structure_chunks))
print("Erster Chunk:\n", structure_chunks[0])
print("Zweiter Chunk:\n", structure_chunks[1])


