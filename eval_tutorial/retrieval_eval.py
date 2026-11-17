"""Retrieval-Metriken mit Teralune-Goldfragen und simulierten Rankings.

Start im Projektverzeichnis:
    uv run python -m eval_tutorial.retrieval_eval

Keine OpenAI-API und keine Chroma-Datenbank erforderlich.
Die Rankings sind bewusst fest vorgegeben und KEINE gemessenen Suchergebnisse.
"""

import json
from pathlib import Path

GOLDSET_FILE = Path(__file__).with_name("goldset.json")
TOP_K = 3

# Simulierte Trefferlisten aus dem Teralune-Korpus.
# Die Reihenfolge dient der Demonstration der Metriken.
SIMULATED_RANKINGS = {
    "Q001": ["D004_2", "D033_1", "D002_1"],
    "Q002": ["D011_1", "D004_1", "D003_1"],
    "Q003": ["D022_0", "D022_1", "D022_2"],
    "Q004": ["D003_2", "D003_3", "D003_4"],
    "Q005": ["D034_2", "D005_2", "D033_2"],
}


def load_goldset():
    return json.loads(GOLDSET_FILE.read_text(encoding="utf-8"))


def hit_at_k(gold_ids: set[str], retrieved_ids: list[str], k: int) -> int:
    return int(bool(gold_ids.intersection(retrieved_ids[:k])))


def precision_at_k(gold_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """
    Welcher Anteil der Top-k-Treffer ist relevant?

    Precision@k = relevante Treffer / k

    Beispiel: Gold = {A, B}, Top-3 = [X, A, B]
    Precision@3 = 2/3 = 66,7 %

    Je höher die Precision, desto weniger irrelevante Treffer.
    """
    top_k_ids = retrieved_ids[:k]
    relevant_ids = gold_ids.intersection(top_k_ids)
    true_positvives = len(relevant_ids)
    precision = true_positvives / k 
    return precision


def recall_at_k(gold_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    """
    Welchen Anteil aller relevanten Gold-Chunks haben wir gefunden?

    Recall@k = gefundene relevante Chunks / alle Gold-Chunks

    Beispiel: Gold = {A, B, C, D}, Top-3 = [X, A, B]
    Recall@3 = 2/4 = 50 %

    Je höher der Recall, desto weniger relevante Chunks fehlen.
    """
    if not gold_ids:
        return 0.0

    top_k_ids = retrieved_ids[:k]
    relevant_ids = gold_ids.intersection(top_k_ids)
    true_positives = len(relevant_ids)
    recall = true_positives / len(gold_ids)
    return recall
    


def reciprocal_rank_at_k(gold_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in gold_ids:
            return 1 / rank
    return 0.0


def main():
    cases = load_goldset()
    totals = {"hit": 0.0, "precision": 0.0, "recall": 0.0, "rr": 0.0}

    print("RETRIEVAL-EVALUATION: TERALUNE")
    print("Simulierte Rankings (keine echten Chroma-Suchergebnisse)")
    print(f"Top K: {TOP_K}\n")

    for case in cases:
        case_id = case["id"]
        if case_id not in SIMULATED_RANKINGS:
            raise ValueError(f"Kein simuliertes Ranking für {case_id}")

        gold_ids: set[str] = {item["chunk_id"] for item in case["evidence"]}
        retrieved_ids = SIMULATED_RANKINGS[case_id]

        hit = hit_at_k(gold_ids, retrieved_ids, TOP_K)
        precision = precision_at_k(gold_ids, retrieved_ids, TOP_K)
        recall = recall_at_k(gold_ids, retrieved_ids, TOP_K)
        rr = reciprocal_rank_at_k(gold_ids, retrieved_ids, TOP_K)

        totals["hit"] += hit
        totals["precision"] += precision
        totals["recall"] += recall
        totals["rr"] += rr

        print(f"{case_id}: {case['question']}")
        print(f"  Gold-Antwort: {case['expected_answer']}")
        print(f"  Gold-Chunks: {', '.join(sorted(gold_ids))}")
        for rank, chunk_id in enumerate(retrieved_ids[:TOP_K], start=1):
            label = "RELEVANT" if chunk_id in gold_ids else "nicht relevant"
            print(f"  Rang {rank}: {chunk_id} ({label})")
        print(f"  Hit@{TOP_K}:       {hit}")
        print(f"  Precision@{TOP_K}: {precision:.3f}")
        print(f"  Recall@{TOP_K}:    {recall:.3f}")
        print(f"  RR@{TOP_K}:        {rr:.3f}\n")

    n = len(cases)
    if not n:
        raise ValueError("Das Goldset ist leer.")
    print("GESAMTERGEBNIS")
    print(f"Goldfragen:       {n}")
    print(f"Mean Hit@{TOP_K}:       {totals['hit'] / n:.3f}")
    print(f"Mean Precision@{TOP_K}: {totals['precision'] / n:.3f}")
    print(f"Mean Recall@{TOP_K}:    {totals['recall'] / n:.3f}")
    print(f"MRR@{TOP_K}:            {totals['rr'] / n:.3f}")


if __name__ == "__main__":
    main()
