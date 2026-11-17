"""
Station 07: Reciprocal Rank Fusion (RRF)

Lernziel:
Zwei Rankings anhand ihrer Rangpositionen kombinieren.

Formel:
RRF-Beitrag = 1 / (k + Rang)

Start:
uv run python -m stations.07_rrf
"""

RRF_K = 60

bm25 = ["A", "B", "C"]
dense = ["B", "C", "A"]


def reciprocal_rank_fusion(rankings: list, k=60):

    scores = {}

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            contribution = 1 / (k + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + contribution

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return ranked


def main():
    print("RECIPROCAL RANK FUSION")

    print("\nBM25-Ranking:")
    for rank, chunk_id in enumerate(bm25, start=1):
        print(f"  {rank}. Chunk {chunk_id}")

    print("\nDense-Ranking:")
    for rank, chunk_id in enumerate(dense, start=1):
        print(f"  {rank}. Chunk {chunk_id}")
   
    fused = reciprocal_rank_fusion([bm25, dense], k=RRF_K)
    print("\nRRF-Ranking:")

    for rank, (chunk_id, score) in enumerate(fused, start=1):
        print(f"  {rank}. Chunk {chunk_id} | RRF: {score:.6f}")


if __name__ == "__main__":
    main()
