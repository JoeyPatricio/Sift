"""BM25 file-level recall@k — calibration tool for picking bm25_top_k.

recall@k answers: "If we keep the top-k BM25 results, what fraction of examples
have their gold file in that set?"  This is the *upper bound* on any downstream
model's file-level acc@k — a model can only localize a file it was shown.  Running
this curve first tells you the ceiling before you ever train anything.
"""

from __future__ import annotations

from collections.abc import Sequence

from sift.retrieval.bm25 import BM25Retriever
from sift.schema import Example

_DEFAULT_KS = (1, 5, 10, 15, 20, 30, 50)
_DEFAULT_TARGET = 0.90  # recall threshold for suggest_top_k


def recall_at_k(examples: Sequence[Example], k: int) -> float:
    """Fraction of examples whose gold file appears in the BM25 top-k.

    Uses the example's ``test_name`` + ``traceback`` as the query, which mirrors
    exactly what ``RetrievalBaseline`` (and ultimately the student) will receive.
    Returns 0.0 for an empty example list.
    """
    if not examples:
        return 0.0
    hits = sum(1 for ex in examples if _gold_in_top_k(ex, k))
    return hits / len(examples)


def recall_curve(
    examples: Sequence[Example],
    ks: Sequence[int] = _DEFAULT_KS,
) -> dict[int, float]:
    """Compute file-level recall@k for every k in one BM25 pass per example.

    Returns a dict mapping each k to its recall value, sorted by k.
    """
    if not examples or not ks:
        return {k: 0.0 for k in ks}

    max_k = max(ks)
    hits: dict[int, int] = {k: 0 for k in ks}

    for ex in examples:
        retriever = BM25Retriever(ex.candidate_files)
        query = f"{ex.test_name}\n{ex.traceback}"
        # Rank once up to max_k; slice for each individual k below.
        top_paths = [c.path for c in retriever.query(query, max_k)]
        gold = ex.ground_truth_files

        for k in ks:
            if set(top_paths[:k]) & gold:
                hits[k] += 1

    n = len(examples)
    return {k: hits[k] / n for k in sorted(ks)}


def suggest_top_k(
    curve: dict[int, float],
    target_recall: float = _DEFAULT_TARGET,
) -> int:
    """Return the smallest k where recall >= target_recall.

    Falls back to the largest k in the curve if the target is never reached.
    """
    for k in sorted(curve):
        if curve[k] >= target_recall:
            return k
    return max(curve)


def _gold_in_top_k(ex: Example, k: int) -> bool:
    retriever = BM25Retriever(ex.candidate_files)
    top_paths = {c.path for c in retriever.query(f"{ex.test_name}\n{ex.traceback}", k)}
    return bool(top_paths & ex.ground_truth_files)
