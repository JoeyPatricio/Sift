"""Run a model to predict fault locations for an example.

Defines ``Predictor`` — the single interface the eval harness depends on. Any model that
produces a ranked list of ``FaultLocation`` for an ``Example`` can be evaluated: the trained
student, the frontier teacher (as a baseline), or a trivial retrieval-only baseline.

This keeps the harness model-agnostic. The portfolio story leans on comparing all three under
the same metrics, so they share one interface.
"""

from __future__ import annotations

from typing import Protocol

from sift.config import settings
from sift.retrieval import rank_candidates
from sift.schema import Example, FaultLocation


class Predictor(Protocol):
    """Anything that ranks fault locations for an example."""

    def predict(self, example: Example) -> list[FaultLocation]:
        """Return a ranked list of candidate fault locations, best first."""
        ...


class RetrievalBaseline:
    """Trivial baseline: blame line 1 of each top-ranked candidate file.

    Establishes the floor that the student must beat — file-level acc@k here is just "did
    BM25 surface the right file", with no reasoning at all. If the student can't beat this,
    the model isn't earning its keep.
    """

    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or settings.bm25_top_k

    def predict(self, example: Example) -> list[FaultLocation]:
        ranked = rank_candidates(
            example.candidate_files, example.test_name, example.traceback, self.top_k
        )
        return [FaultLocation(file=f.path, start_line=1, end_line=f.num_lines) for f in ranked]


class StudentPredictor:
    """The trained QLoRA student. NOT YET IMPLEMENTED.

    Will load the adapter, build the same prompt used in training (see sift.train.qlora),
    run generation, and parse the structured prediction — mirroring sift.teacher.distill.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def predict(self, example: Example) -> list[FaultLocation]:
        raise NotImplementedError(
            "Student inference is not implemented yet. It mirrors sift.teacher.distill: "
            "build the training-time prompt, generate, parse the JSON prediction. The prompt "
            "must match sift.train.qlora exactly."
        )
