"""Fault-localization metrics: acc@k and MRR, at file and line granularity.

A *prediction* is a ranked list of ``FaultLocation`` (best first). *Ground truth* is the set
of locations the fixing commit actually changed. A predicted location counts as a hit if it
matches any ground-truth location under the chosen granularity:

- **file**: same file.
- **line**: same file AND overlapping line range.

These are the standard metrics in the fault-localization literature, so numbers here are
directly comparable to published baselines.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sift.schema import FaultLocation

Granularity = Literal["file", "line"]


def is_hit(pred: FaultLocation, truth: Sequence[FaultLocation], granularity: Granularity) -> bool:
    """Does ``pred`` match any ground-truth location at the given granularity?"""
    if granularity == "file":
        return any(pred.file == t.file for t in truth)
    return any(pred.overlaps(t) for t in truth)


def first_hit_rank(
    predictions: Sequence[FaultLocation],
    truth: Sequence[FaultLocation],
    granularity: Granularity,
) -> int | None:
    """1-indexed rank of the first correct prediction, or None if none are correct."""
    for rank, pred in enumerate(predictions, start=1):
        if is_hit(pred, truth, granularity):
            return rank
    return None


def acc_at_k(
    predictions: Sequence[FaultLocation],
    truth: Sequence[FaultLocation],
    k: int,
    granularity: Granularity,
) -> float:
    """1.0 if a correct location appears in the top-k predictions, else 0.0."""
    rank = first_hit_rank(predictions[:k], truth, granularity)
    return 1.0 if rank is not None else 0.0


def reciprocal_rank(
    predictions: Sequence[FaultLocation],
    truth: Sequence[FaultLocation],
    granularity: Granularity,
) -> float:
    """1 / (rank of first correct prediction), or 0.0 if none are correct."""
    rank = first_hit_rank(predictions, truth, granularity)
    return 1.0 / rank if rank is not None else 0.0


@dataclass
class MetricSummary:
    """Aggregate metrics over a dataset, at one granularity."""

    granularity: Granularity
    n: int
    acc_at_1: float
    acc_at_3: float
    mrr: float

    def __str__(self) -> str:
        return (
            f"[{self.granularity}] n={self.n}  "
            f"acc@1={self.acc_at_1:.3f}  acc@3={self.acc_at_3:.3f}  MRR={self.mrr:.3f}"
        )


def evaluate(
    predictions: Sequence[Sequence[FaultLocation]],
    truths: Sequence[Sequence[FaultLocation]],
    granularity: Granularity = "file",
    ks: Sequence[int] = (1, 3),
) -> MetricSummary:
    """Aggregate acc@1, acc@3, and MRR across a dataset.

    ``predictions[i]`` is the ranked prediction list for example ``i``; ``truths[i]`` is its
    ground truth. The two sequences must be the same length and aligned by index.
    """
    if len(predictions) != len(truths):
        raise ValueError(f"predictions ({len(predictions)}) and truths ({len(truths)}) differ")
    n = len(predictions)
    if n == 0:
        return MetricSummary(granularity, 0, 0.0, 0.0, 0.0)

    acc1 = sum(acc_at_k(p, t, 1, granularity) for p, t in zip(predictions, truths)) / n
    acc3 = sum(acc_at_k(p, t, 3, granularity) for p, t in zip(predictions, truths)) / n
    mrr = sum(reciprocal_rank(p, t, granularity) for p, t in zip(predictions, truths)) / n
    return MetricSummary(granularity, n, acc1, acc3, mrr)
