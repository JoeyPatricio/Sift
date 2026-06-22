"""Rejection sampling: keep only teacher traces whose prediction matches the real diff.

This is the step that filters hallucinated reasoning before it reaches the student. The
teacher produces a CoT trace and a prediction; we keep the trace only if the prediction
actually matches the ground-truth fault location from the fixing commit. Ground truth is
always the real diff — never the teacher's own output.

A trace is accepted if its top prediction is a hit at the configured granularity. Using the
top prediction (not "any of them") keeps the bar honest: we're distilling traces where the
teacher's *primary* answer was right, not ones where it hedged across many guesses.
"""

from __future__ import annotations

from sift.eval.metrics import Granularity, is_hit
from sift.schema import Example, Trace


def accept(trace: Trace, example: Example, granularity: Granularity = "file") -> bool:
    """True if the trace's top prediction matches the example's ground truth."""
    if not trace.prediction:
        return False
    top = trace.prediction[0]
    return is_hit(top, example.ground_truth, granularity)


def filter_traces(
    traces: list[Trace],
    examples: dict[str, Example],
    granularity: Granularity = "file",
) -> list[Trace]:
    """Keep only accepted traces. ``examples`` is keyed by ``Example.id``."""
    return [t for t in traces if t.example_id in examples and accept(t, examples[t.example_id], granularity)]
