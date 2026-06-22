"""Run a Predictor over a dataset and report acc@1 / acc@3 / MRR.

Model-agnostic: give it any ``Predictor`` (student, teacher baseline, retrieval baseline) and
a list of examples, and it produces metrics at both file and line granularity. Reporting both
granularities is deliberate — line-level localization on real diffs is hard even for frontier
models, so a weak line score shouldn't be hidden behind a strong file score.
"""

from __future__ import annotations

from sift.eval.metrics import MetricSummary, evaluate
from sift.infer.predict import Predictor
from sift.schema import Example


def run_eval(predictor: Predictor, examples: list[Example]) -> dict[str, MetricSummary]:
    """Evaluate ``predictor`` over ``examples`` at file and line granularity.

    Returns a dict keyed by granularity ("file", "line").
    """
    predictions = [predictor.predict(ex) for ex in examples]
    truths = [ex.ground_truth for ex in examples]
    return {
        "file": evaluate(predictions, truths, granularity="file"),
        "line": evaluate(predictions, truths, granularity="line"),
    }
