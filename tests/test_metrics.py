from sift.eval.metrics import acc_at_k, evaluate, first_hit_rank, reciprocal_rank
from sift.schema import FaultLocation


def loc(file: str, start: int, end: int) -> FaultLocation:
    return FaultLocation(file=file, start_line=start, end_line=end)


TRUTH = [loc("a.py", 10, 12)]


def test_file_hit_ignores_lines():
    preds = [loc("a.py", 99, 99)]  # right file, wrong lines
    assert acc_at_k(preds, TRUTH, 1, "file") == 1.0
    assert acc_at_k(preds, TRUTH, 1, "line") == 0.0


def test_line_hit_requires_overlap():
    assert acc_at_k([loc("a.py", 11, 11)], TRUTH, 1, "line") == 1.0  # inside range
    assert acc_at_k([loc("a.py", 12, 20)], TRUTH, 1, "line") == 1.0  # boundary overlap
    assert acc_at_k([loc("a.py", 13, 20)], TRUTH, 1, "line") == 0.0  # just past


def test_acc_at_k_respects_k():
    preds = [loc("wrong.py", 1, 1), loc("a.py", 10, 12)]  # correct one is rank 2
    assert acc_at_k(preds, TRUTH, 1, "file") == 0.0
    assert acc_at_k(preds, TRUTH, 3, "file") == 1.0


def test_first_hit_rank_and_mrr():
    preds = [loc("x.py", 1, 1), loc("y.py", 1, 1), loc("a.py", 10, 12)]
    assert first_hit_rank(preds, TRUTH, "file") == 3
    assert reciprocal_rank(preds, TRUTH, "file") == 1 / 3


def test_no_hit():
    preds = [loc("x.py", 1, 1)]
    assert first_hit_rank(preds, TRUTH, "file") is None
    assert reciprocal_rank(preds, TRUTH, "file") == 0.0


def test_evaluate_aggregates():
    predictions = [
        [loc("a.py", 10, 12)],          # acc@1 hit, rr=1
        [loc("x.py", 1, 1), loc("b.py", 5, 5)],  # rank-2 hit, rr=0.5
    ]
    truths = [[loc("a.py", 10, 12)], [loc("b.py", 5, 5)]]
    summary = evaluate(predictions, truths, granularity="file")
    assert summary.n == 2
    assert summary.acc_at_1 == 0.5
    assert summary.acc_at_3 == 1.0
    assert summary.mrr == 0.75


def test_evaluate_empty():
    summary = evaluate([], [], granularity="file")
    assert summary.n == 0 and summary.mrr == 0.0
