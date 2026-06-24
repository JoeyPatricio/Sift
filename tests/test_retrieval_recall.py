"""Tests for sift.retrieval.recall — file-level recall@k calibration."""

from __future__ import annotations

from sift.retrieval.recall import recall_at_k, recall_curve, suggest_top_k
from sift.schema import CandidateFile, Example, FaultLocation


def _make_example(
    gold_path: str,
    candidate_paths: list[str],
    query_hint: str = "",
) -> Example:
    """Build a minimal Example where gold_path is the correct file."""
    candidates = [
        CandidateFile(path=p, content=f"# {p}\n{query_hint if p == gold_path else ''}")
        for p in candidate_paths
    ]
    return Example(
        id=f"test@abc::{gold_path}",
        repo="org/repo",
        commit="abc123",
        test_name=f"test_{gold_path.replace('.py', '')} {query_hint}",
        traceback=f"Error in {gold_path}",
        candidate_files=candidates,
        ground_truth=[FaultLocation(file=gold_path, start_line=1, end_line=5)],
    )


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

def test_recall_at_k_empty_examples():
    assert recall_at_k([], k=5) == 0.0


def test_recall_at_k_gold_in_top1():
    # Gold file has unique query tokens so BM25 should rank it first.
    ex = _make_example(
        gold_path="parser.py",
        candidate_paths=["parser.py", "utils.py", "io.py", "config.py", "db.py"],
        query_hint="parse_config zygote",
    )
    # If gold is in top-1, recall@1 should be 1.0.
    assert recall_at_k([ex], k=1) == 1.0


def test_recall_at_k_gold_not_in_top1_but_in_top3():
    # Gold file shares few tokens with the query; other files dominate at k=1.
    ex = _make_example(
        gold_path="obscure.py",
        candidate_paths=["obscure.py", "utils.py", "io.py", "config.py", "db.py"],
        query_hint="",
    )
    # We can't guarantee rank here without controlling BM25 scores, but we can assert
    # that recall is in [0, 1] and consistent across two calls.
    r1 = recall_at_k([ex], k=1)
    r5 = recall_at_k([ex], k=5)
    assert 0.0 <= r1 <= 1.0
    assert r5 >= r1  # more candidates can only help


def test_recall_at_k_monotone_in_k():
    examples = [
        _make_example("a.py", ["a.py", "b.py", "c.py", "d.py", "e.py"], "alpha beta"),
        _make_example("c.py", ["a.py", "b.py", "c.py", "d.py", "e.py"], "gamma delta"),
    ]
    prev = 0.0
    for k in [1, 2, 3, 4, 5]:
        r = recall_at_k(examples, k)
        assert r >= prev, f"recall@{k}={r} < recall@{k-1}={prev}"
        prev = r


def test_recall_at_k_all_gold_rank1():
    examples = [
        _make_example(
            f"mod{i}.py",
            [f"mod{i}.py"] + [f"other{j}.py" for j in range(10)],
            f"unique_token_{i}",
        )
        for i in range(5)
    ]
    # With unique tokens per example, BM25 should rank each gold file first.
    assert recall_at_k(examples, k=1) == 1.0


# ---------------------------------------------------------------------------
# recall_curve
# ---------------------------------------------------------------------------

def test_recall_curve_keys_match_ks():
    examples = [
        _make_example("x.py", ["x.py", "y.py", "z.py"], "unique_xyz"),
    ]
    ks = [1, 3, 5]
    curve = recall_curve(examples, ks)
    assert set(curve.keys()) == {1, 3, 5}


def test_recall_curve_empty_examples():
    curve = recall_curve([], ks=[1, 5, 10])
    assert all(v == 0.0 for v in curve.values())


def test_recall_curve_monotone():
    examples = [
        _make_example("a.py", ["a.py", "b.py", "c.py", "d.py", "e.py"], "token_a"),
        _make_example("e.py", ["a.py", "b.py", "c.py", "d.py", "e.py"], "token_e"),
    ]
    ks = [1, 2, 3, 4, 5]
    curve = recall_curve(examples, ks)
    values = [curve[k] for k in ks]
    assert values == sorted(values), "recall@k must be non-decreasing in k"


def test_recall_curve_returns_sorted_keys():
    examples = [_make_example("f.py", ["f.py", "g.py"], "hint")]
    curve = recall_curve(examples, ks=[10, 1, 5])
    assert list(curve.keys()) == [1, 5, 10]


# ---------------------------------------------------------------------------
# suggest_top_k
# ---------------------------------------------------------------------------

def test_suggest_top_k_returns_first_above_target():
    curve = {1: 0.5, 5: 0.75, 10: 0.88, 15: 0.91, 20: 0.95}
    assert suggest_top_k(curve, target_recall=0.90) == 15


def test_suggest_top_k_target_never_reached():
    curve = {1: 0.3, 5: 0.5, 10: 0.7}
    # Falls back to largest k.
    assert suggest_top_k(curve, target_recall=0.99) == 10


def test_suggest_top_k_already_hit_at_k1():
    curve = {1: 1.0, 5: 1.0, 10: 1.0}
    assert suggest_top_k(curve, target_recall=0.90) == 1
