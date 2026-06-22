from sift.retrieval.bm25 import rank_candidates, tokenize
from sift.schema import CandidateFile


def test_tokenize_splits_identifiers():
    assert tokenize("parse_config") == ["parse", "config"]
    assert tokenize("parseConfig") == ["parse", "config"]
    assert tokenize("foo.bar(baz)") == ["foo", "bar", "baz"]


def test_rank_surfaces_relevant_file_first():
    # Several distractor files so BM25's IDF is meaningful — on a tiny corpus a term
    # appearing in half the docs gets ~zero IDF, which is a small-corpus artifact, not
    # a property of real repos.
    candidates = [
        CandidateFile(path="add.py", content="def add(a, b):\n    return a + b\n"),
        CandidateFile(path="sub.py", content="def subtract(a, b):\n    return a - b\n"),
        CandidateFile(path="mul.py", content="def multiply(a, b):\n    return a * b\n"),
        CandidateFile(path="io.py", content="def read_file(path):\n    return open(path).read()\n"),
        CandidateFile(
            path="config.py",
            content="def parse_config(path):\n    raise ValueError('bad config')\n",
        ),
    ]
    ranked = rank_candidates(
        candidates,
        test_name="test_parse_config",
        traceback="ValueError: bad config in parse_config",
        top_k=3,
    )
    assert ranked[0].path == "config.py"


def test_top_k_caps_results():
    candidates = [CandidateFile(path=f"f{i}.py", content="x = 1\n") for i in range(5)]
    assert len(rank_candidates(candidates, "t", "tb", top_k=3)) == 3


def test_empty_repo_returns_nothing():
    assert rank_candidates([], "t", "tb", top_k=3) == []
