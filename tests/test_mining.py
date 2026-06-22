"""Unit tests for the pure helper functions in sift.data.mining."""

from __future__ import annotations

from pathlib import Path

import pytest

from sift.data.mining import _collect_candidates, _is_source_file, _parse_test_list
from sift.schema import CandidateFile


# ---------------------------------------------------------------------------
# _parse_test_list
# ---------------------------------------------------------------------------

def test_parse_test_list_already_a_list():
    assert _parse_test_list(["a::b", "c::d"]) == ["a::b", "c::d"]


def test_parse_test_list_python_literal_string():
    assert _parse_test_list("['tests/foo.py::test_bar']") == ["tests/foo.py::test_bar"]


def test_parse_test_list_json_string():
    assert _parse_test_list('["tests/foo.py::test_bar"]') == ["tests/foo.py::test_bar"]


def test_parse_test_list_empty_string():
    assert _parse_test_list("") == []


def test_parse_test_list_empty_list():
    assert _parse_test_list([]) == []


def test_parse_test_list_none():
    assert _parse_test_list(None) == []


# ---------------------------------------------------------------------------
# _is_source_file
# ---------------------------------------------------------------------------

def test_is_source_file_plain_module():
    assert _is_source_file("sift/retrieval/bm25.py") is True


def test_is_source_file_top_level():
    assert _is_source_file("setup.py") is True


def test_is_source_file_in_tests_dir():
    assert _is_source_file("tests/test_foo.py") is False


def test_is_source_file_test_prefix():
    assert _is_source_file("sift/test_utils.py") is False


def test_is_source_file_test_suffix():
    assert _is_source_file("sift/utils_test.py") is False


def test_is_source_file_conftest():
    assert _is_source_file("conftest.py") is False


def test_is_source_file_nested_conftest():
    assert _is_source_file("sift/conftest.py") is False


def test_is_source_file_vendor_dir():
    assert _is_source_file("vendor/requests/api.py") is False


def test_is_source_file_migrations_dir():
    assert _is_source_file("app/migrations/0001_initial.py") is False


def test_is_source_file_non_python():
    assert _is_source_file("sift/utils.js") is False


def test_is_source_file_docs_dir():
    assert _is_source_file("docs/conf.py") is False


# ---------------------------------------------------------------------------
# _collect_candidates
# ---------------------------------------------------------------------------

def test_collect_candidates_basic(tmp_path: Path):
    (tmp_path / "mymod.py").write_text("x = 1")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_mymod.py").write_text("def test_x(): pass")

    candidates = _collect_candidates(tmp_path)
    paths = [c.path for c in candidates]

    assert "mymod.py" in paths
    assert not any("test_" in p for p in paths)


def test_collect_candidates_content(tmp_path: Path):
    (tmp_path / "utils.py").write_text("def helper(): pass\n")
    candidates = _collect_candidates(tmp_path)
    assert candidates[0].content == "def helper(): pass\n"


def test_collect_candidates_excludes_vendor(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "core.py").write_text("")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.py").write_text("")

    paths = [c.path for c in _collect_candidates(tmp_path)]
    assert "src/core.py" in paths
    assert not any("vendor" in p for p in paths)


def test_collect_candidates_returns_candidate_file_instances(tmp_path: Path):
    (tmp_path / "foo.py").write_text("pass")
    candidates = _collect_candidates(tmp_path)
    assert all(isinstance(c, CandidateFile) for c in candidates)
