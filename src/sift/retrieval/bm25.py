"""BM25 retrieval over candidate files.

A real codebase has far too many files to feed every one to the model. We use the failing
test plus its traceback as the query and BM25 to rank the repo's source files, keeping only
the top-k. This keeps context tractable without sacrificing recall — the faulty file is
usually lexically close to the test that exercises it (shared identifiers, module names,
the symbols named in the traceback).

This is deliberately simple and dependency-light (``rank_bm25``). It's a baseline a stronger
retriever (embeddings, call-graph expansion) can be measured against later.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from sift.schema import CandidateFile

# Split on non-identifier characters and camelCase / snake_case boundaries so that
# `parse_config`, `parseConfig`, and `parse.config` all yield the tokens `parse` + `config`.
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Lowercased code-aware tokenization for BM25."""
    return [tok.lower() for tok in _TOKEN_SPLIT.split(text) if tok]


class BM25Retriever:
    """Rank candidate files for a failing test by lexical relevance.

    Built once per example over that example's candidate files (a repo snapshot), then
    queried with the test + traceback.
    """

    def __init__(self, candidates: Sequence[CandidateFile]) -> None:
        self.candidates = list(candidates)
        self._corpus_tokens = [tokenize(f.content) for f in self.candidates]
        # BM25Okapi requires a non-empty corpus; guard so an empty repo doesn't explode.
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def query(self, text: str, top_k: int) -> list[CandidateFile]:
        """Return the top-k candidate files most relevant to ``text``, best first."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(text))
        ranked = sorted(zip(self.candidates, scores), key=lambda cs: cs[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]


def rank_candidates(
    candidates: Sequence[CandidateFile],
    test_name: str,
    traceback: str,
    top_k: int,
) -> list[CandidateFile]:
    """Convenience wrapper: build a retriever and query it with the test + traceback."""
    retriever = BM25Retriever(candidates)
    return retriever.query(f"{test_name}\n{traceback}", top_k)
