"""Core data types, shared across every stage of the pipeline.

These are the contract between mining, retrieval, distillation, training, and eval.
Keeping them in one place means a change to "what an example is" is a change in one file.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CandidateFile(BaseModel):
    """A source file in scope for a given failing test."""

    path: str
    content: str

    @property
    def num_lines(self) -> int:
        return self.content.count("\n") + 1


class FaultLocation(BaseModel):
    """A predicted or ground-truth fault location: a line range within one file.

    Line numbers are 1-indexed and inclusive, matching how diffs and editors count.
    A single-line fault has ``start_line == end_line``.
    """

    file: str
    start_line: int
    end_line: int

    def overlaps(self, other: FaultLocation) -> bool:
        """True if the two locations are in the same file and their ranges intersect.

        This is the line-level match predicate used by the metrics. File-level match
        is just ``self.file == other.file``.
        """
        if self.file != other.file:
            return False
        return self.start_line <= other.end_line and other.start_line <= self.end_line


class Example(BaseModel):
    """One fault-localization instance, mined from a real bug-fix commit.

    The model sees ``test_name``, ``traceback``, and the retrieved subset of
    ``candidate_files``; it must predict ``ground_truth``. The ground truth is the
    set of lines the fixing commit actually changed — never a teacher prediction.
    """

    id: str = Field(..., description="Stable id, e.g. '<repo>@<commit>::<test>'")
    repo: str
    commit: str = Field(..., description="The fixing commit SHA (post-fix)")
    test_name: str
    traceback: str
    candidate_files: list[CandidateFile] = Field(default_factory=list)
    # A bug fix may touch several places — ground truth is a set of locations.
    ground_truth: list[FaultLocation] = Field(default_factory=list)

    @property
    def ground_truth_files(self) -> set[str]:
        return {loc.file for loc in self.ground_truth}


class Trace(BaseModel):
    """A teacher-generated chain-of-thought trace plus its prediction.

    Produced by the distillation step. Only traces whose ``prediction`` matches the
    example's ``ground_truth`` (see ``teacher.reject``) become training data.
    """

    example_id: str
    reasoning: str = Field(..., description="Teacher CoT (summarized thinking)")
    prediction: list[FaultLocation]
