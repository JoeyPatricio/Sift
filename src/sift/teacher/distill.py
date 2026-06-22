"""Generate chain-of-thought fault-localization traces from the frontier teacher.

The teacher (``claude-opus-4-8`` by default) reasons backward from the traceback through the
candidate files and predicts the faulty line ranges. We capture two things per example:

- the **summarized reasoning** (thinking blocks) — this becomes the CoT the student learns;
- the **structured prediction** — a JSON list of {file, start_line, end_line}.

Two API choices matter here:

- ``thinking={"type": "adaptive", "display": "summarized"}`` — adaptive thinking gives the
  teacher room to reason backward through the stack; ``display: "summarized"`` is required to
  actually get the reasoning text back (the default ``"omitted"`` returns empty thinking
  blocks, which would leave us with nothing to distill).
- ``output_config.format`` with a JSON schema — guarantees a parseable prediction so the
  rejection-sampling step can compare it to the real diff without brittle text parsing.

For generating a full training corpus, prefer ``distill_batch`` (Batches API, 50% cost,
trace generation is offline and not latency-sensitive).
"""

from __future__ import annotations

import json

import anthropic

from sift.config import settings
from sift.schema import Example, FaultLocation, Trace

_SYSTEM = """You are a fault-localization expert. Given a failing test, its traceback, and a \
set of candidate source files, identify the file and line range MOST LIKELY responsible for \
the failure.

The traceback shows where the error surfaced, not necessarily where it originated. Reason \
backward through the call stack: the real fault is often a helper, a return value, or a type \
mismatch introduced upstream of where the exception fired. Cite specific lines.

Return the faulty locations as a ranked list, most likely first."""

# JSON schema for the structured prediction. additionalProperties:false is required.
_PREDICTION_SCHEMA = {
    "type": "object",
    "properties": {
        "locations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["file", "start_line", "end_line"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["locations"],
    "additionalProperties": False,
}


def _build_user_prompt(example: Example) -> str:
    files = "\n\n".join(
        f"### FILE: {f.path}\n{f.content}" for f in example.candidate_files
    )
    return (
        f"# Failing test\n{example.test_name}\n\n"
        f"# Traceback\n{example.traceback}\n\n"
        f"# Candidate files\n{files}"
    )


def _parse_response(content: list) -> tuple[str, list[FaultLocation]]:
    """Pull the summarized reasoning and the structured prediction out of a response."""
    reasoning_parts: list[str] = []
    prediction: list[FaultLocation] = []
    for block in content:
        if block.type == "thinking":
            reasoning_parts.append(block.thinking)
        elif block.type == "text":
            data = json.loads(block.text)
            prediction = [FaultLocation(**loc) for loc in data["locations"]]
    return "\n".join(reasoning_parts).strip(), prediction


def distill_one(example: Example, client: anthropic.Anthropic | None = None) -> Trace:
    """Generate a single CoT trace + prediction for one example (synchronous)."""
    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=settings.teacher_model,
        max_tokens=16000,
        system=_SYSTEM,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={
            "effort": settings.teacher_effort,
            "format": {"type": "json_schema", "schema": _PREDICTION_SCHEMA},
        },
        messages=[{"role": "user", "content": _build_user_prompt(example)}],
    )
    reasoning, prediction = _parse_response(response.content)
    return Trace(example_id=example.id, reasoning=reasoning, prediction=prediction)


def distill_batch(examples: list[Example], out_path: str) -> None:
    """Generate traces for many examples via the Batches API (50% cost).

    NOT YET IMPLEMENTED. Trace generation over a full training set is the right job for the
    Batches API — it's offline and not latency-sensitive. The synchronous ``distill_one`` is
    here so the loop, prompt, and parsing are correct first; the batch path reuses the same
    request shape (see ``sift.eval.harness`` for the per-example contract).
    """
    raise NotImplementedError(
        "Batch distillation not implemented yet. Use distill_one to validate the prompt + "
        "parsing on a handful of examples, then port the same request to "
        "client.messages.batches.create() over the full split."
    )
