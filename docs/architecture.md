# Architecture

## Pipeline

```
mine → retrieve → distill → reject → train → eval
```

| Stage    | Module               | Status | What it does |
|----------|----------------------|--------|--------------|
| mine     | `data.mining`        | stub   | Bug-fix commits → `Example`s (pre-fix repo, failing test, traceback, ground-truth diff) |
| —        | `data.ground_truth`  | ✅     | Parse a fixing diff into ground-truth `FaultLocation`s |
| retrieve | `retrieval.bm25`     | ✅     | BM25 scope the repo to top-k candidate files |
| distill  | `teacher.distill`    | partial| Teacher CoT + structured prediction (`distill_one` works; batch path stubbed) |
| reject   | `teacher.reject`     | ✅     | Keep only traces whose prediction matches the real diff |
| train    | `train.qlora`        | stub   | QLoRA fine-tune the 3B student on accepted traces |
| infer    | `infer.predict`      | partial| `Predictor` interface + retrieval baseline (`StudentPredictor` stubbed) |
| eval     | `eval.metrics`/`harness` | ✅ | acc@1 / acc@3 / MRR at file + line granularity |

The data types in `schema.py` are the contract between every stage.

## Key design decisions (made)

- **Ground truth is always the real diff**, never the teacher's prediction. The teacher's
  output is only used as *reasoning* to distill, and only when its prediction was correct.
- **Metrics at two granularities.** File-level and line-level reported separately. Line-level
  is much harder; hiding it inside a file-level average would overstate the result.
- **Retrieval baseline ships as a `Predictor`.** Student, teacher, and retrieval-only all
  share one interface so they're compared under identical metrics — the core portfolio claim.
- **Teacher = `claude-opus-4-8`**, adaptive thinking + `display: "summarized"` (needed to
  capture CoT), `effort: high`, structured output for the prediction. Batch generation over
  the Batches API (50% cost) since distillation is offline.
- **Heavy training stack is an optional extra** (`[train]`) so the repo installs and the eval
  harness runs without a GPU.

## Dataset (decided)

**SWE-bench** (Python). Real GitHub PRs with `FAIL_TO_PASS` tests and a reproducible Docker
harness. Splits:

- **test** = SWE-bench **Verified** (500, human-validated)
- **dev** = **Lite** − Verified (fast iteration)
- **train** = **full** − Verified − Lite (~1.8k)

Rationale: most credible/visible benchmark, turnkey reproducible test execution, Python
(matches the student), and large repos make the BM25 retrieval stage do real work. Defects4J
is a possible later cross-language generalization eval, not the foundation. See
[roadmap.md](roadmap.md) for the build plan.

## Open questions (narrowed)

1. **Traceback capture.** SWE-bench ships test *names* (`FAIL_TO_PASS`), not failure text. A
   real traceback requires running the failing test against the pre-fix commit in the Docker
   harness — the main Phase-0 infra cost. Start on a subset to validate the pipeline before
   scaling. (Traceback is core to the thesis, so we capture real ones, not a heuristic.)
2. **Line-range granularity.** Report file-level and line-level separately from day one; the
   `overlaps` predicate in `schema.py` already supports line-range matching.
3. **Leakage.** A 3B model trained on public GitHub may have seen these (famous) repos. The
   held-out story needs date/repo-based analysis + dedup, or reviewers discount the headline
   number. Addressed in Phase 5.
