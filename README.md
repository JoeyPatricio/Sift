# Sift

**Fault localization for failing tests.** When a test fails, the traceback shows where an
error *surfaced*, not where it *started*. In a real codebase those two locations are often far
apart — the exception fires downstream while the actual fault sits in a helper function, a
return value, or a type mismatch introduced several calls earlier. Finding the origin means
reasoning backward through the call stack.

Sift automates that. Given a failing test, its traceback, and a set of candidate source files,
it predicts the specific **file and line range** most likely responsible.

## How it works

```
failing test + traceback + repo
          │
          ▼
   ┌─────────────┐   scope the codebase to the most
   │  BM25 retr. │   relevant candidate files (keeps
   └─────────────┘   context tractable without losing recall)
          │
          ▼
   ┌─────────────┐   QLoRA-finetuned 3B model predicts
   │   student   │   { file, start_line, end_line }
   └─────────────┘
          │
          ▼
   acc@1 / acc@3 / MRR  vs. the real bug-fix diff
```

The student is a **QLoRA fine-tuned 3B open model**, trained on rejection-sampled
chain-of-thought traces distilled from a frontier teacher (`claude-opus-4-8`).

- **Training data** comes from real bug-fix commits: the pre-fix codebase state, the failing
  test, and the changed lines as ground truth.
- **Distillation with rejection sampling** — the teacher generates CoT traces; only traces
  where the teacher's prediction matches the actual commit diff are kept. This filters
  hallucinated reasoning before it reaches the student.
- **Ground truth is always the real diff**, never the teacher's output.

The student aims to match frontier-level acc@1 at a fraction of the inference cost and
latency — practical in CI, where a frontier API call per test failure doesn't scale.

## Status

Early scaffold. The eval harness, metrics, retrieval, and data schema are implemented;
the data-mining, teacher, and training pipelines are stubbed with clear interfaces. See
[docs/architecture.md](docs/architecture.md) for the build plan and open design questions
(dataset choice, line-range granularity, leakage controls).

## Install

```bash
# Core (retrieval + eval + teacher) — runs on a laptop, no GPU needed
pip install -e ".[dev]"

# Training stack — install on a CUDA box
pip install -e ".[train]"
```

## CLI

```bash
sift mine        --out data/raw              # mine bug-fix commits → examples
sift retrieve    --example <id>              # BM25 candidate files for one example
sift distill     --split train --out data/traces.jsonl   # teacher CoT (rejection-sampled)
sift train       --traces data/traces.jsonl  # QLoRA fine-tune the student
sift eval        --model <path> --split test # acc@1 / acc@3 / MRR
```

(Stubbed commands print a clear "not implemented yet" with the intended contract.)

## Metrics

Standard fault-localization metrics on a held-out set of real bug-fix commits, enabling direct
comparison to published baselines:

- **acc@k** — is the true faulty location in the top-k predictions?
- **MRR** — mean reciprocal rank of the true location.

Reported at both **file** and **line** granularity (line-level is much harder; reporting both
keeps the story honest). Ground truth is always the actual diff.

## License

MIT
