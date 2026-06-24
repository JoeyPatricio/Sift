# Sift

**Fault localization for failing tests.** When a test fails, the traceback shows where an
error *surfaced*, not where it *started*. In a real codebase those two locations are often far
apart — the exception fires downstream while the actual fault sits in a helper function, a
return value, or a type mismatch introduced several calls earlier. Finding the origin means
reasoning backward through the call stack.

Sift automates that. Given a failing test, its traceback, and a set of candidate source files,
it predicts the specific **file and line range** most likely responsible.

Built within a **$60 total API budget** (all phases combined): the teacher is `claude-haiku-4-5`
via the Batches API (50% discount), distilling reasoning traces over the SWE-bench train split
(~1,800 examples). Phase 2 distillation runs ~$27; teacher ceiling evaluation on the test split
(Phase 4) runs ~$8; GPU training (Phase 3) is free on Kaggle. This is enough to produce a
student that meaningfully beats the BM25 retrieval baseline at a fraction of frontier inference
cost — see [Scaling Up](#scaling-up) for what a higher budget buys.

## How it works

```
failing test + traceback + repo
          │
          ▼
   ┌─────────────┐   BM25 keyword match + nomic-embed-code
   │   hybrid    │   cosine similarity; ranked lists merged
   │   retr.     │   via RRF → top-k candidate files
   └─────────────┘
          │
          ▼
   ┌─────────────┐   QLoRA-finetuned 3B model predicts
   │   student   │   ranked { file, start_line, end_line,
   └─────────────┘   confidence } × 2–3
          │
          ▼
   acc@1 / acc@3 / MRR  vs. the real bug-fix diff
```

The student is a **QLoRA fine-tuned 3B open model**, trained on rejection-sampled
chain-of-thought traces distilled from a teacher (`claude-haiku-4-5`).

- **Training data** comes from real bug-fix commits: the pre-fix codebase state, the failing
  test, and the changed lines as ground truth.
- **Distillation with rejection sampling** — the teacher generates CoT traces; only traces
  where the teacher's prediction matches the actual commit diff are kept. This filters
  hallucinated reasoning before it reaches the student.
- **Ground truth is always the real diff**, never the teacher's output.

The student aims to match frontier-level acc@1 at a fraction of the inference cost and
latency — practical in CI, where a frontier API call per test failure doesn't scale.

## Status

- **Phase 0 — Data foundation** ✓ `data/dev.jsonl` mined (207 examples, 100% traceback capture)
- **Phase 1 — Retrieval + eval harness** ✓ BM25 recall@k curve on dev; baseline floor established
- **Phase 2 — Teacher distillation** in progress — Haiku teacher, Batches API, train split
- **Phase 3 — QLoRA fine-tune** pending GPU
- **Phase 4 — Headline comparison** pending
- **Phase 5 — Credibility** pending — leakage controls + ablations
- **Phase 6 — Hybrid retrieval** pending — BM25 + dense embeddings; higher recall ceiling
- **Phase 7 — Multi-fault + confidence scores** pending — ranked predictions with calibration
- **Phase 8 — CI/GitHub Action integration** pending — local inference on Actions runner, PR comments

See [docs/roadmap.md](docs/roadmap.md) for the full plan.

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
sift embed       --repo .                    # build/update dense embedding cache
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

## Scaling Up

The same pipeline with a higher budget produces a substantially stronger student:

**Stronger teacher.** Swap `claude-haiku-4-5` for `claude-opus-4-8` with `effort=high`.
Opus reasons more carefully through multi-hop call stacks and produces higher-quality CoT
traces, which raises the acceptance rate from ~30% to ~60%+ and gives the student better
reasoning to imitate. Estimated cost at the same scale: ~$400–600 via the Batches API.

**More training data.** Mine and distill the full train split (~1,800 examples) plus dev,
targeting 800–1,200 accepted traces. More data is the single largest lever on student acc@1.

**Larger student.** Upgrade from `Qwen2.5-Coder-3B` to the 7B or 14B variant. The 3B model
is constrained by capacity, not training data — a 7B student on the same traces will score
meaningfully higher at file and line granularity, at roughly 2–4× the inference latency.

**Expected outcome.** This configuration should push file-level acc@1 from the ~35–45% range
(current target) toward **50–60%**, approaching published frontier-model baselines on
SWE-bench fault localization — at a fraction of the per-prediction API cost.

## CI Integration

A GitHub Action triggers on test failure, captures the traceback, and runs hybrid retrieval + student inference **locally on the `ubuntu-latest` runner** — no external API calls, no model hosting. The QLoRA-finetuned 3B student (~2–3 GB) fits within the 7 GB runner RAM; CPU inference takes ~30–60 s per failure.

The pre-built embedding cache is stored as a GitHub Actions cache artifact and updated incrementally — only changed files are re-embedded on each push.

The action posts a PR comment with ranked fault predictions linking directly to the file:line range in the diff viewer:

```
Sift fault localization — 2 candidate locations

1. src/parser/tokenizer.py  L42–58   (confidence 0.81)
2. src/utils/string_ops.py  L107–112 (confidence 0.54)
```

See Phase 8 in [docs/roadmap.md](docs/roadmap.md) for implementation details.

## License

MIT
