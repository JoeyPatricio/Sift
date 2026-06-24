# Roadmap

Dataset: **SWE-bench** (Python) — test=Verified, dev=Lite−Verified, train=full−Verified−Lite.
See [architecture.md](architecture.md).

The ordering is deliberate: **get real numbers as early as possible**, and let the teacher
baseline double as the distillation corpus generator. Each phase ends with a concrete,
inspectable artifact. Phases 0–2 need no GPU; only Phase 3 does.

---

## Phase 0 — Data foundation  → `data/mining.py`  ✓

Turn SWE-bench instances into `Example`s.

- Load splits from HuggingFace (`princeton-nlp/SWE-bench`, `SWE-bench_Lite`,
  `SWE-bench_Verified`); build the train/dev/test partition above.
- For each instance: check out `base_commit` (pre-fix), collect candidate source files
  (repo tree, excluding tests/vendored/generated), and parse the gold patch into
  ground-truth `FaultLocation`s (reuse `data/ground_truth.py`).
- **Capture the failing traceback** by running each `FAIL_TO_PASS` test against the pre-fix
  state in the SWE-bench Docker harness. This is the heavy piece — start with Lite (300) +
  a few hundred train instances to validate end-to-end before scaling.
- **Deliverable:** `data/{train,dev,test}.jsonl`; a report of instance counts, how many
  tracebacks captured cleanly, and candidate-file-count distribution.
- **Risk:** Docker harness throughput / disk. Mitigate by subsetting first; cache aggressively.

## Phase 1 — Retrieval + harness on real data  → `retrieval/` + `eval/`  ✓

First real metrics, and calibration of the retrieval stage.

- Measure BM25 **file-level recall@k** (is the gold file in the top-k?) across k to pick
  `bm25_top_k` — this caps every downstream model's achievable acc, so it's the first thing
  to know.
- Run `RetrievalBaseline` through `eval.harness` → the **floor** the student must beat.
- **Deliverable:** recall@k curve; baseline acc@1/@3/MRR (file + line) on dev.
  *Results (dev, 207 examples):* BM25 recall@50 = 89.9%; suggested `bm25_top_k = 50`.
  Baseline floor: acc@1 = 0.217, acc@3 = 0.449, MRR = 0.364.

## Phase 2 — Teacher distillation  → `teacher/distill.py` + `teacher/reject.py`

Training corpus generation. Estimated API cost: **~$27** (of the $60 total project budget).

- Teacher: `claude-haiku-4-5`, `effort=low`, `bm25_top_k=10` files in context.
  Batches API (50% discount) keeps cost predictable — estimate ~$25–40 for the full
  train split (~1,800 examples), leaving headroom for dev and retries.
- **Rejection-sample** (`reject.filter_traces`, file-level): keep only traces whose top
  prediction matches the gold diff. Expected acceptance rate: 25–40% with Haiku → target
  500–700 accepted traces from train.
- **Deliverable:** `data/traces.jsonl` (accepted traces); acceptance-rate and cost report.
- **Risk:** acceptance rate too low to train effectively. Mitigate by validating on 20 dev
  examples with `distill_one` before committing the full batch.

> **Higher-budget path:** swap teacher to `claude-opus-4-8` with `effort=high` (~$400–600
> for the same scale). Acceptance rate roughly doubles and trace quality improves
> substantially — see README [Scaling Up] section.

## Phase 3 — QLoRA fine-tune the student  → `train/qlora.py`  *(needs GPU)*

- Format accepted traces into SFT examples — the prompt **must** match inference
  (`infer/predict.py`) exactly.
- Load Qwen2.5-Coder-3B in 4-bit (nf4), attach LoRA adapters, train with `trl`'s SFTTrainer.
- **Deliverable:** trained adapter in `runs/`; training/val loss curves.
- **Risk:** no local GPU — rent (RunPod/Lambda). Keep the `[train]` extra isolated so only
  this phase needs CUDA.

## Phase 4 — Student eval + the headline comparison  → `infer/StudentPredictor`

The portfolio artifact. Estimated API cost: **~$8** (teacher ceiling run on test split).

- Implement `StudentPredictor` (mirror the teacher prompt/parse), run it through the **same**
  harness on test.
- Run teacher on test split (500 examples, Haiku + Batches) to establish the ceiling.
- Produce the money table: **retrieval baseline vs. student vs. teacher** — acc@1/@3/MRR
  (file + line) **and cost + latency per prediction**. The claim is: student approaches
  teacher acc@1 at a fraction of cost/latency.
- **Deliverable:** results table + short writeup (README results section).

## Phase 5 — Credibility: leakage controls + ablations

- Leakage: repo-held-out and/or date-based splits; report student performance restricted to
  instances after the base model's training cutoff; dedup near-duplicate fixes.
- Ablations: retrieval top-k sensitivity; with/without CoT; file vs. line granularity.
- Optional: Defects4J as a cross-language generalization probe.
- **Deliverable:** robustness section that pre-empts the "did it just memorize?" critique.

---

## Critical path & dependencies

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
(data)      (floor)     (ceiling +  (train)     (headline)  (credibility)
                         corpus)
```

Phase 2's teacher run is the long pole (API time/cost) and Phase 3 needs a GPU — use Kaggle
(30 free GPU hours/week) to keep that cost at $0. Everything through Phase 2 runs on a laptop.

## Cost budget

| Phase | Item | Estimated cost |
|---|---|---|
| 2 | Haiku distillation, train split (~1,800 ex), Batches API | ~$27 |
| 4 | Haiku teacher ceiling, test split (500 ex), Batches API | ~$8 |
| 3 | QLoRA training — Kaggle free tier | $0 |
| — | Buffer (retries, dev distillation, validation runs) | ~$15 |
| | **Total** | **~$50 / $60 cap** |
