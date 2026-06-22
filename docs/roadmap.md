# Roadmap

Dataset: **SWE-bench** (Python) — test=Verified, dev=Lite−Verified, train=full−Verified−Lite.
See [architecture.md](architecture.md).

The ordering is deliberate: **get real numbers as early as possible**, and let the teacher
baseline double as the distillation corpus generator. Each phase ends with a concrete,
inspectable artifact. Phases 0–2 need no GPU; only Phase 3 does.

---

## Phase 0 — Data foundation  → `data/mining.py`

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

## Phase 1 — Retrieval + harness on real data  → `retrieval/` (done) + `eval/` (done)

First real metrics, and calibration of the retrieval stage.

- Measure BM25 **file-level recall@k** (is the gold file in the top-k?) across k to pick
  `bm25_top_k` — this caps every downstream model's achievable acc, so it's the first thing
  to know.
- Run `RetrievalBaseline` through `eval.harness` → the **floor** the student must beat.
- **Deliverable:** recall@k curve; baseline acc@1/@3/MRR (file + line) on dev.

## Phase 2 — Prompted teacher baseline  → `teacher/distill.py` + `teacher/reject.py` (done)

The frontier ceiling and the training corpus, from one pass.

- Run `claude-opus-4-8` over train+dev via `distill_batch` (Batches API, 50% cost). Produces
  per example: summarized CoT + structured prediction.
- **Teacher metrics on test** = the ceiling we distill toward (and a strong portfolio data
  point on its own: "frontier model acc@1 on SWE-bench fault localization").
- **Rejection-sample** (`reject.filter_traces`, file-level first): keep only traces whose top
  prediction matches the gold diff. Report acceptance rate.
- **Deliverable:** teacher metrics; `data/traces.jsonl` (accepted traces); acceptance-rate
  and cost report.
- **Risk:** teacher API cost — estimate on a subset first; batch to halve it.

## Phase 3 — QLoRA fine-tune the student  → `train/qlora.py`  *(needs GPU)*

- Format accepted traces into SFT examples — the prompt **must** match inference
  (`infer/predict.py`) exactly.
- Load Qwen2.5-Coder-3B in 4-bit (nf4), attach LoRA adapters, train with `trl`'s SFTTrainer.
- **Deliverable:** trained adapter in `runs/`; training/val loss curves.
- **Risk:** no local GPU — rent (RunPod/Lambda). Keep the `[train]` extra isolated so only
  this phase needs CUDA.

## Phase 4 — Student eval + the headline comparison  → `infer/StudentPredictor`

The portfolio artifact.

- Implement `StudentPredictor` (mirror the teacher prompt/parse), run it through the **same**
  harness on test.
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

Phase 2's teacher run is the long pole (API time/cost) and Phase 3 needs rented GPU — both
worth starting their infra early. Everything through Phase 2 runs on a laptop.
