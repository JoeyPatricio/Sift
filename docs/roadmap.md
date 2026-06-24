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
  Phase 6 raises this recall ceiling with dense retrieval and recalibrates `bm25_top_k` for the hybrid setup.

## Phase 2 — Teacher distillation  → `teacher/distill.py` + `teacher/reject.py`

Training corpus generation. Estimated API cost: **~$27** (of the $60 total project budget).

- Teacher: `claude-haiku-4-5`, `effort=low`, `bm25_top_k=10` files in context.
  Batches API (50% discount) keeps cost predictable — estimate ~$25–40 for the full
  train split (~1,800 examples), leaving headroom for dev and retries.
- **Rejection-sample** (`reject.filter_traces`, file-level): keep only traces whose top
  prediction matches the gold diff. Expected acceptance rate: 25–40% with Haiku → target
  500–700 accepted traces from train.
- **Deliverable:** `data/traces.jsonl` (accepted traces); acceptance-rate and cost report.
- **Note:** teacher outputs a single `{ file, start_line, end_line }` prediction per trace here. Phase 7 extends this to ranked multi-prediction format via a supplemental distillation run — no need to change this phase's output.
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
- **Note:** the prompt format used here must match inference exactly (see `infer/predict.py`). Phase 7 updates this format for multi-prediction output and fine-tunes the student further on supplemental traces — do not change the format mid-phase.
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
- **Deliverable:** results table + short writeup (README results section). Phase 7 extends this table with confidence calibration (ECE) and per-rank accuracy once multi-prediction output is available.

## Phase 5 — Credibility: leakage controls + ablations

- Leakage: repo-held-out and/or date-based splits; report student performance restricted to
  instances after the base model's training cutoff; dedup near-duplicate fixes.
- Ablations: retrieval top-k sensitivity; with/without CoT; file vs. line granularity.
- Optional: Defects4J as a cross-language generalization probe.
- **Deliverable:** robustness section that pre-empts the "did it just memorize?" critique.

## Phase 6 — Hybrid retrieval  → `retrieval/dense.py`

Raise the recall ceiling above BM25's 89.9% by adding a dense retrieval path.

- Embed all candidate source files with `nomic-embed-code` running locally via Ollama ($0).
- At query time: embed the traceback + test name; retrieve top-k by cosine similarity.
- Merge BM25 and dense ranked lists via reciprocal rank fusion (RRF).
- **Deliverable:** updated recall@k curve; new baseline floor on dev with hybrid retrieval.
- **Cost:** $0 — embeddings run locally; one-time build of the file embedding cache.

## Phase 7 — Multi-fault predictions + confidence scores  → `schema.py` + `teacher/distill.py` + `eval/`

Extend the pipeline from single-prediction to ranked multi-prediction output.

- Update teacher prompt to elicit 2–3 ranked `{ file, start_line, end_line, confidence }` predictions per trace; update rejection filter to require the gold location in the top-3.
- Re-distill a supplemental batch (~200–400 accepted traces) in the new format using remaining budget; fine-tune student on the combined corpus.
- Update eval harness: acc@1/acc@3/MRR already track top-k; add confidence calibration (ECE) and report whether confidence correlates with correctness.
- **Deliverable:** updated results table with calibration metrics; student that surfaces 2–3 candidates ranked by confidence.
- **Cost:** ~$6–10 additional (increased output tokens for multi-location traces).

## Phase 8 — CI/GitHub Action integration  → `.github/workflows/sift.yml`

Ship Sift as a drop-in CI tool that comments fault predictions on failing PRs. All inference runs locally — no external API calls, no model hosting.

- GitHub Action triggers on test failure; captures test name + traceback from the runner log.
- Runs hybrid retrieval + student inference on the `ubuntu-latest` runner. The 3B student (~2–3 GB) fits within the 7 GB runner RAM; CPU inference takes ~30–60 s per failure.
- Pre-built embedding cache stored as a GitHub Actions cache artifact; updated incrementally on each push — only changed files are re-embedded.
- Posts a PR comment with ranked predictions and confidence scores, each linking to the file:line range in the diff viewer.
- **Deliverable:** working `sift.yml`; end-to-end demo on a real failing PR in this repo.
- **Cost:** $0 — GitHub Actions free tier covers compute; student runs locally.

---

## Critical path & dependencies

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
(data)      (floor)     (ceiling +  (train)     (headline)  (credibility)
                         corpus)
                                                    │
                                         ┌──────────┼──────────┐
                                         ▼          ▼          ▼
                                      Phase 6    Phase 7    Phase 8
                                     (hybrid    (multi-      (CI /
                                      retr.)    fault)      Action)
```

Phases 6–8 are independent of each other and all depend on Phase 4 completing. Phase 2's
teacher run is the long pole (API time/cost) and Phase 3 needs a GPU — use Kaggle
(30 free GPU hours/week) to keep that cost at $0. Everything through Phase 2 runs on a laptop.

## Cost budget

| Phase | Item | Estimated cost |
|---|---|---|
| 2 | Haiku distillation, train split (~1,800 ex), Batches API | ~$27 |
| 4 | Haiku teacher ceiling, test split (500 ex), Batches API | ~$8 |
| 3 | QLoRA training — Kaggle free tier | $0 |
| 6 | Dense embeddings — nomic-embed-code via Ollama (local) | $0 |
| 7 | Multi-fault re-distillation, ~200–400 supplemental traces | ~$6–10 |
| 8 | CI/GitHub Action — local inference, no API calls | $0 |
| — | Buffer (retries, dev distillation, validation runs) | ~$15 |
| | **Total** | **~$56–65** |
