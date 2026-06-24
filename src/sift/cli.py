"""Sift command-line interface.

Each subcommand wires together one stage of the pipeline. Stages that aren't implemented yet
print their intended contract rather than failing silently.
"""

from __future__ import annotations

import typer
from rich.console import Console

from sift import __version__

app = typer.Typer(
    add_completion=False,
    help="Fault localization for failing tests — predict the file and line responsible.",
)
console = Console()


@app.command()
def version() -> None:
    """Print the Sift version."""
    console.print(f"sift {__version__}")


@app.command()
def mine(
    split: str = typer.Option("all", help="Split to mine: train | dev | test | all"),
    out: str = typer.Option("", "--out", help="Output directory (default: config data_dir)."),
    max_instances: int | None = typer.Option(None, "--max", help="Cap per split for smoke tests."),
    repo_cache: str = typer.Option("", "--repo-cache", help="Persistent git clone dir."),
    skip_traceback: bool = typer.Option(False, "--skip-traceback", help="Skip traceback capture."),
) -> None:
    """Mine SWE-bench instances into fault-localization Examples.

    Writes {out}/{split}.jsonl for each requested split. Repos are cloned on demand;
    pass --repo-cache to reuse clones across runs. Use --max 10 --skip-traceback for
    a fast end-to-end smoke test before committing to the full dataset.
    """
    import statistics
    from pathlib import Path as _Path

    from sift.config import settings
    from sift.data.mining import VALID_SPLITS, mine_split

    out_dir = _Path(out) if out else settings.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = _Path(repo_cache) if repo_cache else None
    splits_to_run = sorted(VALID_SPLITS) if split == "all" else [split]

    for sp in splits_to_run:
        out_file = out_dir / f"{sp}.jsonl"
        console.rule(f"[bold]Mining split: {sp}[/]")
        console.print(f"Writing → [cyan]{out_file}[/]")

        n_total = 0
        n_tracebacks = 0
        candidate_counts: list[int] = []

        with out_file.open("w", encoding="utf-8") as fh:
            for example in mine_split(
                sp,
                repo_cache_dir=cache_dir,
                max_instances=max_instances,
                skip_traceback=skip_traceback,
            ):
                fh.write(example.model_dump_json() + "\n")
                n_total += 1
                if example.traceback:
                    n_tracebacks += 1
                candidate_counts.append(len(example.candidate_files))
                if n_total % 10 == 0:
                    console.print(f"  {n_total} examples …", end="\r")

        console.print()

        if n_total == 0:
            console.print(f"  [yellow]No examples written for split '{sp}'.[/]")
            continue

        tb_pct = 100 * n_tracebacks / n_total
        candidate_counts.sort()
        p50 = statistics.median(candidate_counts)
        p75 = candidate_counts[int(len(candidate_counts) * 0.75)]

        console.print(
            f"  [green]✓[/] {n_total} examples written\n"
            f"  tracebacks captured : {n_tracebacks}/{n_total} ({tb_pct:.0f}%)\n"
            f"  candidate files p50 / p75 : {p50:.0f} / {p75}\n"
        )


@app.command()
def distill(
    split: str = typer.Option("train", help="Which split to distill."),
    out: str = typer.Option("data/traces.jsonl", help="Where to write teacher traces."),
) -> None:
    """Generate rejection-sampled CoT traces from the frontier teacher."""
    console.print(f"[yellow]distill[/] split={split} out={out}")
    console.print(
        "[red]Batch distillation not implemented yet[/] — sift/teacher/distill.py has a "
        "working single-example path (distill_one) to validate the prompt first."
    )


@app.command()
def train(
    traces: str = typer.Option("data/traces.jsonl", help="Accepted teacher traces."),
    out: str = typer.Option("runs/student", help="Output dir for the QLoRA adapter."),
) -> None:
    """QLoRA fine-tune the student on accepted traces (requires the [train] extra + GPU)."""
    console.print(f"[yellow]train[/] traces={traces} out={out}")
    console.print("[red]Not implemented yet[/] — see sift/train/qlora.py.")


@app.command()
def retrieve(
    example: str = typer.Option(..., "--example", help="Instance ID to retrieve for."),
    split: str = typer.Option("dev", help="Which split to look up the example from."),
    top_k: int = typer.Option(20, "--top-k", help="Number of candidate files to return."),
    data_dir: str = typer.Option("", "--data-dir", help="Directory containing {split}.jsonl."),
) -> None:
    """Show BM25 top-k candidate files for a single example (useful for debugging retrieval)."""
    from pathlib import Path as _Path

    from sift.config import settings
    from sift.data.mining import load_examples
    from sift.retrieval import rank_candidates

    data_path = _Path(data_dir) if data_dir else settings.data_dir
    jsonl_file = data_path / f"{split}.jsonl"

    if not jsonl_file.exists():
        console.print(f"[red]Not found:[/] {jsonl_file} — run `sift mine --split {split}` first.")
        raise typer.Exit(1)

    examples = load_examples(jsonl_file)
    matches = [ex for ex in examples if ex.id == example]
    if not matches:
        console.print(f"[red]Instance ID '{example}' not found in {split} split.[/]")
        raise typer.Exit(1)

    ex = matches[0]
    console.print(f"[bold]{ex.id}[/]  repo={ex.repo}  commit={ex.commit[:8]}")
    console.print(f"test: {ex.test_name}")
    console.print(f"gold files: {sorted(ex.ground_truth_files)}\n")

    ranked = rank_candidates(ex.candidate_files, ex.test_name, ex.traceback, top_k)
    gold = ex.ground_truth_files
    console.print(f"[bold]BM25 top-{top_k}:[/]")
    for i, f in enumerate(ranked, start=1):
        hit = "[green]✓[/]" if f.path in gold else " "
        console.print(f"  {hit} {i:2d}. {f.path}")


@app.command()
def eval(
    split: str = typer.Option("dev", help="Which split to evaluate."),
    model: str = typer.Option("baseline", help="'baseline' or path to a trained student adapter."),
    data_dir: str = typer.Option("", "--data-dir", help="Directory containing {split}.jsonl."),
    ks: str = typer.Option(
        "1,5,10,15,20,30,50",
        "--ks",
        help="Comma-separated k values for the recall@k curve.",
    ),
) -> None:
    """Evaluate a predictor: BM25 recall@k curve + acc@1 / acc@3 / MRR at file and line.

    Always prints the recall@k curve first — this is the retrieval ceiling that caps
    every downstream model's file-level acc@k. Run on dev to calibrate bm25_top_k,
    then on test for the final headline numbers.
    """
    from pathlib import Path as _Path

    from sift.config import settings
    from sift.data.mining import load_examples
    from sift.eval.harness import run_eval
    from sift.infer.predict import RetrievalBaseline
    from sift.retrieval.recall import recall_curve, suggest_top_k

    data_path = _Path(data_dir) if data_dir else settings.data_dir
    jsonl_file = data_path / f"{split}.jsonl"

    if not jsonl_file.exists():
        console.print(
            f"[red]Not found:[/] {jsonl_file} — run `sift mine --split {split}` first."
        )
        raise typer.Exit(1)

    console.print(f"Loading [cyan]{jsonl_file}[/] …")
    examples = load_examples(jsonl_file)
    console.print(f"  {len(examples)} examples\n")

    if not examples:
        console.print("[yellow]No examples to evaluate.[/]")
        raise typer.Exit(0)

    # --- BM25 recall@k curve ---
    k_values = [int(k.strip()) for k in ks.split(",") if k.strip()]
    console.rule("[bold]BM25 Recall@k  (file-level, upper bound for all models)[/]")
    curve = recall_curve(examples, k_values)
    for k, r in sorted(curve.items()):
        bar = "█" * int(r * 40)
        hit_marker = "[green]≥90%[/]" if r >= 0.90 else "     "
        console.print(f"  k={k:3d}  {r:.3f}  {hit_marker}  {bar}")
    best_k = suggest_top_k(curve)
    console.print(f"\n  → Suggested [bold]bm25_top_k = {best_k}[/] (first k ≥ 90% recall)\n")

    # --- Predictor metrics ---
    console.rule(f"[bold]Predictor metrics[/]  split={split}  model={model}")
    if model == "baseline":
        predictor = RetrievalBaseline()
    else:
        console.print("[red]Non-baseline predictors not implemented yet.[/]")
        raise typer.Exit(1)

    results = run_eval(predictor, examples)
    for summary in results.values():
        console.print(f"  {summary}")


if __name__ == "__main__":
    app()
