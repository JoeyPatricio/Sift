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
def eval(
    split: str = typer.Option("test", help="Which split to evaluate."),
    model: str = typer.Option(
        "baseline",
        help="'baseline' (retrieval-only) or a path to a trained student adapter.",
    ),
) -> None:
    """Evaluate a predictor: acc@1 / acc@3 / MRR at file and line granularity."""
    console.print(f"[yellow]eval[/] split={split} model={model}")
    console.print(
        "[red]Wired but no data yet[/] — implement `sift mine` to produce a {split} set, "
        "then this runs sift.eval.harness.run_eval over it."
    )


if __name__ == "__main__":
    app()
