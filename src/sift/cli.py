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
    source: str = typer.Option("bugsinpy", help="Source dataset: bugsinpy | swebench-lite | ..."),
    out: str = typer.Option("data/raw", help="Output directory for mined examples."),
) -> None:
    """Mine bug-fix commits into fault-localization examples."""
    console.print(f"[yellow]mine[/] source={source} out={out}")
    console.print("[red]Not implemented yet[/] — see sift/data/mining.py for the contract.")


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
