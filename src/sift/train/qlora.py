"""QLoRA fine-tune the student on rejection-sampled teacher traces.

NOT YET IMPLEMENTED — requires the [train] extra (torch, transformers, peft, bitsandbytes,
trl) and a CUDA GPU. Kept out of the core install so the eval harness runs on a laptop.

Plan:
    1. Load the base student (settings.student_model) in 4-bit (bitsandbytes nf4).
    2. Attach LoRA adapters (peft) to the attention/MLP projections.
    3. Format each accepted Trace into a supervised example:
         input  = system + test + traceback + retrieved candidate files
         target = teacher reasoning + the structured prediction
    4. Train with trl's SFTTrainer; checkpoint to settings.runs_dir.

The training prompt MUST match the inference prompt (sift.infer.predict) exactly — any
drift between them silently degrades eval numbers.
"""

from __future__ import annotations

from sift.schema import Trace


def train(traces: list[Trace], output_dir: str) -> None:
    """Fine-tune the student on accepted traces and write the adapter to ``output_dir``."""
    raise NotImplementedError(
        "QLoRA training is not implemented yet. Install the training stack with "
        '`pip install -e ".[train]"` on a CUDA box, then implement the steps documented '
        "at the top of this module. Keep the training prompt in sync with sift.infer.predict."
    )
