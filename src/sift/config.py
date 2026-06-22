"""Central configuration. Reads from environment / .env with a SIFT_ prefix."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SIFT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Teacher (distillation) ---
    # Anthropic's most capable Opus-tier model: strong reasoning for generating
    # CoT traces, 1M context to hold the candidate files, and half the cost of
    # Fable 5 — which matters when generating thousands of traces.
    teacher_model: str = "claude-opus-4-8"
    teacher_effort: str = "high"  # low | medium | high | xhigh | max
    # The teacher API key is read by the anthropic SDK from ANTHROPIC_API_KEY
    # directly (no SIFT_ prefix) — see .env.example.

    # --- Student (the thing we ship) ---
    student_model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"

    # --- Retrieval ---
    bm25_top_k: int = 20  # candidate files passed to the model after retrieval

    # --- Paths ---
    data_dir: Path = Path("./data")
    runs_dir: Path = Path("./runs")


settings = Settings()
