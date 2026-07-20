"""Adapter checkpoint helpers shared by SFT and DPO training."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import TrainerCallback


class BestAdapterSaverCallback(TrainerCallback):
    """Save the best adapter according to evaluation loss."""

    def __init__(
        self,
        adapter_dir: Path,
        metric_name: str = "eval_loss",
        selected_adapters: Sequence[str] | None = None,
    ) -> None:
        self.adapter_dir = adapter_dir
        self.metric_name = metric_name
        self.selected_adapters = list(selected_adapters) if selected_adapters is not None else None
        self.best_metric: float | None = None
        self.best_step: int | None = None

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: Mapping[str, Any] | None = None,
        model: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if model is None or not metrics or self.metric_name not in metrics:
            return
        metric = float(metrics[self.metric_name])
        if not math.isfinite(metric):
            return
        if self.best_metric is not None and metric >= self.best_metric:
            return
        self.best_metric = metric
        self.best_step = int(getattr(state, "global_step", 0))
        save_adapter(model, self.adapter_dir, selected_adapters=self.selected_adapters)

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.adapter_dir),
            "metric": self.metric_name,
            "metric_name": self.metric_name,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "best_global_step": self.best_step,
            "saved": self.best_metric is not None,
        }


def ensure_best_adapter_saved(
    callback: BestAdapterSaverCallback,
    model: Any,
    eval_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Ensure a best adapter exists after the final evaluation."""

    if callback.best_metric is None and callback.metric_name in eval_metrics:
        metric = float(eval_metrics[callback.metric_name])
        if math.isfinite(metric):
            callback.best_metric = metric
            callback.best_step = None
            save_adapter(model, callback.adapter_dir, selected_adapters=callback.selected_adapters)
    return callback.metadata()


def save_adapter(model: Any, adapter_dir: Path, selected_adapters: Sequence[str] | None = None) -> None:
    """Save a PEFT adapter directory, replacing a previous copy if present."""

    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if selected_adapters is None:
        model.save_pretrained(adapter_dir)
        return
    try:
        model.save_pretrained(adapter_dir, selected_adapters=list(selected_adapters))
    except TypeError:
        model.save_pretrained(adapter_dir)
