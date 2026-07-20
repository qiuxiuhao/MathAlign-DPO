"""Preference accuracy helpers for Stage 5 evaluation."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from mathalign_dpo.evaluation.eval_data import _read_jsonl


def load_preference_validation_rows(
    config: Mapping[str, Any],
    dpo_metadata: Mapping[str, Any],
    sample_count: int,
) -> list[dict[str, Any]]:
    """Load the exact DPO validation rows selected by a Mini-only Stage 4 run."""

    selected_ids = [str(row_id) for row_id in dpo_metadata.get("token_statistics", {}).get("validation", {}).get("selected_ids", [])]
    if not selected_ids:
        raise ValueError("DPO metadata is missing validation selected_ids for preference evaluation")
    selected_ids = selected_ids[:sample_count]
    rows = _read_jsonl(Path(str(config["data"]["dpo_validation_file"])))
    by_id = {str(row["id"]): row for row in rows}
    missing = [row_id for row_id in selected_ids if row_id not in by_id]
    if missing:
        raise ValueError(f"Selected DPO validation rows are missing: {missing[:3]}")
    return [by_id[row_id] for row_id in selected_ids]


def preference_summary(rows: Sequence[Mapping[str, Any]], model_stage: str) -> dict[str, Any]:
    """Summarize chosen-vs-rejected log-prob preference rows."""

    stage_rows = [row for row in rows if row.get("model_stage") == model_stage]
    if not stage_rows:
        raise ValueError(f"No preference rows available for {model_stage}")
    correct = [row for row in stage_rows if bool(row.get("preferred_chosen"))]
    margins = [float(row["logp_margin"]) for row in stage_rows]
    return {
        "model_stage": model_stage,
        "num_examples": len(stage_rows),
        "preference_accuracy": len(correct) / len(stage_rows),
        "preference_margin_mean": sum(margins) / len(margins),
    }


def score_preference_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    model_stage: str,
    device: str,
) -> list[dict[str, Any]]:
    """Compute average completion log-prob for chosen and rejected responses."""

    output: list[dict[str, Any]] = []
    for row in rows:
        chosen = average_completion_logp(model, tokenizer, list(row["prompt"]), list(row["chosen"]), device)
        rejected = average_completion_logp(model, tokenizer, list(row["prompt"]), list(row["rejected"]), device)
        output.append(
            {
                "schema_version": "1.0",
                "id": str(row["id"]),
                "source_id": str(row["source_id"]),
                "model_stage": model_stage,
                "chosen_avg_logp": chosen,
                "rejected_avg_logp": rejected,
                "logp_margin": chosen - rejected,
                "preferred_chosen": chosen > rejected,
            }
        )
    return output


def average_completion_logp(model: Any, tokenizer: Any, prompt: list[Mapping[str, str]], completion: list[Mapping[str, str]], device: str) -> float:
    """Return average token log-prob for completion tokens in a chat example."""

    torch = importlib.import_module("torch")
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
    full_ids = tokenizer.apply_chat_template(prompt + completion, tokenize=True, return_tensors="pt")
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    prompt_len = len(prompt_ids)
    input_ids = full_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    token_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    completion_logps = token_logps[:, max(prompt_len - 1, 0) :]
    if completion_logps.numel() == 0:
        raise ValueError("Preference completion has no scored tokens")
    return float(completion_logps.mean().detach().cpu().item())
