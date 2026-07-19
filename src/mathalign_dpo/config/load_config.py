"""Load and validate the two approved Stage 1 configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "evaluation")


@dataclass(frozen=True)
class Stage1Configs:
    """Validated Mini/formal configuration pair."""

    mini_path: Path
    formal_path: Path
    mini: dict[str, Any]
    formal: dict[str, Any]

    @property
    def dataset_name(self) -> str:
        return str(self.formal["data"]["dataset_name"])

    @property
    def dataset_revision(self) -> str:
        return str(self.formal["data"]["dataset_revision"])

    @property
    def source_split(self) -> str:
        return str(self.formal["data"]["source_split"])

    @property
    def seed(self) -> int:
        return int(self.formal["project"]["seed"])

    @property
    def processed_dir(self) -> Path:
        return Path(str(self.formal["data"]["processed_dir"]))


def load_stage1_configs(mini_config: str | Path, formal_config: str | Path) -> Stage1Configs:
    """Load and validate Mini and formal configs for shared Stage 1 preparation."""

    mini_path = Path(mini_config)
    formal_path = Path(formal_config)
    mini = _load_yaml(mini_path)
    formal = _load_yaml(formal_path)
    _validate_single_config(mini, mini_path, expected_mode="mini")
    _validate_single_config(formal, formal_path, expected_mode="formal")
    _validate_shared_data_config(mini, formal)
    return Stage1Configs(mini_path=mini_path, formal_path=formal_path, mini=mini, formal=formal)


def split_ratios(config: dict[str, Any]) -> dict[str, float]:
    """Return configured split ratios keyed by contract split name."""

    data = config["data"]
    return {
        "train": float(data["train_ratio"]),
        "validation": float(data["validation_ratio"]),
        "evaluation": float(data["evaluation_ratio"]),
    }


def sample_counts(config: dict[str, Any]) -> dict[str, int]:
    """Return configured sample counts keyed by contract split name."""

    data = config["data"]
    return {
        "train": int(data["train_samples"]),
        "validation": int(data["validation_samples"]),
        "evaluation": int(data["evaluation_samples"]),
    }


def output_paths(config: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Path]:
    """Return Stage 1 final output paths, optionally rooted at a debug output dir."""

    data = config["data"]
    if output_dir is None:
        return {
            "train": Path(str(data["normalized_train_file"])),
            "validation": Path(str(data["normalized_validation_file"])),
            "evaluation": Path(str(data["normalized_eval_file"])),
            "statistics": Path(str(data["statistics_file"])),
            "manifest": Path(str(data["split_manifest_file"])),
        }
    root = Path(output_dir)
    return {
        "train": root / "normalized_train.jsonl",
        "validation": root / "normalized_validation.jsonl",
        "evaluation": root / "normalized_eval.jsonl",
        "statistics": root / "data_statistics.json",
        "manifest": root / "split_manifest.json",
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
    except ImportError:
        loaded = _simple_yaml_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse the limited YAML subset used by the two checked-in configs."""

    lines = _strip_yaml_comments(text.splitlines())
    parsed, index = _parse_mapping(lines, 0, 0)
    if index != len(lines):
        raise ValueError("Unsupported YAML structure in config")
    return parsed


def _strip_yaml_comments(raw_lines: list[str]) -> list[str]:
    lines: list[str] = []
    for raw_line in raw_lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        lines.append(raw_line.rstrip())
    return lines


def _parse_mapping(lines: list[str], start: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    index = start
    while index < len(lines):
        line = lines[index]
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation in YAML line: {line}")
        stripped = line.strip()
        if stripped.startswith("- "):
            break
        key, separator, remainder = stripped.partition(":")
        if not separator:
            raise ValueError(f"Expected YAML mapping entry: {line}")
        remainder = remainder.strip()
        if remainder in {">-", "|-"}:
            value, index = _parse_block_scalar(lines, index + 1, indent + 2)
        elif remainder:
            value = _parse_scalar(remainder)
            index += 1
        else:
            value, index = _parse_child(lines, index + 1, indent + 2)
        result[key] = value
    return result, index


def _parse_child(lines: list[str], start: int, indent: int) -> tuple[Any, int]:
    if start >= len(lines):
        return {}, start
    stripped = lines[start].strip()
    if stripped.startswith("- "):
        return _parse_list(lines, start, indent)
    return _parse_mapping(lines, start, indent)


def _parse_list(lines: list[str], start: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    index = start
    while index < len(lines):
        line = lines[index]
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if current_indent != indent:
            raise ValueError(f"Unexpected list indentation in YAML line: {line}")
        stripped = line.strip()
        if not stripped.startswith("- "):
            break
        result.append(_parse_scalar(stripped[2:].strip()))
        index += 1
    return result, index


def _parse_block_scalar(lines: list[str], start: int, indent: int) -> tuple[str, int]:
    chunks: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        chunks.append(line[indent:].strip())
        index += 1
    return " ".join(chunk for chunk in chunks if chunk), index


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." not in value:
            return int(value)
        return float(value)
    except ValueError:
        return value


def _validate_single_config(config: dict[str, Any], path: Path, expected_mode: str) -> None:
    required_top_keys = {
        "schema_version",
        "project",
        "model",
        "quantization",
        "lora",
        "data",
        "preprocessing",
        "negative_sampling",
        "sft",
        "dpo",
        "evaluation",
        "runtime",
        "output",
        "smoke_test",
    }
    missing = sorted(required_top_keys - set(config))
    if missing:
        raise ValueError(f"{path}: missing top-level config keys: {missing}")

    project = config["project"]
    if "stage" in project:
        raise ValueError(f"{path}: project.stage must be removed; stage belongs in docs/reports")
    if project.get("run_mode") != expected_mode:
        raise ValueError(f"{path}: project.run_mode must be {expected_mode!r}")
    if not isinstance(project.get("seed"), int):
        raise ValueError(f"{path}: project.seed must be an integer")

    data = config["data"]
    if not data.get("dataset_revision"):
        raise ValueError(f"{path}: data.dataset_revision must be pinned to a commit hash")
    ratios = split_ratios(config)
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"{path}: split ratios must sum to 1.0, got {ratios}")
    for split, count in sample_counts(config).items():
        if count <= 0:
            raise ValueError(f"{path}: data.{split}_samples must be positive")

    backend = config["runtime"].get("backend")
    quantization = config["quantization"]
    optimizer = config["sft"].get("optimizer")
    if backend == "mps":
        if quantization.get("enabled") or quantization.get("load_in_4bit"):
            raise ValueError(f"{path}: MPS config must not enable BitsAndBytes or 4-bit loading")
        if optimizer != "adamw_torch":
            raise ValueError(f"{path}: MPS SFT optimizer must be adamw_torch")
    elif backend == "cuda":
        if not quantization.get("enabled") or not quantization.get("load_in_4bit"):
            raise ValueError(f"{path}: CUDA config must enable 4-bit quantization")
        if quantization.get("quant_type") != "nf4":
            raise ValueError(f"{path}: CUDA config must use quantization.quant_type = nf4")
    else:
        raise ValueError(f"{path}: runtime.backend must be mps or cuda")


def _validate_shared_data_config(mini: dict[str, Any], formal: dict[str, Any]) -> None:
    shared_data_keys = [
        "dataset_name",
        "dataset_revision",
        "source_split",
        "train_ratio",
        "validation_ratio",
        "evaluation_ratio",
        "processed_dir",
        "normalized_train_file",
        "normalized_validation_file",
        "normalized_eval_file",
        "statistics_file",
        "split_manifest_file",
    ]
    for key in shared_data_keys:
        if mini["data"].get(key) != formal["data"].get(key):
            raise ValueError(f"Mini and formal configs must share data.{key}")
    if mini["project"].get("seed") != formal["project"].get("seed"):
        raise ValueError("Mini and formal configs must share project.seed")

    mini_counts = sample_counts(mini)
    formal_counts = sample_counts(formal)
    for split in SPLITS:
        if mini_counts[split] > formal_counts[split]:
            raise ValueError(f"Mini {split} count cannot exceed formal {split} count")
