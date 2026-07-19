"""Stage 3 SFT training command-line entrypoint."""

from __future__ import annotations

import argparse
import sys

from mathalign_dpo.training.train_sft import cli_payload, train_sft_from_config


def build_parser() -> argparse.ArgumentParser:
    """Build the SFT CLI parser."""

    parser = argparse.ArgumentParser(description="Train the Stage 3 Mini SFT LoRA adapter.")
    parser.add_argument("--config", required=True, help="Path to one YAML run config.")
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke caps from the config.")
    parser.add_argument("--output-dir", default=None, help="Override the SFT run output directory.")
    parser.add_argument("--train-samples", type=int, default=None, help="Override selected train rows for debugging.")
    parser.add_argument("--validation-samples", type=int, default=None, help="Override selected validation rows for debugging.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override SFT max steps for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing a non-empty output directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run Stage 3 SFT."""

    args = build_parser().parse_args(argv)
    result = train_sft_from_config(
        config_path=args.config,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        max_steps=args.max_steps,
        overwrite=args.overwrite,
    )
    print(cli_payload(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Stage 3 SFT failed: {exc}", file=sys.stderr)
        raise
