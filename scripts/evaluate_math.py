"""Stage 5 unified evaluation command-line entrypoint."""

from __future__ import annotations

import argparse
import sys

from mathalign_dpo.evaluation.evaluate_math import cli_payload, evaluate_math_from_config


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 5 evaluation parser."""

    parser = argparse.ArgumentParser(description="Evaluate Base/SFT/DPO on the Stage 5 Mini math set.")
    parser.add_argument("--config", required=True, help="Path to one YAML run config.")
    parser.add_argument("--sft-run-dir", required=True, help="Completed Stage 3 SFT output directory.")
    parser.add_argument("--dpo-run-dir", required=True, help="Completed Stage 4 DPO output directory.")
    parser.add_argument("--smoke-test", action="store_true", help="Use smoke evaluation sample count.")
    parser.add_argument("--output-dir", default=None, help="Override the evaluation output directory.")
    parser.add_argument("--samples", type=int, default=None, help="Override evaluation sample count for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing a non-empty output directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run Stage 5 evaluation."""

    args = build_parser().parse_args(argv)
    result = evaluate_math_from_config(
        config_path=args.config,
        sft_run_dir=args.sft_run_dir,
        dpo_run_dir=args.dpo_run_dir,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
        samples=args.samples,
        overwrite=args.overwrite,
    )
    print(cli_payload(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Stage 5 evaluation failed: {exc}", file=sys.stderr)
        raise
