"""Stage 2 data construction command-line entrypoint."""

from __future__ import annotations

import argparse
import json

from mathalign_dpo.data.stage2_pipeline import build_stage2_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 2 step, SFT, and DPO data.")
    parser.add_argument("--mini-config", required=True, help="Path to the Mini YAML config.")
    parser.add_argument("--formal-config", required=True, help="Path to the formal YAML config.")
    parser.add_argument("--smoke-test", action="store_true", help="Use deterministic smoke-test caps from config.")
    parser.add_argument("--output-dir", default=None, help="Override Stage 2 output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing Stage 2 outputs.")
    args = parser.parse_args()

    result = build_stage2_data(
        mini_config=args.mini_config,
        formal_config=args.formal_config,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
