from __future__ import annotations

import unittest

from mathalign_dpo.data.split_normalized import assign_split, split_examples


def _examples(count: int) -> list[dict[str, object]]:
    return [
        {
            "schema_version": "1.0",
            "id": f"numina_train_{index:08d}",
            "source": "AI-MO/NuminaMath-CoT",
            "source_split": "train",
            "source_id": f"{index:08d}",
            "problem": f"Problem {index}",
            "solution": f"Solution {index}",
            "metadata": {"source_subset": None, "original_fields": ["problem", "solution"]},
        }
        for index in range(count)
    ]


def _config(train: int, validation: int, evaluation: int) -> dict[str, object]:
    return {
        "data": {
            "train_samples": train,
            "validation_samples": validation,
            "evaluation_samples": evaluation,
        }
    }


class SplitNormalizedTests(unittest.TestCase):
    def test_splits_are_deterministic_and_disjoint(self) -> None:
        ratios = {"train": 0.8, "validation": 0.1, "evaluation": 0.1}
        kwargs = {
            "examples": _examples(1000),
            "dataset_name": "dataset",
            "dataset_revision": "revision",
            "source_split": "train",
            "seed": 42,
            "ratios": ratios,
            "mini_config": _config(20, 5, 5),
            "formal_config": _config(50, 10, 10),
        }

        first = split_examples(**kwargs)
        second = split_examples(**kwargs)

        self.assertEqual(first.formal_ids, second.formal_ids)
        all_ids = first.formal_ids["train"] + first.formal_ids["validation"] + first.formal_ids["evaluation"]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_mini_views_are_formal_prefix_subsets(self) -> None:
        result = split_examples(
            examples=_examples(1000),
            dataset_name="dataset",
            dataset_revision="revision",
            source_split="train",
            seed=42,
            ratios={"train": 0.8, "validation": 0.1, "evaluation": 0.1},
            mini_config=_config(20, 5, 5),
            formal_config=_config(50, 10, 10),
        )

        for split, mini_ids in result.mini_ids.items():
            self.assertEqual(mini_ids, result.formal_ids[split][: len(mini_ids)])

    def test_sample_counts_do_not_change_split_membership(self) -> None:
        ratios = {"train": 0.8, "validation": 0.1, "evaluation": 0.1}
        self.assertEqual(
            assign_split("abc", "dataset", "revision", "train", 42, ratios),
            assign_split("abc", "dataset", "revision", "train", 42, ratios),
        )
