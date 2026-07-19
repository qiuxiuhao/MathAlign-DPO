from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_MAC_DEPENDENCIES = {"bitsandbytes"}
REQUIRED_DEPENDENCIES = {
    "accelerate",
    "datasets",
    "huggingface_hub",
    "peft",
    "psutil",
    "pyarrow",
    "pyyaml",
    "pytest",
    "safetensors",
    "torch",
    "transformers",
    "trl",
}


class PackagingTests(unittest.TestCase):
    def test_requirements_have_unique_bounded_dependencies(self) -> None:
        requirements = _read_requirements()
        names = [_dependency_name(requirement) for requirement in requirements]

        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(REQUIRED_DEPENDENCIES.issubset(set(names)))
        self.assertTrue(FORBIDDEN_MAC_DEPENDENCIES.isdisjoint(set(names)))
        for requirement in requirements:
            self.assertRegex(requirement, r"[<>=~!]=?")

    def test_pyproject_does_not_duplicate_runtime_dependencies(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = pyproject["project"]

        self.assertNotIn("dependencies", project)
        self.assertNotIn("optional-dependencies", project)
        self.assertEqual(project["requires-python"], ">=3.11,<3.12")
        package_finder = pyproject["tool"]["setuptools"]["packages"]["find"]
        self.assertIn("src", package_finder["where"])
        self.assertIn("mathalign_dpo*", package_finder["include"])

    def test_no_manual_sys_path_bootstrap_exists(self) -> None:
        forbidden_calls = [f"sys.path.{method}" for method in ("insert", "append")]
        for path in [*ROOT.glob("scripts/**/*.py"), *ROOT.glob("src/**/*.py")]:
            text = path.read_text(encoding="utf-8")
            for forbidden_call in forbidden_calls:
                self.assertNotIn(forbidden_call, text, str(path))

    def test_installed_package_imports_from_outside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-c", "import mathalign_dpo"],
                cwd=tmp,
                check=False,
                text=True,
                capture_output=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_prepare_data_help_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.prepare_data", "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mini-config", result.stdout)

    def test_build_stage2_data_help_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.build_stage2_data", "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--formal-config", result.stdout)

    def test_train_sft_help_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.train_sft", "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)

    def test_train_dpo_help_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.train_dpo", "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--sft-run-dir", result.stdout)


def _read_requirements() -> list[str]:
    lines = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def _dependency_name(requirement: str) -> str:
    match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
    if not match:
        raise ValueError(f"Invalid requirement: {requirement}")
    return match.group(1).lower().replace("-", "_")
