from __future__ import annotations

import types
import unittest
from unittest import mock

from mathalign_dpo.training import model_loader


class ModelLoaderTests(unittest.TestCase):
    def test_mps_loader_does_not_create_bitsandbytes_config(self) -> None:
        fake_torch = _FakeTorch(backend="mps")
        fake_transformers = _FakeTransformers("mps")
        fake_transformers.BitsAndBytesConfig = mock.Mock(side_effect=AssertionError("unexpected bnb"))
        fake_peft = _FakePeft()

        with mock.patch.object(model_loader.importlib, "import_module", side_effect=_imports(fake_torch, fake_transformers, fake_peft)):
            loaded = model_loader.load_model_and_tokenizer(_config("mps"))

        self.assertEqual(loaded.metadata["backend"], "mps")
        fake_transformers.BitsAndBytesConfig.assert_not_called()
        self.assertEqual(loaded.model.to_device, "mps")

    def test_cuda_loader_creates_bitsandbytes_config_only_in_cuda_branch(self) -> None:
        fake_torch = _FakeTorch(backend="cuda")
        fake_transformers = _FakeTransformers("cuda")
        fake_peft = _FakePeft()

        with mock.patch.object(model_loader.importlib, "import_module", side_effect=_imports(fake_torch, fake_transformers, fake_peft)):
            loaded = model_loader.load_model_and_tokenizer(_config("cuda"))

        self.assertEqual(loaded.metadata["backend"], "cuda")
        self.assertEqual(fake_transformers.bitsandbytes_calls, 1)

    def test_base_loader_does_not_apply_lora(self) -> None:
        fake_torch = _FakeTorch(backend="mps")
        fake_transformers = _FakeTransformers("mps")
        fake_peft = _FakePeft()

        with mock.patch.object(model_loader.importlib, "import_module", side_effect=_imports(fake_torch, fake_transformers, fake_peft)):
            loaded = model_loader.load_base_model_and_tokenizer(_config("mps"))

        self.assertEqual(loaded.metadata["backend"], "mps")
        self.assertEqual(fake_peft.get_peft_model_calls, 0)
        self.assertNotIn("lora", loaded.metadata)

    def test_policy_loader_attaches_trainable_sft_adapter(self) -> None:
        fake_torch = _FakeTorch(backend="mps")
        fake_transformers = _FakeTransformers("mps")
        fake_peft = _FakePeft()

        with mock.patch.object(model_loader.importlib, "import_module", side_effect=_imports(fake_torch, fake_transformers, fake_peft)):
            loaded = model_loader.load_policy_model_from_sft_adapter(_config("mps"), "adapter")

        self.assertEqual(fake_peft.from_pretrained_calls, [("adapter", True)])
        self.assertEqual(loaded.metadata["adapter_initialization"], "stage3_sft")
        self.assertEqual(loaded.model.to_device, "mps")

    def test_zero_trainable_parameters_fail(self) -> None:
        fake_torch = _FakeTorch(backend="mps")
        fake_transformers = _FakeTransformers("mps", trainable=False)
        fake_peft = _FakePeft()

        with mock.patch.object(model_loader.importlib, "import_module", side_effect=_imports(fake_torch, fake_transformers, fake_peft)):
            with self.assertRaisesRegex(ValueError, "zero trainable"):
                model_loader.load_model_and_tokenizer(_config("mps"))

    def test_mps_fallback_environment_fails(self) -> None:
        fake_torch = _FakeTorch(backend="mps")
        fake_transformers = _FakeTransformers("mps")
        fake_peft = _FakePeft()

        with mock.patch.dict("os.environ", {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with mock.patch.object(model_loader.importlib, "import_module", side_effect=_imports(fake_torch, fake_transformers, fake_peft)):
                with self.assertRaisesRegex(RuntimeError, "must not silently fall back"):
                    model_loader.load_model_and_tokenizer(_config("mps"))


class _FakeParam:
    def __init__(self, requires_grad: bool, device_type: str):
        self.requires_grad = requires_grad
        self.device = types.SimpleNamespace(type=device_type)


class _FakeModel:
    def __init__(self, device_type: str, trainable: bool = True):
        self.config = types.SimpleNamespace(use_cache=True)
        self.to_device = device_type
        self.params = [_FakeParam(trainable, device_type)]

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def to(self, device):
        self.to_device = device
        for param in self.params:
            param.device = types.SimpleNamespace(type=device)
        return self

    def named_parameters(self):
        return [(f"param_{index}", param) for index, param in enumerate(self.params)]


class _FakeTransformers:
    def __init__(self, device_type: str, trainable: bool = True):
        self.bitsandbytes_calls = 0
        self.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
        self.AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: _FakeModel(device_type, trainable=trainable)
        )

    def BitsAndBytesConfig(self, **kwargs):
        self.bitsandbytes_calls += 1
        return {"bnb": kwargs}


class _FakePeft:
    def __init__(self):
        self.get_peft_model_calls = 0
        self.from_pretrained_calls = []

    def LoraConfig(self, **kwargs):
        return {"lora": kwargs}

    def get_peft_model(self, model, lora_config):
        self.get_peft_model_calls += 1
        return model

    def prepare_model_for_kbit_training(self, model, use_gradient_checkpointing):
        return model

    @property
    def PeftModel(self):
        parent = self

        class _PeftModel:
            @staticmethod
            def from_pretrained(model, model_id, is_trainable=False, **kwargs):
                parent.from_pretrained_calls.append((model_id, is_trainable))
                for param in model.params:
                    param.requires_grad = bool(is_trainable)
                return model

        return _PeftModel


class _FakeTorch:
    float16 = object()
    bfloat16 = object()
    float32 = object()

    def __init__(self, backend: str):
        self.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_built=lambda: backend == "mps", is_available=lambda: backend == "mps"))
        self.cuda = types.SimpleNamespace(is_available=lambda: backend == "cuda")


def _imports(fake_torch, fake_transformers, fake_peft):
    def import_module(name):
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        if name == "peft":
            return fake_peft
        raise ModuleNotFoundError(name)

    return import_module


def _config(backend: str) -> dict[str, object]:
    return {
        "model": {
            "name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "trust_remote_code": False,
            "torch_dtype": "float16" if backend == "mps" else "bfloat16",
            "use_cache": False,
            "gradient_checkpointing": True,
        },
        "quantization": {
            "enabled": backend == "cuda",
            "load_in_4bit": backend == "cuda",
            "quant_type": "nf4" if backend == "cuda" else None,
            "use_double_quant": backend == "cuda",
            "compute_dtype": "bfloat16",
        },
        "lora": {
            "enabled": True,
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj"],
        },
        "runtime": {"backend": backend},
    }
