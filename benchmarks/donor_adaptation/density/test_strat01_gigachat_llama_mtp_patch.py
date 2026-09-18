"""Synthetic checks for the local, pinned llama.cpp GigaChat MTP patch.

No model weights are opened.  The test imports only ``conversion.deepseek``
from the checked-out llama.cpp tree and exercises its name filter with fake
tensor loaders.  Set ``SILICONLLM_LLAMA_MTP_SOURCE`` to test another clean
checkout of the pinned upstream commit.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


PINNED_SOURCE = Path(
    os.environ.get(
        "SILICONLLM_LLAMA_MTP_SOURCE",
        r"C:\Users\giosa\AppData\Local\Temp\siliconllm-llama-mtp-5b335f4",
    )
)


@unittest.skipUnless((PINNED_SOURCE / "conversion" / "deepseek.py").is_file(), "pinned llama.cpp checkout absent")
class GigaChatLlamaMtpPatchFilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path[:0] = [str(PINNED_SOURCE), str(PINNED_SOURCE / "gguf-py")]
        from conversion.deepseek import DeepseekV3Model

        cls.model = DeepseekV3Model

    def setUp(self) -> None:
        self.saved = {
            "mtp_only": self.model.mtp_only,
            "no_mtp": self.model.no_mtp,
            "n_main": self.model._n_main_layers,
        }
        self.model._n_main_layers = 26

    def tearDown(self) -> None:
        self.model.mtp_only = self.saved["mtp_only"]
        self.model.no_mtp = self.saved["no_mtp"]
        self.model._n_main_layers = self.saved["n_main"]

    def _filter(self, name: str):
        return self.model.filter_tensors((name, lambda: None))

    def test_default_remains_base_only(self) -> None:
        self.model.mtp_only = False
        self.model.no_mtp = False
        self.assertIsNotNone(self._filter("model.layers.25.self_attn.q_proj.weight"))
        self.assertIsNone(self._filter("model.layers.26.self_attn.q_proj.weight"))
        self.assertIsNotNone(self._filter("model.norm.weight"))

    def test_no_mtp_remains_base_only(self) -> None:
        self.model.mtp_only = False
        self.model.no_mtp = True
        self.assertIsNotNone(self._filter("model.layers.25.self_attn.q_proj.weight"))
        self.assertIsNone(self._filter("model.layers.26.self_attn.q_proj.weight"))
        self.assertIsNotNone(self._filter("lm_head.weight"))

    def test_mtp_keeps_only_layer26_and_required_root_tensors(self) -> None:
        self.model.mtp_only = True
        self.model.no_mtp = False
        self.assertIsNotNone(self._filter("model.layers.26.self_attn.q_proj.weight"))
        self.assertIsNone(self._filter("model.layers.25.self_attn.q_proj.weight"))
        self.assertIsNone(self._filter("model.layers.26.embed_tokens.weight"))
        self.assertIsNone(self._filter("model.layers.26.shared_head.head.weight"))
        for root_name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            with self.subTest(root_name=root_name):
                self.assertIsNotNone(self._filter(root_name))

    def test_standalone_draft_loader_uses_draft_model_params_and_path(self) -> None:
        source = (PINNED_SOURCE / "common" / "speculative.cpp").read_text(encoding="utf-8")
        self.assertIn("common_params params_dft = common_base_params_to_speculative(params);", source)
        self.assertIn("auto mparams_dft = common_model_params_to_llama(params_dft);", source)
        self.assertIn("model_path = params_dft.model.path;", source)
        self.assertIn(
            "llama_model_load_from_file(model_path.c_str(), mparams_dft);",
            source,
        )
        self.assertNotIn(
            "llama_model_load_from_file(params.model.path.c_str(), mparams);",
            source,
        )
        self.assertIn("} else if (spec_mtp) {", source)


if __name__ == "__main__":
    unittest.main()
