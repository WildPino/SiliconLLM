"""Local-safe structural/receipt tests: no torch import, weights or training.

The tensor/causality/physical-bypass smoke test is `pilot.py --self-test` and
must run inside Kaggle before source acquisition, as enforced by the launcher.
"""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).with_name("pilot.py")
SPEC = importlib.util.spec_from_file_location("context_program_pilot", SOURCE)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class StructuralTests(unittest.TestCase):
    def test_import_has_no_model_or_tensor_side_effect(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf8"))
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = [alias.name for node in top_imports for alias in getattr(node, "names", [])]
        self.assertNotIn("torch", names)
        self.assertNotIn("transformers", names)
        self.assertNotIn("datasets", names)

    def test_replacement_physically_overrides_all_omitted_forwards(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                        node.name == "replace_layers")
        text = ast.get_source_segment(SOURCE.read_text(encoding="utf8"), function)
        self.assertIn("layers[index].forward = first", text)
        self.assertIn("layers[index].forward = original", text)
        self.assertNotIn("register_forward_hook", text)
        self.assertIn("finally:", text)
        self.assertIn('kwargs.get("use_cache", False)', text)

    def test_dev_decision_is_written_before_test_evaluation(self):
        text = SOURCE.read_text(encoding="utf8")
        main = text[text.index("def main():"):]
        self.assertLess(main.index('output / "dev_decision.json"'),
                        main.index('mapping_test = mapping_evaluation'))
        self.assertIn('gate["status"] == "EXPLORATORY_POSITIVE"', main)


class ReceiptTests(unittest.TestCase):
    def test_revision_and_unknown_settings_fail_closed(self):
        self.assertEqual(pilot.validate_config({})["horizons"], [16, 32, 64])
        with self.assertRaises(ValueError):
            pilot.validate_config({"model_revision": "main"})
        with self.assertRaises(ValueError):
            pilot.validate_config({"typo": 1})
        with self.assertRaises(ValueError):
            pilot.validate_config({"wall_seconds": 14401})

    def test_articles_group_subheadings_in_same_document(self):
        rows = [{"text": value} for value in (
            " = First article = \n", "paragraph one", " = = Subheading = = \n",
            "paragraph two", " = Second article = \n", "paragraph three")]
        docs = list(pilot.split_articles(rows))
        self.assertEqual(len(docs), 2)
        self.assertIn("paragraph two", docs[0])
        self.assertIn("Subheading", docs[0])
        self.assertNotIn("Second article", docs[0])

    def test_json_receipts_reject_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            pilot.write_json(path, {"finite": 1.0})
            self.assertTrue(path.exists())
            with self.assertRaises(ValueError):
                pilot.write_json(path, {"invalid": float("nan")})

    def gate_fixture(self):
        metrics = {"fixed": {"pooled_sse": 100.0},
                   "conditional": {"pooled_sse": 80.0},
                   "static": {"pooled_sse": 100.0},
                   "shuffled": {"pooled_sse": 95.0},
                   "conditional_gain_ci95": [0.1, 0.2, 0.3]}
        mapping = {"dev": {"16": metrics}}
        whole = {"dev": [{"arm": "fixed", "horizon": 32, "nll": 3.0},
                         {"arm": "conditional", "horizon": 16, "nll": 2.9}]}
        training = {arm: {"completed_step": 100, "stop": "fixed_steps"}
                    for arm in ("fixed", "conditional", "static")}
        return mapping, whole, training, {"steps_per_arm": 100}

    def test_incomplete_arms_cannot_pass(self):
        mapping, whole, training, cfg = self.gate_fixture()
        self.assertEqual(pilot.exploratory_gate(mapping, whole, training, cfg)["status"],
                         "EXPLORATORY_POSITIVE")
        training["static"]["completed_step"] = 99
        self.assertEqual(pilot.exploratory_gate(mapping, whole, training, cfg)["status"],
                         "INCONCLUSIVE")

    def test_shuffle_and_static_controls_are_required(self):
        mapping, whole, training, cfg = self.gate_fixture()
        mapping["dev"]["16"]["shuffled"]["pooled_sse"] = 81.0
        self.assertEqual(pilot.exploratory_gate(mapping, whole, training, cfg)["status"],
                         "EXPLORATORY_NEGATIVE")
        mapping["dev"]["16"]["shuffled"]["pooled_sse"] = 95.0
        mapping["dev"]["16"]["static"]["pooled_sse"] = 79.0
        self.assertEqual(pilot.exploratory_gate(mapping, whole, training, cfg)["status"],
                         "EXPLORATORY_NEGATIVE")

    def test_test_data_cannot_change_dev_gate(self):
        mapping, whole, training, cfg = self.gate_fixture()
        before = pilot.exploratory_gate(mapping, whole, training, cfg)
        mapping["test"] = {"16": "unreadable sentinel"}
        whole["test"] = "unreadable sentinel"
        self.assertEqual(before, pilot.exploratory_gate(mapping, whole, training, cfg))


if __name__ == "__main__":
    unittest.main()
