"""Pure-stdlib AST/receipt tests. Never import torch or perform model work."""
import ast
import importlib.util
from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("geometry.py")
SPEC = importlib.util.spec_from_file_location("predictive_geometry", SOURCE)
geometry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geometry)


class GeometryTests(unittest.TestCase):
    def test_import_is_inert_and_has_no_tensor_dependency(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf8"))
        names = [alias.name for item in tree.body if isinstance(item, (ast.Import, ast.ImportFrom))
                 for alias in item.names]
        self.assertNotIn("torch", names)
        self.assertNotIn("transformers", names)

    def test_single_position_hook_is_removed_in_finally(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf8"))
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                    node.name == "logits_with_delta")
        source = ast.get_source_segment(SOURCE.read_text(encoding="utf8"), node)
        self.assertIn("== position", source)
        self.assertIn("layer_end - 1", source)
        self.assertIn("finally:", source)
        self.assertIn("handle.remove()", source)
        self.assertIn("use_cache=False", source)
        self.assertNotIn("no_grad", source)
        self.assertNotIn(".detach()", source)

    def test_score_sampling_uses_full_distribution(self):
        source = SOURCE.read_text(encoding="utf8")
        self.assertIn("torch.multinomial(probabilities[0, q]", source)
        self.assertNotIn(".topk(", source)
        self.assertIn("retain_graph=not last", source)
        self.assertIn('for split in ("dev", "test"):', source)
        self.assertIn('geometry_{split}_decision.json', source)

    def test_decision_requires_all_independent_controls(self):
        projections = {"79": {
            "fisher": {"32": {"trace_fraction": 0.7}},
            "fisher_minus_random_rank32_ci95": [0.2, 0.3, 0.4],
            "fisher_minus_pca_rank32_ci95": [0.1, 0.2, 0.3]}}
        rows = [dict(query=79, relative_norm=0.02, direction="fisher_null", kl=0.1),
                dict(query=79, relative_norm=0.02, direction="fisher_retained", kl=1.0)]
        self.assertEqual(geometry._decision(projections, dict(status="COMPLETE", rows=rows), 63)["status"],
                         "SUPPORTED_EXPLORATORY")
        projections["79"]["fisher_minus_pca_rank32_ci95"][0] = -0.01
        self.assertEqual(geometry._decision(projections, dict(status="COMPLETE", rows=rows), 63)["status"],
                         "NOT_SUPPORTED")
        self.assertEqual(geometry._decision(projections, dict(status="RUNNING", rows=rows), 63)["status"],
                         "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
