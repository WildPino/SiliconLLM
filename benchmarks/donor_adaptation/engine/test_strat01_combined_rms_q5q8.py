from __future__ import annotations

import unittest

from benchmarks.donor_adaptation.engine import run_strat01_combined_rms_q5q8 as runner


class CombinedRMSQ5Q8Tests(unittest.TestCase):
    def test_protocol_and_source_contract(self) -> None:
        protocol = runner.PROTOCOL.read_text(encoding="utf-8")
        header = runner.HEADER.read_text(encoding="utf-8")
        engine = runner.ENGINE.read_text(encoding="utf-8")
        self.assertIn("COMBINED_RMS_Q5Q8_CLOSES_PROJECTION_GATES", protocol)
        self.assertIn(runner.SOURCE_VOID_ADJUDICATION_SHA, protocol)
        self.assertIn(runner.SOURCE_VOID_MANIFEST_SHA, protocol)
        self.assertIn(runner.KB_Q8_SHA, header)
        self.assertIn(runner.KB_OUT_SHA, header)
        self.assertIn("--strat01-combined-rms-q5q8", engine)

    def test_prior_bindings_and_controls(self) -> None:
        prior = runner.validate_prior_bindings()
        self.assertEqual(prior["status"], "Q5_0_Q8_0_SUFFICIENT_FOR_KB_BOUNDARY")
        self.assertTrue(all(prior["adjudication"]["controls"].values()))

    def test_source_coordinate_counts(self) -> None:
        combined = runner.HEADER.read_text(encoding="utf-8")
        upstream = runner.upstream.HEADER.read_text(encoding="utf-8")
        self.assertEqual(combined.count("strat01_r2b_rmsnorm_double_sum("), 2)
        self.assertEqual(combined.count("strat01_combined_kb_range("), 2)
        self.assertIn("strat01_r2b_rmsnorm_double_sum(input->ffn_inp", upstream)

    def test_source_inventory(self) -> None:
        inventory = runner.source_inventory()
        self.assertEqual(set(inventory), {"runner", "engine", "rung2a_header", "rung2b_header", "upstream_header", "kb_header", "diagnostic_header", "tests", "protocol"})
        self.assertTrue(all(len(item["sha256"]) == 64 for item in inventory.values()))

    def test_archived_upstream_sources_are_self_consistent(self) -> None:
        sources = runner.archived_upstream_sources()
        self.assertEqual(set(sources), {"engine", "diagnostic_header"})
        self.assertNotEqual(sources["engine"]["sha256"], runner.source_inventory()["engine"]["sha256"])

    def test_preserved_void_capture_binding(self) -> None:
        record, manifest = runner.validate_void_capture(runner.DEFAULT_OUTPUT)
        self.assertEqual(record["donor_graph_executions"], 1)
        self.assertEqual(record["provenance"]["commands"]["candidate"]["returncode"], 0)
        self.assertEqual(manifest["status"], "VOID_COMBINED_RMS_Q5Q8_PROPAGATION")


if __name__ == "__main__": unittest.main()
