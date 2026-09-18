"""Synthetic, offline tests for the read-only STRAT-01 GGUF contract."""

from __future__ import annotations

import copy
import unittest

from benchmarks.donor_adaptation.density import strat01_gigachat_mtp_gguf_contract as contract


def _manifest() -> dict[str, object]:
    return {
        "schema": "strat01_gigachat_mtp_sidecar_manifest_v1",
        "sidecar": {
            "name": contract.SIDECAR_NAME,
            "bytes": contract.SIDECAR_BYTES,
            "sha256": contract.SIDECAR_SHA256,
            "self_read_parity": "PASS_names_shapes_dtypes_offsets",
        },
        "source": {
            "repository": contract.REPOSITORY,
            "revision": contract.REVISION,
            "shard": contract.SHARD,
            "lfs_sha256_expected": contract.SOURCE_LFS_SHA256,
        },
        "tensors": [
            {"name": name, "dtype": spec.dtype, "shape": list(spec.shape)}
            for name, spec in sorted(contract.EXPECTED_SPECS.items())
        ],
    }


def _record(value: dict[str, object], name: str) -> dict[str, object]:
    for record in value["tensors"]:  # type: ignore[index]
        if record["name"] == name:  # type: ignore[index]
            return record  # type: ignore[return-value]
    raise AssertionError(name)


class GigaChatMtpGgufContractTest(unittest.TestCase):
    def test_complete_synthetic_manifest_passes_and_never_claims_execution(self) -> None:
        report = contract.preflight_manifest(_manifest())
        self.assertEqual(report["metadata_contract"], "PASS")
        self.assertEqual(report["manifest_tensor_count"], 210)
        self.assertEqual(report["expected_tensor_count"], 210)
        self.assertIn("NO_GGUF_WRITTEN.", report["limitations"])
        self.assertIn("NO_LOGITS_OR_NUMERICAL_FIDELITY_CLAIM.", report["limitations"])

    def test_missing_and_extra_names_fail_closed(self) -> None:
        value = _manifest()
        tensors = value["tensors"]  # type: ignore[index]
        tensors.pop()  # type: ignore[union-attr]
        tensors.append({"name": "model.layers.26.unrecognized.weight", "dtype": "BF16", "shape": [1]})  # type: ignore[union-attr]
        report = contract.preflight_manifest(value)
        self.assertEqual(report["metadata_contract"], "FAIL")
        joined = "\n".join(report["errors"])
        self.assertIn("missing exact expected names", joined)
        self.assertIn("unexpected names", joined)

    def test_wrong_shape_dtype_and_revision_fail_closed(self) -> None:
        value = _manifest()
        value["source"]["revision"] = "0" * 40  # type: ignore[index]
        q = _record(value, "model.layers.26.self_attn.q_proj.weight")
        q["shape"] = [1, 1]
        q["dtype"] = "F32"
        report = contract.preflight_manifest(value)
        joined = "\n".join(report["errors"])
        self.assertEqual(report["metadata_contract"], "FAIL")
        self.assertIn("source.revision", joined)
        self.assertIn("q_proj.weight.dtype", joined)
        self.assertIn("q_proj.weight.shape", joined)

    def test_every_expert_projection_is_required(self) -> None:
        value = _manifest()
        removed = "model.layers.26.mlp.experts.63.down_proj.weight"
        value["tensors"] = [record for record in value["tensors"] if record["name"] != removed]  # type: ignore[index]
        report = contract.preflight_manifest(value)
        joined = "\n".join(report["errors"])
        self.assertEqual(report["metadata_contract"], "FAIL")
        self.assertIn(removed, joined)

    def test_kv_split_and_expert_stack_plan_match_pinned_converter(self) -> None:
        report = contract.preflight_manifest(_manifest())
        operations = report["plan"]["operations"]
        targets = {item["target"]: item for item in operations}
        self.assertEqual(targets["blk.26.attn_k_b.weight"]["ggml_logical_shape"], [128, 512, 32])
        self.assertEqual(targets["blk.26.attn_v_b.weight"]["ggml_logical_shape"], [512, 192, 32])
        self.assertEqual(
            targets["blk.26.attn_k_b.weight"]["transform"],
            "view_source_as_[32,320,512]; slice_axis1_[0:128]; transpose_axes_1_2",
        )
        self.assertEqual(len(targets["blk.26.ffn_gate_exps.weight"]["source"]), 64)
        self.assertEqual(targets["blk.26.ffn_down_exps.weight"]["ggml_logical_shape"], [1280, 1536, 64])

    def test_root_requirements_and_runtime_blocker_are_explicit(self) -> None:
        report = contract.preflight_manifest(_manifest())
        roots = {item["target"]: item for item in report["plan"]["required_external_root_tensors"]}
        self.assertEqual(roots["output_norm.weight"]["required_bf16_bytes"], 3072)
        self.assertEqual(roots["output_norm.weight"]["required_bf16_sha256"], contract.ROOT_NORM_BF16_SHA256)
        self.assertEqual(roots["token_embd.weight"]["status"], "REQUIRED_PHYSICAL_TENSOR_IN_STANDALONE_DRAFT_GGUF")
        self.assertEqual(roots["token_embd.weight"]["alias_status"], "PROVEN_BYTE_IDENTICAL; MTP_BLOCK_COPY_OPTIONAL_IF_ROOT_IS_MATERIALIZED")
        self.assertEqual(report["runtime_blockers"][0]["status"], "BLOCKED_IN_PINNED_LLAMA_CPP")

    def test_sidecar_hash_claim_is_validated_without_reading_any_sidecar(self) -> None:
        value = copy.deepcopy(_manifest())
        value["sidecar"]["sha256"] = "f" * 64  # type: ignore[index]
        report = contract.preflight_manifest(value)
        self.assertEqual(report["metadata_contract"], "FAIL")
        self.assertIn("sidecar.sha256", "\n".join(report["errors"]))


if __name__ == "__main__":
    unittest.main()
