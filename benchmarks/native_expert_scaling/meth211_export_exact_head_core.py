#!/usr/bin/env python3
"""Materialize the fixed METH-210 Q8 core with exact BF16 tied head."""
import argparse
import json
from pathlib import Path
import time

import psutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

import meth210_q8_exact_head_diagnostic as D

P = D.P
RESULT = P.DOC / "meth210_q8_exact_head_diagnostic_result.json"
RESULT_SHA = "50ae15a1df4803277e24de79e28fd952dd850fb1d3015007ac8258f795a22471"
HEAD = "model.embed_tokens.weight"
FORMAT = "QWEN25_EXACT_BF16_HEAD_Q8FFN_R8F16_PROPOSAL_V1"
EXPECTED_PAYLOAD = 820666624


def budget(start):
    row = {"seconds": time.monotonic() - start,
           "rss_bytes": psutil.Process().memory_info().rss}
    if row["seconds"] > 600 or row["rss_bytes"] > 20 * (1 << 30):
        raise RuntimeError(f"METH-211 resource stop: {row}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    assert not args.out.exists() and not args.report.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    stage = "bindings"
    try:
        torch.set_num_threads(6)
        assert P.digest(RESULT) == RESULT_SHA
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        assert all(result["gates"].values())
        assert P.digest(P.CORE) == D.CORE_SHA
        source = Path(hf_hub_download(P.M42.MODEL, "model.safetensors", revision=P.M42.REV, local_files_only=True))
        assert P.digest(source) == P.M57.MODEL_SHA
        tensors = {}
        stage = "assemble_stored_core"
        with safe_open(str(P.CORE), framework="pt", device="cpu") as previous, \
             safe_open(str(source), framework="pt", device="cpu") as donor:
            assert len(previous.keys()) == 363
            for name in previous.keys():
                value = previous.get_tensor(name)
                if name == HEAD + ".scale":
                    assert value.dtype == torch.float32 and value.shape == (151936,)
                    value = value.half()
                    assert bool(torch.isfinite(value).all() and (value > 0).all())
                tensors[name] = value.contiguous()
            tensors[HEAD + ".bf16"] = donor.get_tensor(HEAD).contiguous()
            assert tensors[HEAD + ".bf16"].dtype == torch.bfloat16
            assert tensors[HEAD + ".bf16"].shape == (151936,896)
            assert P.M17.sha(tensors[HEAD + ".q"].numpy().tobytes()) == result["proposal_codes_sha256"]
            assert P.M17.sha(tensors[HEAD + ".scale"].view(torch.uint16).numpy().tobytes()) == result["proposal_fp16_scales_sha256"]
            assert len(tensors) == 364
            payload = sum(value.numel() * value.element_size() for value in tensors.values())
            assert payload == EXPECTED_PAYLOAD
            metadata = {"experiment": "METH-211", "format": FORMAT,
                "model": P.M42.MODEL, "model_revision": P.M42.REV,
                "model_sha256": P.M57.MODEL_SHA, "meth193_core_sha256": D.CORE_SHA,
                "meth210_result_sha256": RESULT_SHA,
                "exact_tied_head_key": HEAD + ".bf16",
                "proposal_code_key": HEAD + ".q", "proposal_scale_key": HEAD + ".scale",
                "proposal_scale_dtype": "float16", "shortlist_k": "64",
                "embedding_rule": "exact BF16 row lookup",
                "probability_rule": "full exact BF16 head",
                "greedy_proposal_rule": "R8-F16 proposal then exact BF16 rows, lowest-ID ties",
                "ffn_codes": "signed symmetric int8 [-127,127]", "ffn_group_width": "64",
                "ffn_scale_dtype": "float16", "attention_dtype": "bfloat16",
                "control_dtype": "float32"}
            budget(start)
            stage = "save_and_readback"
            save_file(tensors, str(args.out), metadata=metadata)
            assert args.out.stat().st_size < 1_000_000_000
            with safe_open(str(args.out), framework="pt", device="cpu") as reread:
                assert reread.metadata() == metadata
                assert set(reread.keys()) == set(tensors)
                assert all(torch.equal(reread.get_tensor(name), value) for name, value in tensors.items())
                unchanged = [name for name in previous.keys() if name != HEAD + ".scale"]
                assert all(torch.equal(reread.get_tensor(name), previous.get_tensor(name)) for name in unchanged)
                assert torch.equal(reread.get_tensor(HEAD + ".bf16"), donor.get_tensor(HEAD))
                ffn_names = [name for name in previous.keys() if name.endswith(".q8")]
                assert len(ffn_names) == 72
                assert all(torch.equal(reread.get_tensor(name), previous.get_tensor(name)) and
                           torch.equal(reread.get_tensor(name[:-3] + ".scale"), previous.get_tensor(name[:-3] + ".scale"))
                           for name in ffn_names)
        report = {"experiment": "METH-211-exact-head-Q8-core-export",
            "path": str(args.out.resolve()), "sha256": P.digest(args.out),
            "physical_bytes": args.out.stat().st_size, "payload_bytes": payload,
            "header_bytes": args.out.stat().st_size - payload, "tensor_count": 364,
            "metadata": metadata, "source_sha256": P.M57.MODEL_SHA,
            "meth193_core_sha256": D.CORE_SHA, "meth210_result_sha256": RESULT_SHA,
            "exact_readback": True, "unchanged_tensor_count": len(unchanged),
            "all_72_ffn_codes_scales_identical": True, "exact_tied_head_equal_source": True,
            "proposal_matches_meth210": True, "ledger": result["ledger"],
            "decision": "stored_exact_head_Q8_core_ready_for_quality_and_native_composition",
            "runtime": budget(start),
            "scope": "Physical core and identity checks only; no fresh quality or native rate"}
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        assert args.out.stat().st_size + args.report.stat().st_size < 1_000_000_000
        print(json.dumps({"decision": report["decision"], "sha256": report["sha256"],
                          "physical_bytes": report["physical_bytes"], "runtime": report["runtime"]}), flush=True)
    except BaseException as error:
        args.report.with_suffix(".failure.json").write_text(json.dumps({"experiment": "METH-211-failure",
            "stage": stage, "error": repr(error), "runtime": {"seconds": time.monotonic() - start,
                "rss_bytes": psutil.Process().memory_info().rss}}, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
