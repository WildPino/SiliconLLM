#!/usr/bin/env python3
"""Load every compact-core parameter from METH-211 and reconcile development."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

import meth210_q8_exact_head_diagnostic as D
import meth211_export_exact_head_core as X

P = D.P
CORE = P.ROOT / "results/native_expert_scaling/meth211_qwen05b_exact_head_q8_core.safetensors"
CORE_SHA = "4bb34a2ac14bfa3f4fe42490a6755513cf6b08e82e981c86926e374b12353c1a"
EXPORT = P.DOC / "meth211_exact_head_core_export.json"
EXPORT_SHA = "5e48b34dacdf1e3d9f7fb7badb22e63941f064566c115c729fcdd52140a989b5"
RESULT = P.DOC / "meth210_q8_exact_head_diagnostic_result.json"
RESULT_SHA = "50ae15a1df4803277e24de79e28fd952dd850fb1d3015007ac8258f795a22471"


def load_stored(device):
    """Construct from config; overwrite all 290 parameters, with no source-weight fallback."""
    assert P.digest(CORE) == CORE_SHA
    assert P.digest(EXPORT) == EXPORT_SHA
    assert P.digest(P.M122.SPECIALIZED) == P.M122.SPECIALIZED_SHA
    assert P.digest(P.M122.TRAINING) == P.M57.TRAINING_SHA
    parent = json.loads(P.M122.TRAINING.read_text(encoding="utf-8"))["checkpoints"]["512"]
    assert P.digest(parent["path"]) == P.M57.CHECKPOINT_SHA
    config = AutoConfig.from_pretrained(P.M42.MODEL, revision=P.M42.REV, local_files_only=True)
    torch.manual_seed(212212)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16,
                                          attn_implementation="sdpa").to(device).eval()
    model.config.use_cache = False
    params = dict(model.named_parameters())
    assert len(params) == 290
    loaded, used = {}, set()
    with safe_open(str(CORE), framework="pt", device="cpu") as archive:
        metadata = archive.metadata()
        assert metadata["format"] == X.FORMAT and metadata["model_sha256"] == P.M57.MODEL_SHA
        assert metadata["meth210_result_sha256"] == RESULT_SHA
        with torch.no_grad():
            for name, param in params.items():
                organ = P.M24.classify(name, tuple(param.shape))
                if organ == "ffn":
                    keys = (name + ".q8", name + ".scale")
                    effective = P.M193.reconstruct(archive.get_tensor(keys[0]).to(device),
                                                  archive.get_tensor(keys[1]).to(device))
                elif organ == "tied_head":
                    keys = (name + ".bf16",)
                    effective = archive.get_tensor(keys[0]).to(device)
                else:
                    keys = (name,)
                    effective = archive.get_tensor(name).to(device=device, dtype=param.dtype)
                assert effective.shape == param.shape and effective.dtype == param.dtype
                param.copy_(effective)
                assert torch.equal(param, effective)
                used.update(keys)
                loaded[organ] = loaded.get(organ, 0) + 1
        proposal = {"codes": archive.get_tensor(X.HEAD + ".q").to(device),
                    "scales": archive.get_tensor(X.HEAD + ".scale").to(device)}
        assert proposal["codes"].dtype == torch.int8 and proposal["codes"].shape == (151936,896)
        assert proposal["scales"].dtype == torch.float16 and proposal["scales"].shape == (151936,)
        used.update((X.HEAD + ".q", X.HEAD + ".scale"))
        assert used == set(archive.keys()) and len(used) == 364
    assert loaded == {"tied_head":1, "attention":96, "ffn":72, "control":121}
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    del params, effective
    D.H.load_centered(model, device, parent["path"])
    wrappers = [layer.mlp for layer in model.model.layers]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, wrappers, proposal, {"parameters_loaded": 290, "tensors_consumed":364,
        "organs": loaded, "every_parameter_copy_equal":True, "tied_head_pointer_equal":True,
        "source_core_weights_loaded":False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    assert not args.out.exists()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    stage = "bindings"
    try:
        assert P.digest(RESULT) == RESULT_SHA
        previous = json.loads(RESULT.read_text(encoding="utf-8"))
        assert P.digest(P.M122.MANIFEST) == P.M122.MANIFEST_SHA
        items = json.loads(P.M122.MANIFEST.read_text(encoding="utf-8"))["items"]
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        matches = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
        assert len(matches) == 1
        device = torch.device(f"cuda:{matches[0]}")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        P.M17.MAX_SECONDS = P.MAX_SECONDS
        P.M17.MAX_RSS_BYTES = P.MAX_RSS
        P.M17.MAX_GPU_BYTES = P.MAX_GPU
        stage = "donor_choices"
        reference = AutoModelForCausalLM.from_pretrained(P.M42.MODEL, revision=P.M42.REV,
            dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(device).eval()
        donor_top = {}
        with torch.inference_mode():
            for item in items:
                ids = torch.as_tensor(item["prompt_ids"], dtype=torch.long, device=device)[None]
                donor_top[item["source_id"]] = reference(ids, use_cache=False).logits.argmax(-1)[0].cpu()
        del reference
        gc.collect()
        torch.cuda.empty_cache()
        stage = "stored_loader"
        model, wrappers, proposal, load_record = load_stored(device)
        P.budget(start, device)
        stage = "development_reconciliation"
        D.E.P = P
        cell = D.E.evaluate(model, wrappers, items, donor_top, "stored_exact_tied", device, start)
        expected = previous["arms"]["q8_exact_tied"]
        for name, metric in (("document_rows", "nats"), ("prompt_rows", "matching")):
            for row, old in zip(cell[name], expected[name]):
                assert row["source_id"] == old["source_id"] and row[metric] == old[metric]
        assert P.M17.sha(proposal["codes"].cpu().numpy().tobytes()) == previous["proposal_codes_sha256"]
        assert P.M17.sha(proposal["scales"].view(torch.uint16).cpu().numpy().tobytes()) == previous["proposal_fp16_scales_sha256"]
        result = {"experiment":"METH-212-stored-core-loader-parity", "core_sha256":CORE_SHA,
            "export_sha256":EXPORT_SHA, "meth210_result_sha256":RESULT_SHA,
            "manifest_sha256":P.M122.MANIFEST_SHA, "load_record":load_record,
            "document_rows":cell["document_rows"], "prompt_rows":cell["prompt_rows"],
            "all_document_nll_exact":True, "all_prompt_match_counts_exact":True,
            "proposal_bits_exact":True, "decision":"stored_loader_pass_fresh_quality_next",
            "runtime":{**P.budget(start,device), "gpu":torch.cuda.get_device_name(device)},
            "scope":"Stored core loading and viewed-source reconciliation only"}
        args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"decision":result["decision"],"load_record":load_record,
                          "runtime":result["runtime"]}),flush=True)
    except BaseException as error:
        args.out.with_suffix(".failure.json").write_text(json.dumps({"experiment":"METH-212-failure",
            "stage":stage,"error":repr(error),"elapsed_seconds":time.monotonic()-start},indent=2)+"\n",encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
