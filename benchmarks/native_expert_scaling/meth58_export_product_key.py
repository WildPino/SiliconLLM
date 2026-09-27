"""Export bound METH-56 E128 product keys/factors and frozen native fixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "donor_adaptation/s1"))
import meth15_zero_residual_expert_smoke as M15
import meth17_fresh_transfer_audit as M17
import meth42_instruct_prompt_manifest as M42
import meth44_instruct_full_chat_smoke as M44
import meth55_product_key_experts as M55
import meth57_product_key_external_audit as M57


ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "docs/research/NATIVE_EXPERT_SCALING_20260925"
TRAINING = DIR / "meth56_product_key_retention_result.json"
MAX_SECONDS = 10 * 60
MAX_GPU_BYTES = int(10.5 * (1 << 30))
MAX_RSS_BYTES = 20 * (1 << 30)
BANK_MAGIC = b"M58PK001"
FIXTURE_MAGIC = b"M58FX001"
HEADER = "<8s7I"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def budget(start, device):
    elapsed = time.monotonic() - start
    rss = psutil.Process().memory_info().rss
    gpu = torch.cuda.max_memory_allocated(device)
    if elapsed > MAX_SECONDS or rss > MAX_RSS_BYTES or gpu > MAX_GPU_BYTES:
        raise RuntimeError(f"METH-58 budget exceeded: {elapsed:.1f}s, RSS {rss}, GPU {gpu}")
    return {"elapsed_seconds": elapsed, "rss_end_bytes": rss,
            "gpu_peak_allocated_bytes": gpu}


def bf16_bytes(tensor):
    return tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.uint16).numpy().tobytes()


def gold(wrapper, x):
    chosen, scores = wrapper.routes(x[None])
    gate = F.softmax(scores, dim=-1).to(x.dtype)
    a = wrapper.a[chosen].to(x.dtype)
    b = wrapper.b[chosen].to(x.dtype)
    hidden = F.silu(torch.einsum("nd,nkrd->nkr", x[None], a))
    out = torch.einsum("nkr,nkdr->nkd", hidden, b)
    residual = (out * gate.unsqueeze(-1)).sum(dim=1)[0]
    return chosen[0].cpu().numpy().astype("<u4"), gate[0].float().cpu().numpy().astype("<f4"), residual.float().cpu().numpy().astype("<f4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert sha(TRAINING) == M57.TRAINING_SHA
    assert sha(M57.EXTERNAL) == M57.EXTERNAL_SHA
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    checkpoint = Path(training["checkpoints"]["512"]["path"])
    assert sha(checkpoint) == M57.CHECKPOINT_SHA
    manifest = json.loads(M57.EXTERNAL.read_text(encoding="utf-8"))
    first = manifest["items"][0]
    assert first["category"] == "code" and M17.sha(np.asarray(first["prompt_ids"], dtype=np.int32).tobytes()) == first["prompt_ids_sha256"]
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = hf_hub_download(M42.MODEL, "model.safetensors", revision=M42.REV, local_files_only=True)
    assert M15.M13.sha256(source) == M57.MODEL_SHA
    tokenizer = AutoTokenizer.from_pretrained(M42.MODEL, revision=M42.REV, local_files_only=True)
    assert M15.M13.C.tok_fingerprint(tokenizer) == M15.M13.TOK_FP
    matches = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_name(i) == "NVIDIA GeForce RTX 3060"]
    assert len(matches) == 1
    device = torch.device(f"cuda:{matches[0]}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(M42.MODEL, revision=M42.REV,
        dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["updates"] == 512 and state["source_sha256"] == M57.MODEL_SHA
    assert state["teacher_sha256"] == M44.TEACHER_SHA
    assert len(state["expert_state"]) == M15.M13.L
    wrappers = []
    captured = {}
    hooks = []
    for li, layer in enumerate(model.model.layers):
        wrapper = M55.ProductKeyExperts(layer.mlp, li).to(device)
        layer.mlp = wrapper
        for key in ("a", "b", "router"):
            value = state["expert_state"][li][key]
            assert value.dtype == torch.float32 and torch.isfinite(value).all()
            getattr(wrapper, key).copy_(value.to(device))
        wrapper.enabled = True
        wrappers.append(wrapper)
        def capture(mod, inputs, layer_id=li):
            x = inputs[0].detach()
            captured[layer_id] = (x[0, 0].clone(), x[0, -1].clone())
        hooks.append(wrapper.register_forward_pre_hook(capture))
    ids = torch.tensor(first["prompt_ids"], dtype=torch.long, device=device)[None]
    with torch.inference_mode():
        model(ids, use_cache=False)
    for hook in hooks:
        hook.remove()
    assert set(captured) == set(range(M15.M13.L))
    budget(start, device)

    args.bank.parent.mkdir(parents=True, exist_ok=True)
    args.fixtures.parent.mkdir(parents=True, exist_ok=True)
    dims = (M15.M13.L, M15.M13.D, M15.R, M15.E,
            M55.ROUTER_RANK, M55.AXIS_A, M55.AXIS_B)
    with args.bank.open("wb") as file:
        file.write(struct.pack(HEADER, BANK_MAGIC, *dims))
        for li, values in enumerate(state["expert_state"]):
            router = values["router"]
            assert router.numel() == M55.ROUTER_RANK * M15.M13.D + (M55.AXIS_A + M55.AXIS_B) * M55.ROUTER_RANK
            file.write(router.detach().contiguous().numpy().astype("<f4", copy=False).tobytes())
            for key, shape in (("a", (M15.E, M15.R, M15.M13.D)),
                               ("b", (M15.E, M15.M13.D, M15.R))):
                assert tuple(values[key].shape) == shape
                file.write(bf16_bytes(values[key]))
            budget(start, device)

    # Independent byte-level readback of the complete versioned payload.
    data = np.memmap(args.bank, dtype=np.uint8, mode="r")
    header_bytes = struct.calcsize(HEADER)
    router_bytes = (M55.ROUTER_RANK * M15.M13.D +
                    (M55.AXIS_A + M55.AXIS_B) * M55.ROUTER_RANK) * 4
    factor_bytes = M15.E * M15.R * M15.M13.D * 2
    layer_bytes = router_bytes + 2 * factor_bytes
    assert len(data) == header_bytes + M15.M13.L * layer_bytes
    assert bytes(data[:header_bytes]) == struct.pack(HEADER, BANK_MAGIC, *dims)
    unique_experts_by_layer = []
    for li, values in enumerate(state["expert_state"]):
        begin = header_bytes + li * layer_bytes
        router_read = np.frombuffer(data[begin:begin + router_bytes], dtype="<f4")
        assert np.array_equal(router_read, values["router"].contiguous().numpy())
        pair_bytes = []
        for j, key in enumerate(("a", "b")):
            offset = begin + router_bytes + j * factor_bytes
            actual = bytes(data[offset:offset + factor_bytes])
            expected = bf16_bytes(values[key])
            assert actual == expected, (li, key)
            pair_bytes.append(actual)
        stride = factor_bytes // M15.E
        distinct = len({hashlib.sha256(pair_bytes[0][e * stride:(e + 1) * stride] +
                                       pair_bytes[1][e * stride:(e + 1) * stride]).digest()
                        for e in range(M15.E)})
        unique_experts_by_layer.append(distinct)
    assert min(unique_experts_by_layer) == M15.E
    del data

    generator = torch.Generator(device="cpu").manual_seed(58058)
    fixture_rows = []
    with args.fixtures.open("wb") as file:
        file.write(struct.pack(HEADER, FIXTURE_MAGIC, *dims))
        for li, wrapper in enumerate(wrappers):
            randoms = (torch.randn(M15.M13.D, generator=generator).to(torch.bfloat16),
                       (torch.randn(M15.M13.D, generator=generator) * 0.1).to(torch.bfloat16))
            for kind, x in zip(("real_first", "real_last", "synthetic_unit", "synthetic_small"),
                               (*captured[li], *randoms)):
                x = x.to(device)
                with torch.inference_mode():
                    ids_out, gate, residual = gold(wrapper, x)
                raw_x = bf16_bytes(x)
                file.write(struct.pack("<I", li))
                file.write(raw_x)
                file.write(ids_out.tobytes())
                file.write(gate.tobytes())
                file.write(residual.tobytes())
                fixture_rows.append({"layer": li, "kind": kind,
                    "activation_sha256": hashlib.sha256(raw_x).hexdigest(),
                    "chosen": ids_out.tolist()})
            budget(start, device)
    result = {"experiment": "METH-58-product-key-native-component-export",
        "checkpoint_sha256": M57.CHECKPOINT_SHA,
        "external_manifest_sha256": M57.EXTERNAL_SHA,
        "prompt_source_id": first["source_id"],
        "prompt_ids_sha256": first["prompt_ids_sha256"],
        "prompt_positions": [0, len(first["prompt_ids"]) - 1],
        "bank": {"path": str(args.bank), "sha256": sha(args.bank), "bytes": args.bank.stat().st_size},
        "fixtures": {"path": str(args.fixtures), "sha256": sha(args.fixtures), "bytes": args.fixtures.stat().st_size},
        "dimensions": dict(zip(("layers", "width", "factor_rank", "experts", "router_rank", "axis_a", "axis_b"), dims)),
        "unique_expert_factor_pairs_by_layer": unique_experts_by_layer,
        "fixture_rows": fixture_rows,
        "runtime": {**budget(start, device), "gpu": torch.cuda.get_device_name(device)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bank_sha256": result["bank"]["sha256"],
        "fixture_sha256": result["fixtures"]["sha256"], "fixture_count": len(fixture_rows),
        "runtime": result["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
