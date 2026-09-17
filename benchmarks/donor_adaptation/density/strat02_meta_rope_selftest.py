"""Offline tiny-Emo parity test for restoring nonpersistent RoPE from meta.

Uses the pinned local model source but random, tiny weights.  It does not open
or hash pretrained shards and makes no quality or performance claim.
"""

from __future__ import annotations

import json

import strat02_mmap_teacher as teacher


def main() -> None:
    report = teacher.preflight()
    teacher._require_target_environment()
    import torch

    Config, Model = teacher._import_pinned_emo(report.snapshot)
    config_values = dict(teacher._read_json(report.snapshot / "config.json", "pinned config"))
    config_values.update(
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        vocab_size=256,
        pad_token_id=0,
        eos_token_id=2,
        max_position_embeddings=128,
        output_router_logits=False,
    )
    config = Config(**config_values)
    torch.manual_seed(20260917)
    reference = Model(config).eval()
    source = reference.state_dict()
    with torch.device("meta"):
        mapped = Model(config)
    result = mapped.load_state_dict(source, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"tiny assign=True load failed: {result}")
    before = [name for name, tensor in mapped.named_buffers() if tensor.is_meta]
    if before != ["model.rotary_emb.inv_freq"]:
        raise AssertionError(f"unexpected pre-restoration meta buffers: {before}")
    ids = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    mapped.eval()
    try:
        with torch.inference_mode():
            mapped.model(input_ids=ids, use_cache=False, return_dict=True)
    except NotImplementedError as exc:
        if "Cannot copy out of meta tensor" not in str(exc):
            raise
    else:
        raise AssertionError("unrestored meta RoPE unexpectedly completed a forward")
    teacher._restore_nonpersistent_rope_buffer(mapped)
    teacher._assert_pointer_sharing(mapped, source, "all")
    mapped.eval()
    if not torch.equal(reference.model.rotary_emb.inv_freq, mapped.model.rotary_emb.inv_freq):
        raise AssertionError("restored RoPE inv_freq differs from CPU constructor")
    with torch.inference_mode():
        expected = reference.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        observed = mapped.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    max_abs = float((expected - observed).abs().max().item())
    if max_abs > 1e-5:
        raise AssertionError(f"tiny CPU-vs-meta-loaded hidden mismatch: {max_abs}")
    print(json.dumps({
        "ok": True,
        "test": "tiny_pinned_Emo_CPU_vs_meta_assign_RoPE",
        "max_abs_hidden": max_abs,
        "parameters": sum(tensor.numel() for tensor in source.values()),
        "pretrained_shards_opened": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
