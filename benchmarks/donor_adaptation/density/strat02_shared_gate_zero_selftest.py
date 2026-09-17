"""Offline deterministic proof for the one-shared-expert router row.

This is a tiny, random-weight self-test of the pinned local Emo implementation.
It does not open donor shards, read a corpus, train, benchmark, or use a GPU.
The primary assertion is deliberately narrower than full API parity: zeroing
the final (shared) router row changes the diagnostic logits, but not the model
function, because the shared branch has one expert and therefore coefficient
one.  A routed-row negative control must change the routed selector.
"""

from __future__ import annotations

import json
from typing import Any

import strat02_mmap_teacher as teacher


def _tiny_config_values(teacher_module: Any, snapshot: Any) -> dict[str, Any]:
    values = dict(teacher_module._read_json(snapshot / "config.json", "pinned config"))
    values.update(
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        num_shared_experts_per_layer=None,
        always_active_experts=None,
        always_active_experts_per_layer=None,
        vocab_size=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=128,
        output_router_logits=False,
        use_cache=False,
    )
    return values


def _clone_model(model_class: Any, config: Any, source: Any) -> Any:
    clone = model_class(config).eval()
    result = clone.load_state_dict(source.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"clone state load failed: {result}")
    return clone


def _route_snapshot(block: Any, hidden_states: Any) -> tuple[Any, Any, Any]:
    """Reproduce the pinned legacy shared/routed split for direct inspection."""
    import torch

    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = block.gate(flat)
    shared_count = int(block.num_shared_experts)
    if shared_count != 1 or block.always_active_experts is not None:
        raise AssertionError("self-test assumptions about shared/always-active routing changed")
    standard = logits[:, :-shared_count]
    shared = logits[:, -shared_count:]
    standard_weights = torch.softmax(standard, dim=-1, dtype=torch.float32)
    routed_weights, routed_indices = torch.topk(
        standard_weights, int(block.top_k) - shared_count, dim=-1
    )
    shared_weights = torch.softmax(shared, dim=-1, dtype=torch.float32)
    return routed_indices, routed_weights.to(flat.dtype), shared_weights


def _assert_equal_tuple(left: Any, right: Any, label: str) -> None:
    if len(left) != len(right):
        raise AssertionError(f"{label} tuple lengths differ: {len(left)} != {len(right)}")
    for index, (a, b) in enumerate(zip(left, right)):
        if not torch_equal(a, b):
            raise AssertionError(f"{label}[{index}] differs")


def torch_equal(left: Any, right: Any) -> bool:
    import torch

    return bool(torch.equal(left, right))


def main() -> None:
    teacher_report = teacher.preflight()
    teacher._require_target_environment()
    import torch

    config_class, model_class = teacher._import_pinned_emo(teacher_report.snapshot)
    config = config_class(**_tiny_config_values(teacher, teacher_report.snapshot))

    torch.manual_seed(20260917)
    reference = model_class(config).eval()
    changed = _clone_model(model_class, config, reference)
    negative = _clone_model(model_class, config, reference)

    block = reference.model.layers[0].mlp
    changed_block = changed.model.layers[0].mlp
    negative_block = negative.model.layers[0].mlp
    if int(block.num_experts) != 4 or int(block.num_shared_experts) != 1:
        raise AssertionError("unexpected tiny MoE dimensions")
    if block.always_active_experts is not None:
        raise AssertionError("always_active_experts must be None")
    if tuple(block.gate.weight.shape) != (4, 64):
        raise AssertionError(f"unexpected gate shape: {tuple(block.gate.weight.shape)}")

    ids = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    with torch.no_grad():
        changed_block.gate.weight[-1].zero_()

    with torch.inference_mode():
        reference_model = reference.model(
            input_ids=ids, use_cache=False, return_dict=True, output_router_logits=False
        )
        changed_model = changed.model(
            input_ids=ids, use_cache=False, return_dict=True, output_router_logits=False
        )
        reference_lm = reference(
            input_ids=ids, use_cache=False, return_dict=True, output_router_logits=False
        )
        changed_lm = changed(
            input_ids=ids, use_cache=False, return_dict=True, output_router_logits=False
        )
        reference_diag = reference(
            input_ids=ids, use_cache=False, return_dict=True, output_router_logits=True
        )
        changed_diag = changed(
            input_ids=ids, use_cache=False, return_dict=True, output_router_logits=True
        )

    if not torch.equal(reference_model.last_hidden_state, changed_model.last_hidden_state):
        raise AssertionError("zeroing the shared gate row changed model hidden output")
    if not torch.equal(reference_lm.logits, changed_lm.logits):
        raise AssertionError("zeroing the shared gate row changed final logits")

    reference_router = reference_diag.router_logits
    changed_router = changed_diag.router_logits
    if not isinstance(reference_router, tuple) or not isinstance(changed_router, tuple):
        raise AssertionError("pinned diagnostic router logits are not a tuple")
    _assert_equal_tuple(reference_router[:-1], changed_router[:-1], "non-final router logits")
    if torch.equal(reference_router[-1], changed_router[-1]):
        raise AssertionError("zeroing shared gate row did not change raw router diagnostics")
    if not torch.equal(reference_router[-1][:, :-1], changed_router[-1][:, :-1]):
        raise AssertionError("zeroing shared gate row changed routed raw logits")

    with torch.inference_mode():
        reference_hidden = reference.model.embed_tokens(ids)
        changed_hidden = changed.model.embed_tokens(ids)
        route_indices_ref, route_weights_ref, shared_ref = _route_snapshot(block, reference_hidden)
        route_indices_changed, route_weights_changed, shared_changed = _route_snapshot(
            changed_block, changed_hidden
        )
    if not torch.equal(route_indices_ref, route_indices_changed):
        raise AssertionError("shared-row zero changed routed top-k selection")
    if not torch.equal(route_weights_ref, route_weights_changed):
        raise AssertionError("shared-row zero changed routed weights")
    if not torch.equal(shared_ref, torch.ones_like(shared_ref)):
        raise AssertionError(f"shared coefficient is not exactly one: {shared_ref}")
    if not torch.equal(shared_ref, shared_changed):
        raise AssertionError("shared coefficient changed after shared-row zero")

    # Negative control: choose the first fixed token sequence/routed row for
    # which zeroing a routed row changes top-k.  This remains a pure selector
    # control, which is stronger and less numerically fragile than requiring a
    # downstream hidden-state difference from random tiny expert weights.
    candidate_ids = (
        torch.tensor([[3, 4, 5, 6]], dtype=torch.long),
        torch.tensor([[17, 2, 31, 9]], dtype=torch.long),
        torch.tensor([[41, 42, 43, 44]], dtype=torch.long),
        torch.tensor([[101, 7, 13, 29]], dtype=torch.long),
    )
    negative_control = None
    routed_count = int(block.num_experts) - int(block.num_shared_experts)
    for candidate in candidate_ids:
        with torch.inference_mode():
            hidden = reference.model.embed_tokens(candidate)
            baseline_indices, _, _ = _route_snapshot(block, hidden)
        for routed_row in range(routed_count):
            negative.load_state_dict(reference.state_dict(), strict=True)
            with torch.no_grad():
                negative_block.gate.weight[routed_row].zero_()
            with torch.inference_mode():
                altered_indices, _, _ = _route_snapshot(negative_block, hidden)
            if not torch.equal(baseline_indices, altered_indices):
                negative_control = {
                    "token_ids": candidate.tolist(),
                    "zeroed_routed_row": routed_row,
                    "baseline_topk": baseline_indices.tolist(),
                    "altered_topk": altered_indices.tolist(),
                }
                break
        if negative_control is not None:
            break
    if negative_control is None:
        raise AssertionError("negative routed-row control never changed top-k selection")

    print(json.dumps({
        "ok": True,
        "test": "tiny_pinned_Emo_shared_gate_zero_function_invariance",
        "seed": 20260917,
        "token_ids": ids.tolist(),
        "tiny": {
            "hidden_size": 64,
            "layers": 1,
            "experts": 4,
            "routed_experts": 3,
            "shared_experts": 1,
            "top_k": int(block.top_k),
            "always_active_experts": None,
        },
        "shared_gate_row": 3,
        "model_hidden_exact_equal": True,
        "final_logits_exact_equal": True,
        "raw_router_logits_differ": True,
        "routed_topk_exact_equal": True,
        "routed_weights_exact_equal": True,
        "shared_coefficient_exact_one": True,
        "negative_control": negative_control,
        "pretrained_shards_opened": 0,
        "caveat": "random tiny functional/selector proof only; no quality, speed, donor, training, or engine.c claim",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
