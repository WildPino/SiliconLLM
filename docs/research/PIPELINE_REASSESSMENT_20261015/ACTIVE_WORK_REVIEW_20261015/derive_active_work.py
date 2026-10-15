"""Reproduce favorable LongCat source-structure counts from sealed metadata.

Shapes follow the reviewed v4.57.6 source and config, not acquired tensor
headers. This is an incomplete analytical stream screen, never CPU timing.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
META = HERE / "metadata"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    retrieval = json.loads((META / "retrieval.json").read_text())
    assert retrieval["status"] == "PASS_METADATA_ONLY"
    for d in retrieval["records"]:
        p = META / d["name"]
        assert p.stat().st_size == d["bytes"] and sha(p) == d["sha256"]
    c = json.loads((META / "longcat_lite_config.json").read_text())
    index = json.loads((META / "longcat_lite_model.safetensors.index.json").read_text())
    src = (META / "modeling_longcat_flash.py").read_text()
    assert "self.intermediate_size = config.ffn_hidden_size" in src
    assert "self.mlps = nn.ModuleList([LongcatFlashMLP(config) for _ in [0, 1]])" in src
    assert "hidden_states = self.mlps[0](hidden_states)" in src
    assert "hidden_states = self.mlps[1](hidden_states)" in src
    d, h, l = c["hidden_size"], c["num_attention_heads"], c["num_layers"]
    q, k = c["q_lora_rank"], c["kv_lora_rank"]
    v, nope, rope = c["v_head_dim"], c["qk_nope_head_dim"], c["qk_rope_head_dim"]
    shapes = {"q_a_proj": (q, d), "q_b_proj": (h * (nope + rope), q),
              "kv_a_proj_with_mqa": (k + rope, d),
              "kv_b_proj": (h * (nope + v), k), "o_proj": (d, h * v)}
    dense_shapes = {"gate_proj": (c["ffn_hidden_size"], d),
                    "up_proj": (c["ffn_hidden_size"], d),
                    "down_proj": (d, c["ffn_hidden_size"])}
    roles = {}
    def add(name, shape, role, bits):
        assert name in index["weight_map"], name
        n = shape[0] * shape[1]
        roles[name] = {"shape_inferred": shape, "coefficients": n,
                       "role": role, "hypothetical_bits": bits,
                       "ideal_payload_bytes": n * bits // 8}
    for layer in range(l):
        for part in (0, 1):
            for name, shape in shapes.items():
                add(f"model.layers.{layer}.self_attn.{part}.{name}.weight", shape, "attention", 4)
            for name, shape in dense_shapes.items():
                add(f"model.layers.{layer}.mlps.{part}.{name}.weight", shape, "dense_ffn", 4)
        add(f"model.layers.{layer}.mlp.router.classifier.weight",
            (c["n_routed_experts"] + c["zero_expert_num"], d), "router", 32)
    add("lm_head.weight", (c["vocab_size"], d), "head", 4)
    branches = c["emb_split_num"] * (c["emb_neighbor_num"] - 1)
    width = d // branches
    assert branches * width == d
    table_rows = []
    for i in range(branches):
        add(f"model.ngram_embeddings.post_projs.{i}.weight", (d, width), "memory_projection", 4)
        assert f"model.ngram_embeddings.embedders.{i}.weight" in index["weight_map"]
        table_rows.append(c["ngram_vocab_size_ratio"] * c["vocab_size"] + 2 * i + 1)
    totals = {}
    for role in ("attention", "dense_ffn", "router", "head", "memory_projection"):
        values = [r for r in roles.values() if r["role"] == role]
        totals[role] = {"matrix_names_checked": len(values),
                        "coefficients": sum(x["coefficients"] for x in values),
                        "ideal_payload_bytes": sum(x["ideal_payload_bytes"] for x in values)}
    total_bytes = sum(x["ideal_payload_bytes"] for x in totals.values())
    result = {"schema": "FAVORABLE_SOURCE_ACTIVE_WORK_V1", "status": "UNCHANGED_PORT_DEPRIORITIZED",
              "model_revision": retrieval["revisions"]["meituan-longcat/LongCat-Flash-Lite"],
              "reference_revision": retrieval["revisions"]["huggingface/transformers@v4.57.6"],
              "totals": totals, "matrices": roles, "accounted_ideal_bytes": total_bytes,
              "required_GBps_at_50": total_bytes * 50 / 1e9,
              "memory": {"branches": branches, "row_width": width, "table_rows": table_rows,
                         "inferred_table_coefficients": sum(table_rows) * width,
                         "BF16_rows_bytes_per_token": branches * width * 2},
              "index_names": len(index["weight_map"]),
              "index_declared_bytes": index["metadata"]["total_size"],
              "not_acquired": ["tensor headers", "tensor values", "full checkpoint"],
              "excluded_costs": ["experts", "KV/cache/attention arithmetic", "norms and biases",
                                 "memory base row", "scales and format padding", "sampling/frontend"],
              "limits": "Ideal payloads and source-derived shapes; no physical format, quality, DRAM or latency measured. A transformed or amortized implementation needs a separate ledger.",
              "inputs_sha256": {x["name"]: x["sha256"] for x in retrieval["records"]},
              "code_sha256": sha(Path(__file__))}
    with (HERE / "active_work_ledger.json").open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "ideal_bytes": total_bytes,
                      "required_GBps_at_50": result["required_GBps_at_50"],
                      "ngram_coefficients": result["memory"]["inferred_table_coefficients"]}))


if __name__ == "__main__":
    main()
