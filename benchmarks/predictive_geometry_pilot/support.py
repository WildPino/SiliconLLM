"""Cloud-only acquisition and custody for the predictive-geometry campaign."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


class Budget:
    def __init__(self, seconds, reserve=0):
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.training_deadline = self.deadline - reserve

    def check(self, training=False):
        if time.monotonic() >= (self.training_deadline if training else self.deadline):
            raise TimeoutError("bounded predictive geometry budget reached")

    def elapsed(self):
        return time.monotonic() - self.started


def acquire_documents(cfg, tokenizer, output, budget):
    from datasets import load_dataset
    from common import split_articles
    exclusions = json.loads(Path(__file__).with_name("exclusions.json").read_text())
    excluded = set(exclusions["document_sha256"])
    data = load_dataset(cfg["dataset_id"], cfg["dataset_name"], revision=cfg["dataset_revision"])
    seen, windows, receipt = set(), {}, {}
    for split, source in (("train", "train"), ("dev", "validation"), ("test", "test")):
        budget.check()
        candidates = sorted(split_articles(data[source]),
                            key=lambda x: sha((str(cfg["seed"]) + ":" + x).encode()))
        selected, identities = [], []
        for text in candidates:
            budget.check()
            digest = sha(text.encode())
            if digest in excluded or digest in seen:
                continue
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(tokens) < cfg["context"] + 1:
                continue
            offset = int(sha((digest + ":" + str(cfg["seed"])).encode())[:12], 16) % (len(tokens) - cfg["context"])
            ids = tokens[offset:offset + cfg["context"] + 1]
            seen.add(digest)
            selected.append(ids)
            identities.append({"document_sha256": digest, "source_split": source,
                               "source_tokens": len(tokens), "offset": offset,
                               "token_sha256": sha(json.dumps(ids).encode())})
            if len(selected) == cfg[split + "_docs"]:
                break
        if len(selected) != cfg[split + "_docs"]:
            raise RuntimeError(f"not enough fresh disjoint {source} articles: {len(selected)}")
        windows[split] = selected
        receipt[split] = identities
    write(output / "split_manifest.json", {"splits": receipt,
          "excluded_previous_articles": len(excluded),
          "exclusions_file_sha256": sha(Path(__file__).with_name("exclusions.json").read_bytes())})
    write(output / "token_windows.json", windows)
    return windows


def capture(torch, donor, windows, cfg, output, budget, device):
    captured, records = {}, {}

    def before(module, args, kwargs):
        captured["x"] = (args[0] if args else kwargs["hidden_states"]).detach().float().cpu()

    def after(module, args, value):
        value = value[0] if isinstance(value, tuple) else value
        captured["y"] = value.detach().float().cpu()

    layers = donor.model.layers
    hooks = [layers[cfg["layer_start"]].register_forward_pre_hook(before, with_kwargs=True),
             layers[cfg["layer_end"] - 1].register_forward_hook(after)]
    try:
        for split, docs in windows.items():
            xs, ys = [], []
            for i, ids in enumerate(docs):
                budget.check()
                with torch.inference_mode():
                    donor.model(torch.tensor([ids[:-1]], device=device), use_cache=False)
                for name in ("x", "y"):
                    if not torch.isfinite(captured[name]).all().item():
                        raise FloatingPointError("nonfinite source capture")
                xs.append(captured.pop("x")); ys.append(captured.pop("y"))
                if (i + 1) % 16 == 0:
                    print(f"CAPTURE {split} {i + 1}/{len(docs)}", flush=True)
            records[split] = {"x": torch.cat(xs), "y": torch.cat(ys)}
            torch.save(records[split], output / ("capture_" + split + ".pt"))
    finally:
        for hook in hooks:
            hook.remove()
    return records


def validate_config(cfg):
    for name in ("model_revision", "dataset_revision"):
        value = cfg[name]
        if len(value) != 40 or any(x not in "0123456789abcdef" for x in value):
            raise ValueError("immutable revision required")
    if not 0 < cfg["wall_seconds"] <= 6500:
        raise ValueError("internal cap exceeds frozen two-hour allocation")
    if cfg["mode"] not in ("geometry", "temporal", "reuse"):
        raise ValueError("unknown experiment mode")
    if (cfg["train_docs"], cfg["dev_docs"], cfg["test_docs"]) != (64, 12, 12):
        raise ValueError("frozen article cohort changed")
    if cfg["context"] != 192 or cfg["anchor"] != 63:
        raise ValueError("frozen intervention geometry changed")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/kaggle").exists():
        raise RuntimeError("model and tensor execution is restricted to Kaggle")
    cfg = validate_config(json.loads(args.config.read_text()))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists():
        raise RuntimeError("do not overwrite an experiment")
    budget = Budget(cfg["wall_seconds"], 600)
    status = {"status": "RUNNING", "mode": cfg["mode"], "started_wall_time": time.time()}
    write(output / "status.json", status)
    write(output / "effective_config.json", cfg)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if importlib.metadata.version("transformers") != "4.51.3":
            raise RuntimeError("pinned decoder implementation required")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("the requested T4 pair is required")
        names = [torch.cuda.get_device_name(i) for i in range(2)]
        if any("T4" not in name for name in names):
            raise RuntimeError("wrong GPU type")
        torch.set_num_threads(2)
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        random.seed(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")
        module = importlib.import_module(cfg["mode"])
        geometry = importlib.import_module("geometry")
        geometry.self_test(torch)
        if cfg["mode"] != "geometry" and hasattr(module, "self_test"):
            module.self_test(torch)
        write(output / "environment.json", {"python": sys.version, "torch": torch.__version__,
              "packages": {name: importlib.metadata.version(name) for name in
                           ("transformers", "datasets", "huggingface-hub", "safetensors")},
              "gpus": names, "used_gpu_indices": [0], "unused_gpu_indices": [1],
              "donor_dtype": "float32", "local_model_compute": False})
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
        windows = acquire_documents(cfg, tokenizer, output, budget)
        budget.check()
        donor = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"],
                torch_dtype=torch.float32, attn_implementation="sdpa").to(device).eval()
        donor.requires_grad_(False)
        if getattr(donor.config, "_commit_hash", None) != cfg["model_revision"]:
            raise RuntimeError("source revision custody mismatch")
        write(output / "donor_config.json", donor.config.to_dict())
        records = capture(torch, donor, windows, cfg, output, budget, device)
        result = module.run(torch, donor, records, windows, cfg, output, budget, device)
        write(output / "result.json", result if result is not None else {"scope": "inspect module report"})
        status.update(status="COMPLETE", finished_wall_time=time.time(),
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device))
    except BaseException as exc:
        status.update(status="FAILED_OR_INCOMPLETE", exception_type=type(exc).__name__,
                      exception=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        status["elapsed_seconds"] = budget.elapsed()
        write(output / "status.json", status)
        pins = {}
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != output / "artifact_manifest.json":
                h = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        h.update(block)
                pins[path.relative_to(output).as_posix()] = {"bytes": path.stat().st_size, "sha256": h.hexdigest()}
        write(output / "artifact_manifest.json", pins)


if __name__ == "__main__":
    main()
