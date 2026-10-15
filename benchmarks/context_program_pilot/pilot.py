"""Bounded, causal context-program pilot. Execute numerical work on Kaggle only.

No useful-chat or CPU-throughput admission is implied by this exploratory test.
The only retained donor states used online are inputs BEFORE the omitted layers.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback


DEFAULTS = dict(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    model_revision="7ae557604adf67be50417f59c2c2f167def9a775",
    dataset_id="Salesforce/wikitext", dataset_name="wikitext-2-raw-v1",
    dataset_revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    seed=1729, context=256, prefix=32, horizons=[16, 32, 64],
    layer_start=8, layer_end=12, rank=64, program_rank=32,
    train_docs=128, dev_docs=24, test_docs=24, steps_per_arm=1200,
    batch_docs=8, learning_rate=0.001, wall_seconds=12600,
    reserve_seconds=2400, arm_seconds=1800, eval_docs=12,
    generation_docs=4, generation_tokens=16,
)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def validate_config(raw):
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = DEFAULTS | raw
    for key in ("model_revision", "dataset_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", cfg[key]):
            raise ValueError(f"{key} must be an immutable 40-character SHA")
    if cfg["prefix"] < 1 or cfg["context"] <= cfg["prefix"]:
        raise ValueError("Require 0 < prefix < context")
    if cfg["horizons"] != [16, 32, 64]:
        raise ValueError("Frozen first-session horizons are [16,32,64]")
    if not 0 <= cfg["layer_start"] < cfg["layer_end"]:
        raise ValueError("Invalid contiguous layer interval")
    if not 0 < cfg["reserve_seconds"] < cfg["wall_seconds"] <= 14400:
        raise ValueError("Invalid wall cap/evaluation reserve")
    for key in ("rank", "program_rank", "train_docs", "dev_docs", "test_docs",
                "steps_per_arm", "batch_docs", "eval_docs", "generation_docs",
                "generation_tokens", "arm_seconds"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    return cfg


class Budget:
    def __init__(self, seconds, reserve):
        self.start = time.monotonic()
        self.deadline = self.start + seconds
        self.training_deadline = self.deadline - reserve

    def check(self, training=False):
        deadline = self.training_deadline if training else self.deadline
        if time.monotonic() >= deadline:
            raise TimeoutError("Training reserve reached" if training else "Wall cap reached")

    def elapsed(self):
        return time.monotonic() - self.start


def split_articles(rows):
    """Wikitext's top-level '= title =' rows delimit original documents."""
    current = []
    for row in rows:
        text = row["text"].strip()
        heading = bool(re.fullmatch(r"=\s+[^=].*?\s+=", text))
        if heading and current:
            yield "\n".join(current)
            current = []
        if text:
            current.append(text)
    if current:
        yield "\n".join(current)


def load_documents(cfg, tokenizer, output, budget):
    from datasets import load_dataset
    ds = load_dataset(cfg["dataset_id"], cfg["dataset_name"],
                      revision=cfg["dataset_revision"])
    used = {}
    result, receipt = {}, {}
    for label, source in (("train", "train"), ("dev", "validation"), ("test", "test")):
        budget.check(training=True)
        articles = list(split_articles(ds[source]))
        # Hash membership is decided before selecting usable windows. The source
        # partition is preserved; duplicate documents are dropped, never reassigned.
        ranked = sorted(articles, key=lambda text: sha_bytes(
            f"{cfg['seed']}:{text}".encode("utf8")))
        chosen, manifests = [], []
        duplicate_count = 0
        for text in ranked:
            digest = sha_bytes(text.encode("utf8"))
            if digest in used:
                duplicate_count += 1
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) < cfg["context"] + 1:
                continue
            used[digest] = label
            # One deterministic bounded window per document, including gold next ID.
            offset = int(digest[:12], 16) % (len(ids) - cfg["context"])
            window = ids[offset:offset + cfg["context"] + 1]
            chosen.append(window)
            manifests.append(dict(document_sha256=digest, source_split=source,
                                  source_tokens=len(ids), offset=offset,
                                  token_sha256=sha_bytes(json.dumps(window).encode())))
            if len(chosen) == cfg[f"{label}_docs"]:
                break
        if len(chosen) < cfg[f"{label}_docs"]:
            raise RuntimeError(f"Insufficient disjoint natural documents in {source}: {len(chosen)}")
        result[label] = chosen
        receipt[label] = dict(documents=manifests, duplicate_documents_dropped=duplicate_count,
                              source_fingerprint=getattr(ds[source], "_fingerprint", None),
                              cache_files=[dict(name=Path(item["filename"]).name,
                                                bytes=Path(item["filename"]).stat().st_size,
                                                sha256=sha_file(item["filename"]))
                                           for item in ds[source].cache_files])
    write_json(output / "splits.json", receipt)
    write_json(output / "token_windows.json", result)
    return result


def build_classes(torch):
    nn = torch.nn

    class Surrogate(nn.Module):
        def __init__(self, dim, rank, program_rank, mode, scale):
            super().__init__()
            self.mode = mode
            self.register_buffer("scale", torch.tensor(float(scale)))
            self.down = nn.Linear(dim, rank)
            self.up = nn.Linear(rank, dim)
            if mode != "fixed":
                self.program = nn.Sequential(nn.Linear(dim, program_rank), nn.SiLU(),
                                             nn.Linear(program_rank, 2 * rank))
            # A shared initialization begins at identity, without copying donor maps.
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        def forward(self, x, prefix, plan=None):
            z = torch.nn.functional.silu(self.down(x / self.scale))
            if self.mode != "fixed":
                # Evaluate the generator once per prefix update, then reuse its
                # modulation over the following block. Early prefill has one
                # causal update per available prefix, explicitly charged below.
                p = prefix if plan is None else prefix[:, plan[0]]
                code = self.program(p / self.scale)
                if plan is not None:
                    code = code[:, plan[1]]
                gain, shift = code.chunk(2, dim=-1)
                if self.mode == "static":
                    # Prefix-dependent additive-vector memory control. Unlike
                    # dynamic modulation, it cannot change the token map's gain.
                    z = z + 0.5 * torch.tanh(gain + shift)
                else:
                    z = z * (1 + 0.5 * torch.tanh(gain)) + 0.5 * torch.tanh(shift)
            return x + self.scale * self.up(z)

    return Surrogate


def prefix_features(torch, x, prefix, horizon):
    """Prefix mean excludes current token and is frozen over each future block.

    For training t>=prefix. During whole-model replacement early positions use
    only available preceding inputs (zero at t=0); no original block executes.
    """
    length = x.shape[1]
    positions = torch.arange(length, device=x.device)
    anchors = torch.where(positions < prefix, positions,
                          prefix + ((positions - prefix) // horizon) * horizon)
    starts = (anchors - prefix).clamp(min=0)
    sums = torch.cat([torch.zeros_like(x[:, :1]), x.cumsum(dim=1)], dim=1)
    totals = sums[:, anchors] - sums[:, starts]
    return totals / (anchors - starts).clamp(min=1)[None, :, None]


def program_plan(torch, length, prefix, horizon, device):
    positions = torch.arange(length, device=device)
    anchors = torch.where(positions < prefix, positions,
                          prefix + ((positions - prefix) // horizon) * horizon)
    return torch.unique(anchors, sorted=True, return_inverse=True)


def apply_surrogate(torch, model, x, cfg, horizon, program_source=None):
    source = x if program_source is None else program_source
    program = prefix_features(torch, source, cfg["prefix"], horizon)
    plan = program_plan(torch, x.shape[1], cfg["prefix"], horizon, x.device)
    return model(x, program, plan)


def finite(torch, value, what):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {what}")


def capture(torch, donor, windows, cfg, output, budget, device):
    layers = donor.model.layers
    records = {}
    buffer = {}

    def before(_module, args, kwargs):
        buffer["x"] = (args[0] if args else kwargs["hidden_states"]).detach().cpu().half()

    def after(_module, args, value):
        hidden = value[0] if isinstance(value, tuple) else value
        buffer["y"] = hidden.detach().cpu().half()

    handles = [layers[cfg["layer_start"]].register_forward_pre_hook(before, with_kwargs=True),
               layers[cfg["layer_end"] - 1].register_forward_hook(after)]
    try:
        for split, docs in windows.items():
            xs, ys = [], []
            for ids in docs:
                budget.check(training=True)
                tokens = torch.tensor([ids[:-1]], device=device)
                with torch.inference_mode():
                    result = donor(tokens, use_cache=False)
                finite(torch, buffer["x"], "capture input")
                finite(torch, buffer["y"], "capture target")
                finite(torch, result.logits, "donor logits")
                xs.append(buffer.pop("x"))
                ys.append(buffer.pop("y"))
                del result
            records[split] = dict(x=torch.cat(xs), y=torch.cat(ys))
            torch.save(records[split], output / f"capture_{split}.pt")
            append_json(output / "events.jsonl", dict(event="capture", split=split, count=len(docs),
                                                       elapsed=budget.elapsed()))
    finally:
        for handle in handles:
            handle.remove()
    return records


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_arms(torch, cls, records, cfg, output, budget, device):
    train = records["train"]
    dim = train["x"].shape[-1]
    scale = train["x"].float().square().mean().sqrt().item()
    if not math.isfinite(scale) or scale <= 0:
        raise FloatingPointError("Invalid training-only input scale")
    prototype = cls(dim, cfg["rank"], cfg["program_rank"], "conditional", scale)
    target_count = count_parameters(prototype)
    # Select fixed width solely by parameter budget, before observing errors.
    fixed_rank = max(1, round((target_count - dim) / (2 * dim + 1)))
    del prototype
    arms, receipts = {}, {}
    for arm in ("fixed", "conditional", "static"):
        budget.check(training=True)
        torch.manual_seed(cfg["seed"])
        rank = fixed_rank if arm == "fixed" else cfg["rank"]
        model = cls(dim, rank, cfg["program_rank"], arm, scale).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)
        generator = torch.Generator().manual_seed(cfg["seed"] + 1)
        started = time.monotonic()
        step, applied_steps, reason, best_dev, best_step = 0, 0, "fixed_steps", float("inf"), 0
        curves = []
        try:
            for step in range(1, cfg["steps_per_arm"] + 1):
                budget.check(training=True)
                if time.monotonic() - started >= cfg["arm_seconds"]:
                    reason = "arm_wall_cap"
                    break
                indices = torch.randint(len(train["x"]), (cfg["batch_docs"],), generator=generator)
                x = train["x"][indices].to(device=device, dtype=torch.float32)
                y = train["y"][indices].to(device=device, dtype=torch.float32)
                horizon = cfg["horizons"][(step - 1) % len(cfg["horizons"])]
                prediction = apply_surrogate(torch, model, x, cfg, horizon)
                tail = slice(cfg["prefix"], None)
                loss = (prediction[:, tail] - y[:, tail]).square().mean() / scale ** 2
                finite(torch, loss, "training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None:
                        finite(torch, p.grad, "gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                applied_steps += 1
                if step == 1 or step % 50 == 0 or step == cfg["steps_per_arm"]:
                    model.eval()
                    with torch.inference_mode():
                        dx = records["dev"]["x"].to(device=device, dtype=torch.float32)
                        dy = records["dev"]["y"].to(device=device, dtype=torch.float32)
                        dp = apply_surrogate(torch, model, dx, cfg, 16)
                        dev_loss = (dp[:, tail] - dy[:, tail]).square().mean().item() / scale ** 2
                    if not math.isfinite(dev_loss):
                        raise FloatingPointError("Nonfinite dev loss")
                    row = dict(step=step, loss=loss.item(), dev_loss=dev_loss,
                               elapsed=time.monotonic() - started)
                    curves.append(row)
                    append_json(output / "curves.jsonl", dict(arm=arm, **row))
                    checkpoint = dict(state_dict=model.state_dict(), step=step,
                                      optimizer=optimizer.state_dict(), mode=arm, rank=rank,
                                      scale=scale, dim=dim, config=cfg)
                    torch.save(checkpoint, output / f"{arm}_latest.pt")
                    if dev_loss < best_dev:
                        best_dev, best_step = dev_loss, step
                        torch.save(checkpoint, output / f"{arm}_best.pt")
                    model.train()
        except TimeoutError:
            reason = "training_reserve"
        if best_step == 0:
            raise RuntimeError(f"No usable checkpoint for {arm}")
        for parameter in model.parameters():
            finite(torch, parameter, "trained parameter")
        torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                        step=applied_steps, mode=arm, rank=rank, scale=scale, dim=dim, config=cfg),
                   output / f"{arm}_latest.pt")
        checkpoint = torch.load(output / f"{arm}_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        arms[arm] = model
        receipts[arm] = dict(parameters=count_parameters(model), rank=rank,
                            selected_step=best_step, completed_step=applied_steps, stop=reason,
                            seconds=time.monotonic() - started, best_dev=best_dev)
        write_json(output / "training.json", receipts)
    return arms, receipts


def summarize(values):
    return sum(values) / len(values)


def paired_bootstrap_gain(torch, fixed, conditional, seed):
    a = torch.tensor(fixed, dtype=torch.float64)
    b = torch.tensor(conditional, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(a), (1000, len(a)), generator=generator)
    gains = 1 - b[indices].sum(dim=1) / a[indices].sum(dim=1).clamp(min=1e-30)
    return [torch.quantile(gains, q).item() for q in (0.025, 0.5, 0.975)]


def mapping_evaluation(torch, arms, records, cfg, output, budget, device, splits=("dev", "test")):
    report = {}
    for split in splits:
        data = records[split]
        report[split] = {}
        for horizon in cfg["horizons"]:
            metrics = {}
            for arm in ("fixed", "conditional", "static", "shuffled"):
                errors, energy, residual, cosine = [], [], [], []
                for index in range(len(data["x"])):
                    budget.check()
                    x = data["x"][index:index + 1].to(device=device, dtype=torch.float32)
                    y = data["y"][index:index + 1].to(device=device, dtype=torch.float32)
                    # A fixed nonidentity circular permutation gives no self-programs.
                    other = (index + 1) % len(data["x"])
                    source = data["x"][other:other + 1].to(device=device, dtype=torch.float32) if arm == "shuffled" else x
                    with torch.inference_mode():
                        prediction = apply_surrogate(torch, arms["conditional" if arm == "shuffled" else arm],
                                                     x, cfg, horizon, source)
                    tail = slice(cfg["prefix"], None)
                    delta = prediction[:, tail] - y[:, tail]
                    finite(torch, delta, "mapping evaluation")
                    errors.append(delta.square().sum().item())
                    energy.append(y[:, tail].square().sum().item())
                    residual.append((y[:, tail] - x[:, tail]).square().sum().item())
                    cosine.append(torch.nn.functional.cosine_similarity(prediction[:, tail], y[:, tail], dim=-1).mean().item())
                metrics[arm] = dict(document_sse=errors, pooled_sse=sum(errors),
                                    relative_output_sse=sum(errors) / sum(energy),
                                    relative_residual_sse=sum(errors) / sum(residual),
                                    cosine=summarize(cosine))
            metrics["conditional_gain_ci95"] = paired_bootstrap_gain(
                torch, metrics["fixed"]["document_sse"], metrics["conditional"]["document_sse"], cfg["seed"])
            report[split][str(horizon)] = metrics
    write_json(output / ("mapping_" + "_".join(splits) + ".json"), report)
    return report


@contextlib.contextmanager
def replace_layers(torch, donor, surrogate, cfg, horizon):
    """Physically bypass every omitted forward; no output hook runs originals.

    Transformers 4.51.3 Qwen2DecoderLayer returns a tuple. The caller enforces
    use_cache=False. Full prefixes are recomputed; no omitted-layer KV survives.
    """
    layers = donor.model.layers
    originals = [layers[i].forward for i in range(cfg["layer_start"], cfg["layer_end"])]

    def first(hidden_states, *args, **kwargs):
        if kwargs.get("use_cache", False) or kwargs.get("output_attentions", False):
            raise ValueError("Replacement requires use_cache=False, output_attentions=False")
        x = hidden_states.float()
        value = apply_surrogate(torch, surrogate, x, cfg, horizon).to(dtype=hidden_states.dtype)
        return (value,)

    def skipped(hidden_states, *args, **kwargs):
        if kwargs.get("use_cache", False) or kwargs.get("output_attentions", False):
            raise ValueError("Replacement requires use_cache=False, output_attentions=False")
        return (hidden_states,)

    try:
        for index in range(cfg["layer_start"], cfg["layer_end"]):
            layers[index].forward = first if index == cfg["layer_start"] else skipped
        yield
    finally:
        for index, original in zip(range(cfg["layer_start"], cfg["layer_end"]), originals):
            layers[index].forward = original


def whole_model_evaluation(torch, donor, tokenizer, arms, windows, cfg, output, budget, device,
                           splits=("dev", "test"), generate=True):
    report = {}
    for split in splits:
        rows = []
        for index, ids in enumerate(windows[split][:cfg["eval_docs"]]):
            budget.check()
            tokens = torch.tensor([ids[:-1]], device=device)
            gold = torch.tensor([ids[1:]], device=device)
            tail = slice(cfg["prefix"], None)
            with torch.inference_mode():
                teacher = donor(tokens, use_cache=False).logits[:, tail].float()
                teacher_log = teacher.log_softmax(dim=-1)
                teacher_prob = teacher_log.exp()
                donor_nll = -teacher_log.gather(-1, gold[:, tail, None]).mean().item()
                rows.append(dict(document=index, arm="donor", horizon=0, nll=donor_nll, kl=0.0, disagreement=0.0))
                for arm, horizon in (("fixed", 32), ("static", 32), ("conditional", 16),
                                     ("conditional", 32), ("conditional", 64)):
                    budget.check()
                    with replace_layers(torch, donor, arms[arm], cfg, horizon):
                        logits = donor(tokens, use_cache=False).logits[:, tail].float()
                    finite(torch, logits, "whole-model logits")
                    logp = logits.log_softmax(dim=-1)
                    nll = -logp.gather(-1, gold[:, tail, None]).mean().item()
                    kl = (teacher_prob * (teacher_log - logp)).sum(dim=-1).mean().item()
                    disagreement = (logits.argmax(-1) != teacher.argmax(-1)).float().mean().item()
                    rows.append(dict(document=index, arm=arm, horizon=horizon,
                                     nll=nll, kl=kl, disagreement=disagreement))
                    del logits, logp
                del teacher, teacher_log, teacher_prob
        report[split] = rows
        write_json(output / ("whole_model_" + "_".join(splits) + ".json"), report)
    generations = []
    generation_windows = windows["test"][:cfg["generation_docs"]] if generate else []
    for index, ids in enumerate(generation_windows):
        prompt = ids[:cfg["prefix"] * 2]
        for arm in ("donor", "fixed", "conditional", "static"):
            budget.check()
            tokens = torch.tensor([prompt], device=device)
            started = time.monotonic()
            manager = contextlib.nullcontext() if arm == "donor" else replace_layers(torch, donor, arms[arm], cfg, 16)
            with manager, torch.inference_mode():
                for _ in range(cfg["generation_tokens"]):
                    budget.check()
                    logits = donor(tokens, use_cache=False).logits[:, -1]
                    finite(torch, logits, "own-history generation")
                    next_id = logits.argmax(-1, keepdim=True)
                    tokens = torch.cat([tokens, next_id], dim=1)
            generated = tokens[0, len(prompt):].tolist()
            generations.append(dict(document=index, arm=arm, prompt_ids=prompt,
                                    generated_ids=generated, text=tokenizer.decode(generated),
                                    seconds=time.monotonic() - started,
                                    unique_token_fraction=len(set(generated)) / len(generated)))
            write_json(output / "generations.json", generations)
    return report, generations


def exploratory_gate(mapping, whole, training=None, cfg=None, splits=("dev",)):
    checks = {}
    for split in splits:
        m = mapping[split]["16"]
        fixed, conditional = m["fixed"]["pooled_sse"], m["conditional"]["pooled_sse"]
        gain = 1 - conditional / fixed
        shuffle_loss = (m["shuffled"]["pooled_sse"] - conditional) / fixed
        static_gain = 1 - conditional / m["static"]["pooled_sse"]
        rows = whole[split]
        fixed_nll = summarize([r["nll"] for r in rows if r["arm"] == "fixed"])
        cond_nll = summarize([r["nll"] for r in rows if r["arm"] == "conditional" and r["horizon"] == 16])
        checks[split] = dict(mapping_gain=gain, shuffle_loss=shuffle_loss,
                             gain_over_static=static_gain, fixed_nll=fixed_nll, conditional_nll=cond_nll,
                             pass_mapping=gain >= 0.10,
                             pass_shuffle=shuffle_loss >= max(0.0, gain / 2),
                             pass_static=static_gain >= 0.05,
                             pass_whole_nll=cond_nll <= fixed_nll,
                             pass_ci=m["conditional_gain_ci95"][0] > 0)
    passed = all(v for split in checks.values() for key, v in split.items() if key.startswith("pass_"))
    complete = training is None or all(r["completed_step"] == cfg["steps_per_arm"] and
                                      r["stop"] == "fixed_steps" for r in training.values())
    status = "INCONCLUSIVE" if not complete else "EXPLORATORY_POSITIVE" if passed else "EXPLORATORY_NEGATIVE"
    return dict(status=status,
                checks=checks, admission="No useful chatbot, capacity scaling, or CPU speed claim")


def self_test():
    """Small CPU invariance checks only; intended to run inside Kaggle."""
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and not Path("/kaggle").exists():
        raise RuntimeError("Tensor self-tests are restricted to Kaggle; run test_pilot.py locally")
    import torch
    x = torch.arange(2 * 96 * 4, dtype=torch.float32).reshape(2, 96, 4)
    for horizon in (16, 32, 64):
        p = prefix_features(torch, x, 32, horizon)
        assert torch.equal(p[:, 32], x[:, :32].mean(1))
        changed = x.clone()
        changed[:, 32:] += 100000
        changed_p = prefix_features(torch, changed, 32, horizon)
        assert torch.equal(p[:, 32:32 + horizon], changed_p[:, 32:32 + horizon])
        assert torch.equal(p[:, 0], torch.zeros_like(p[:, 0]))
    cls = build_classes(torch)
    model = cls(4, 3, 2, "conditional", 1.0)
    assert torch.equal(model(x, prefix_features(torch, x, 32, 32)), x)
    generated_shapes = []
    handle = model.program.register_forward_hook(lambda module, inputs, value: generated_shapes.append(value.shape))
    apply_surrogate(torch, model, x, {"prefix": 32}, 16)
    handle.remove()
    assert generated_shapes == [torch.Size([2, 36, 6])], generated_shapes
    # Confirm omitted originals are never invoked and all original methods restore.
    class Layer:
        def forward(self, hidden_states, **kwargs):
            raise AssertionError("An omitted donor layer executed")
    class Holder:
        pass
    donor = Holder()
    donor.model = Holder()
    donor.model.layers = [Layer(), Layer()]
    originals = [layer.forward for layer in donor.model.layers]
    cfg = dict(layer_start=0, layer_end=2, prefix=32)
    with replace_layers(torch, donor, model, cfg, 32):
        for layer in donor.model.layers:
            x = layer.forward(x, use_cache=False)[0]
    assert all(layer.forward == original for layer, original in zip(donor.model.layers, originals))
    # Exercise the pinned decoder's actual tuple/call convention, without a
    # checkpoint download or any training. This tiny random model exists only
    # for the remote CPU smoke test.
    from transformers import Qwen2Config, Qwen2ForCausalLM
    tiny_config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                              num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=2, max_position_embeddings=32)
    tiny = Qwen2ForCausalLM(tiny_config).eval()
    tiny_surrogate = cls(16, 3, 2, "conditional", 1.0).eval()
    tiny_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.inference_mode():
        before = tiny(tiny_ids, use_cache=False).logits
        with replace_layers(torch, tiny, tiny_surrogate,
                            dict(layer_start=0, layer_end=2, prefix=4), 16):
            patched = tiny(tiny_ids, use_cache=False).logits
        after = tiny(tiny_ids, use_cache=False).logits
    assert patched.shape == before.shape
    assert torch.isfinite(patched).all()
    assert torch.equal(before, after)
    print("CPU invariance and physical-bypass self-tests passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None or args.config is None:
        parser.error("--config and --output are required")
    cfg = validate_config(json.loads(args.config.read_text(encoding="utf8")))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise RuntimeError("Output already contains a run; use a fresh directory")
    budget = Budget(cfg["wall_seconds"], cfg["reserve_seconds"])
    write_json(output / "config.json", cfg)
    status = dict(status="RUNNING", started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  exploratory_only=True)
    write_json(output / "status.json", status)
    try:
        if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and not Path("/kaggle").exists():
            raise RuntimeError("Numerical/model work is restricted to Kaggle")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if importlib.metadata.version("transformers") != "4.51.3":
            raise RuntimeError("This physical-bypass implementation requires transformers==4.51.3")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("This first-session receipt requires Kaggle T4 x2")
        if any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
            raise RuntimeError("Configured T4 x2 allocation is required; refusing a different GPU")
        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(2)
        device = torch.device("cuda:0")
        environment = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                           packages={name: importlib.metadata.version(name) for name in
                                     ("transformers", "datasets", "huggingface-hub", "safetensors")},
                           gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                           used_gpu=0, unused_gpu_indices=list(range(1, torch.cuda.device_count())),
                           source_sha256=sha_bytes(Path(__file__).read_bytes()),
                           no_cpu_throughput_claim=True)
        write_json(output / "environment.json", environment)
        self_test()
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
        windows = load_documents(cfg, tokenizer, output, budget)
        budget.check(training=True)
        donor = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float16,
            attn_implementation="sdpa").to(device).eval()
        donor.requires_grad_(False)
        if cfg["layer_end"] > len(donor.model.layers):
            raise ValueError("Layer interval exceeds donor depth")
        write_json(output / "donor_config.json", donor.config.to_dict())
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(repo_id=cfg["model_id"], revision=cfg["model_revision"],
                                          local_files_only=True))
        write_json(output / "donor_source_files.json", {
            str(path.relative_to(snapshot)): dict(bytes=path.stat().st_size, sha256=sha_file(path))
            for path in sorted(snapshot.rglob("*")) if path.is_file()})
        records = capture(torch, donor, windows, cfg, output, budget, device)
        cls = build_classes(torch)
        arms, training = train_arms(torch, cls, records, cfg, output, budget, device)
        # DEV decision is sealed on disk before any TEST evaluation. Checkpoint
        # choice uses DEV only; TEST confirms a fixed recipe and never selects it.
        mapping = mapping_evaluation(torch, arms, records, cfg, output, budget, device, splits=("dev",))
        whole, _ = whole_model_evaluation(torch, donor, tokenizer, arms, windows,
                                          cfg, output, budget, device, splits=("dev",), generate=False)
        gate = exploratory_gate(mapping, whole, training, cfg)
        write_json(output / "dev_decision.json", gate)
        mapping_test = mapping_evaluation(torch, arms, records, cfg, output, budget, device, splits=("test",))
        whole_test, generations = whole_model_evaluation(torch, donor, tokenizer, arms, windows,
                                                         cfg, output, budget, device, splits=("test",))
        confirmation = exploratory_gate(mapping_test, whole_test, training, cfg, splits=("test",))
        write_json(output / "test_confirmation.json", confirmation)
        write_json(output / "gate.json", gate)
        if gate["status"] == "EXPLORATORY_POSITIVE":
            import followup
            followups = followup.run(torch, arms["conditional"], records, cfg, output,
                                    budget, device, prefix_features)
        else:
            followups = dict(experiment2="PREPARED_NOT_EXECUTED_NO_POSITIVE_DEV_GATE",
                             experiment3="DEFERRED_NO_POSITIVE_DEV_GATE")
        write_json(output / "followup_status.json", followups)
        dim = donor.config.hidden_size
        program_updates = {str(h): cfg["prefix"] + math.ceil((cfg["context"] - cfg["prefix"]) / h)
                           for h in cfg["horizons"]}
        surrogate_counts = {key: count_parameters(value) for key, value in arms.items()}
        ledger = dict(removed_layers=list(range(cfg["layer_start"], cfg["layer_end"])),
                      removed_parameters=sum(count_parameters(donor.model.layers[i]) for i in
                                             range(cfg["layer_start"], cfg["layer_end"])),
                      surrogate_parameters=surrogate_counts,
                      surrogate_parameter_bytes_fp32={key: 4 * count for key, count in surrogate_counts.items()},
                      dtype="FP32 surrogate, FP16 donor", prefix_input_storage_values=cfg["prefix"] * dim,
                      active_program_values=2 * cfg["rank"], active_program_bytes_fp32=8 * cfg["rank"],
                      program_updates_per_full_prefill=program_updates,
                      shared_map_multiply_adds_per_token=2 * dim * cfg["rank"],
                      generator_multiply_adds_per_update=dim * cfg["program_rank"] +
                      cfg["program_rank"] * 2 * cfg["rank"],
                      generator_amortization_horizons=cfg["horizons"],
                      implementation="Generator executes once per anchor; prefix means use a full-prefix cumsum",
                      execution="Full-prefix recomputation with use_cache=False, no omitted-layer KV",
                      limitation="First32 prefill positions are extrapolated from shorter prefixes without direct supervised loss",
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      unmeasured=["CPU accepted token/s", "DRAM traffic", "quantized kernels", "useful chat quality", "large capacity scaling"])
        write_json(output / "cost_ledger.json", ledger)
        status.update(status="COMPLETE", exploratory_gate=gate["status"])
    except Exception as exc:
        status.update(status="INCOMPLETE", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        raise
    finally:
        status["elapsed_seconds"] = budget.elapsed()
        write_json(output / "status.json", status)
        manifest = {}
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != "artifact_manifest.json":
                manifest[path.name] = dict(bytes=path.stat().st_size, sha256=sha_file(path))
        write_json(output / "artifact_manifest.json", manifest)


if __name__ == "__main__":
    main()
