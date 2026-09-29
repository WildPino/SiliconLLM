#!/usr/bin/env python3
"""Exact, bounded, crash-safe CPU training snapshots for METH-175."""

import hashlib
import io
import json
from pathlib import Path
import shutil

import torch
import zstandard as zstd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _within(root, path):
    root = root.resolve()
    path = path.resolve()
    assert path.parent == root and path.name.startswith("u")
    return path


def _optimizer_state(optimizer):
    return {"used": optimizer.used, "capacity": optimizer.capacity,
            "locations": optimizer.locations,
            "m": optimizer.m, "v": optimizer.v,
            "steps": optimizer.steps}


def _restore_optimizer(optimizer, state):
    used, capacity = int(state["used"]), int(state["capacity"])
    assert 0 <= used <= capacity <= optimizer.bank.shape[0]
    optimizer.grow(capacity)
    assert optimizer.capacity == capacity
    optimizer.locations.copy_(state["locations"])
    optimizer.used = used
    count = state["m"].shape[0]
    assert count in (used, capacity)
    optimizer.m[:count].copy_(state["m"])
    optimizer.v[:count].copy_(state["v"])
    optimizer.steps[:count].copy_(state["steps"])
    assert int((optimizer.locations >= 0).sum()) == used
    assert int(optimizer.locations.max()) < used


def peek(root, identity):
    root = Path(root)
    latest = root / "latest.json"
    assert latest.is_file(), latest
    pointer = json.loads(latest.read_text(encoding="utf-8"))
    snapshot = _within(root, root / pointer["directory"])
    metadata = snapshot / "metadata.json"
    assert digest(metadata) == pointer["metadata_sha256"]
    state = json.loads(metadata.read_text(encoding="utf-8"))
    assert state["identity"] == identity
    assert state["completed_update"] == pointer["completed_update"]
    assert len(state["layers"]) == 24
    return state, snapshot


def save(root, identity, update, elapsed_seconds, arm_seconds, wrappers, banks,
         optimizers, route_counts, content_counts, structural_counts,
         records, gradient_transfer_total, max_bytes):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    old = None
    old_bytes = 0
    if (root / "latest.json").exists():
        previous, old = peek(root, identity)
        assert previous["completed_update"] < update
        old_bytes = sum(item["bytes"] for item in previous["layers"])
        old_bytes += (old / "metadata.json").stat().st_size
    temporary = _within(root, root / f"u{update:04d}.partial")
    final = _within(root, root / f"u{update:04d}")
    for stale in (temporary, final):
        if stale.exists():
            assert old is None or stale != old
            shutil.rmtree(_within(root, stale))
    temporary.mkdir()
    layers = []
    total = 0
    factorized = identity["arm"] == "candidate"
    for li, (wrapper, bank, optimizer) in enumerate(zip(wrappers, banks, optimizers)):
        state = {"layer": li, "route_counts": route_counts[li],
                 "forward_B_transfer_bytes": wrapper.forward_B_transfer_bytes,
                 "gather_calls": wrapper.gather_calls}
        if factorized:
            state.update({
                "base_bank": wrapper.base_bank,
                "residual_bank": wrapper.residual_bank,
                "base_optimizer": _optimizer_state(optimizer.base_optimizer),
                "residual_optimizer": _optimizer_state(optimizer.residual_optimizer),
                "audit_rng_state": optimizer.audit_rng.bit_generator.state,
                "audit_checks": optimizer.audit_checks,
                "content_counts": content_counts[li],
                "structural_counts": structural_counts[li]})
        else:
            state.update({"bank": bank,
                          "optimizer": _optimizer_state(optimizer)})
        file = temporary / f"layer{li:02d}.pt.zst"
        with file.open("wb") as stream:
            with zstd.ZstdCompressor(level=1).stream_writer(stream) as compressed:
                torch.save(state, compressed)
        size = file.stat().st_size
        total += size
        if total + old_bytes > max_bytes:
            raise RuntimeError(f"METH-175 checkpoint scratch cap: {total + old_bytes} > {max_bytes}")
        layers.append({"file": file.name, "bytes": size, "sha256": digest(file)})
        del state
    metadata = {
        "experiment": "METH-175-exact-CPU-training-checkpoint",
        "identity": identity, "completed_update": update,
        "elapsed_seconds": elapsed_seconds(), "arm_seconds": arm_seconds(),
        "gradient_transfer_total": gradient_transfer_total,
        "records": records, "layers": layers,
        "torch_rng_state": torch.get_rng_state().tolist(),
        "cuda_rng_state": torch.cuda.get_rng_state().tolist()}
    metadata_path = temporary / "metadata.json"
    _json(metadata_path, metadata)
    total += metadata_path.stat().st_size
    if total + old_bytes > max_bytes:
        raise RuntimeError(f"METH-175 checkpoint scratch cap: {total + old_bytes} > {max_bytes}")
    temporary.rename(final)
    pointer = {"directory": final.name, "completed_update": update,
               "metadata_sha256": digest(final / "metadata.json"),
               "snapshot_bytes": total}
    pointer_tmp = root / "latest.json.tmp"
    _json(pointer_tmp, pointer)
    pointer_tmp.replace(root / "latest.json")
    if old is not None:
        shutil.rmtree(_within(root, old))
    return pointer


def restore(root, identity, wrappers, banks, optimizers, route_counts,
            content_counts, structural_counts):
    metadata, snapshot = peek(root, identity)
    factorized = identity["arm"] == "candidate"
    for li, entry in enumerate(metadata["layers"]):
        file = snapshot / entry["file"]
        assert file.stat().st_size == entry["bytes"]
        assert digest(file) == entry["sha256"]
        if file.suffix == ".zst":
            with file.open("rb") as stream:
                with zstd.ZstdDecompressor().stream_reader(stream) as compressed:
                    raw = compressed.read()
            state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            del raw
        else:
            state = torch.load(file, map_location="cpu", weights_only=False)
        assert state["layer"] == li
        wrappers[li].forward_B_transfer_bytes = state.get("forward_B_transfer_bytes", 0)
        wrappers[li].gather_calls = state.get("gather_calls", 0)
        route_counts[li].copy_(state["route_counts"])
        if factorized:
            wrapper = wrappers[li]
            wrapper.base_bank.copy_(state["base_bank"])
            wrapper.residual_bank.copy_(state["residual_bank"])
            residual = wrapper.residual_bank.view(1280, 10, 896, 8)
            combined = wrapper.cpu_bank.view(1280, 10, 896, 8)
            for first in range(0, 1280, 16):
                end = min(first + 16, 1280)
                combined[first:end].copy_(
                    wrapper.base_bank[first:end, None] + residual[first:end])
            _restore_optimizer(optimizers[li].base_optimizer,
                               state["base_optimizer"])
            _restore_optimizer(optimizers[li].residual_optimizer,
                               state["residual_optimizer"])
            optimizers[li].audit_rng.bit_generator.state = state["audit_rng_state"]
            optimizers[li].audit_checks = state.get(
                "audit_checks", metadata["completed_update"])
            content_counts[li].copy_(state["content_counts"])
            structural_counts[li].copy_(state["structural_counts"])
        else:
            banks[li].copy_(state["bank"])
            _restore_optimizer(optimizers[li], state["optimizer"])
        del state
    torch.set_rng_state(torch.tensor(metadata["torch_rng_state"], dtype=torch.uint8))
    torch.cuda.set_rng_state(torch.tensor(metadata["cuda_rng_state"], dtype=torch.uint8))
    assert len(metadata["records"]) == metadata["completed_update"]
    assert metadata["records"][-1]["update"] == metadata["completed_update"]
    return metadata


def cleanup(root):
    root = Path(root)
    if not root.exists():
        return
    root = root.resolve()
    assert root.name.endswith(".checkpoint")
    for item in root.iterdir():
        if item.is_dir():
            shutil.rmtree(_within(root, item))
        else:
            assert item.name in ("latest.json", "latest.json.tmp")
            item.unlink()
    root.rmdir()
