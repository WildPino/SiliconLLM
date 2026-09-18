"""Offline synthetic HTTP Range tests for the STRAT-01 MTP sidecar extractor."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from benchmarks.donor_adaptation.density import strat01_gigachat_mtp_extract as extract


def _synthetic_source() -> tuple[extract.SourceSpec, bytes, bytes]:
    entries = {
        "model.layers.26.alpha": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
        "model.layers.26.beta": {"dtype": "BF16", "shape": [3], "data_offsets": [4, 10]},
        "model.layers.0.not_mtp": {"dtype": "BF16", "shape": [1], "data_offsets": [10, 12]},
    }
    header = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = bytes(range(12))
    source = len(header).to_bytes(8, "little") + header + payload
    first = 8 + len(header)
    spec = extract.SourceSpec(
        repository="synthetic/repository",
        revision="a" * 40,
        shard="synthetic.safetensors",
        source_file_bytes=len(source),
        source_lfs_sha256="b" * 64,
        source_header_bytes=len(header),
        mtp_key_prefix="model.layers.26.",
        mtp_key_count=2,
        mtp_absolute_first=first,
        mtp_absolute_last=first + 9,
        mtp_payload_bytes=10,
    )
    index = json.dumps({"weight_map": {name: spec.shard for name in entries}}, sort_keys=True).encode("utf-8")
    return spec, source, index


class _RangeHandler(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        return None

    def do_GET(self) -> None:  # noqa: N802 - http.server API spelling
        if self.path.startswith("/source"):
            body = self.server.source  # type: ignore[attr-defined]
            self.server.source_range_requests += 1  # type: ignore[attr-defined]
        elif self.path.startswith("/index"):
            body = self.server.index  # type: ignore[attr-defined]
        else:
            self.send_error(404)
            return
        mode = self.server.mode  # type: ignore[attr-defined]
        range_header = self.headers.get("Range")
        if not range_header or not range_header.startswith("bytes="):
            self.send_error(400)
            return
        start_text, end_text = range_header[6:].split("-", 1)
        start, requested_end = int(start_text), int(end_text)
        end = min(requested_end, len(body) - 1)
        if start < 0 or start > end:
            self.send_error(416)
            return
        selected = body[start:end + 1]
        if mode == "status_200":
            self.send_response(200)
            self.send_header("Content-Length", str(len(selected)))
            self.end_headers()
            self.wfile.write(selected)
            return
        self.send_response(206)
        if mode == "bad_content_range":
            self.send_header("Content-Range", f"bytes 0-0/{len(body)}")
        else:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
        self.send_header("Content-Length", str(len(selected)))
        self.end_headers()
        interrupt = mode == "interrupted" or (
            mode == "interrupted_after_one" and self.server.source_range_requests > 1  # type: ignore[attr-defined]
        )
        if interrupt and self.path.startswith("/source") and len(selected) > 1:
            self.wfile.write(selected[:1])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(selected)


class RangeServer:
    def __init__(self, source: bytes, index: bytes) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        self.httpd.source = source  # type: ignore[attr-defined]
        self.httpd.index = index  # type: ignore[attr-defined]
        self.httpd.mode = "normal"  # type: ignore[attr-defined]
        self.httpd.source_range_requests = 0  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "RangeServer":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.httpd.server_close()

    @property
    def source_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}/source"

    @property
    def index_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}/index"

    @property
    def mode(self) -> str:
        return self.httpd.mode  # type: ignore[attr-defined]

    @mode.setter
    def mode(self, value: str) -> None:
        self.httpd.mode = value  # type: ignore[attr-defined]


class MtpExtractorSyntheticHttpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="strat01-mtp-extract-test-")
        self.root = Path(self.temp.name)
        self.spec, self.source, self.index = _synthetic_source()
        self.server = RangeServer(self.source, self.index)
        self.server.__enter__()

    def tearDown(self) -> None:
        self.server.__exit__()
        self.temp.cleanup()

    def _plan(self) -> extract.ExtractionPlan:
        return extract.fetch_plan(self.spec, index_url=self.server.index_url, source_url=self.server.source_url)

    def test_fresh_range_plan_and_sidecar_self_read_parity(self) -> None:
        plan = self._plan()
        self.assertEqual((len(plan.records), plan.sidecar_payload_bytes), (2, 10))
        self.assertEqual(plan.records[0].source_start, 0)
        self.assertEqual(plan.records[-1].source_end, 10)
        summary = extract.plan_summary(plan)
        self.assertEqual(summary["mode"], "PLAN_ONLY_NO_MTP_PAYLOAD_DOWNLOADED")
        output = self.root / "mtp.safetensors"
        result = extract.execute_plan(plan, output)
        self.assertTrue(output.is_file())
        self.assertTrue(Path(result["manifest"]).is_file())
        self.assertEqual(result["tensors"], 2)
        sidecar = extract._read_sidecar(output, plan)
        self.assertEqual(set(sidecar) - {"__metadata__"}, {"model.layers.26.alpha", "model.layers.26.beta"})
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["sidecar"]["sha256"], extract._sha256_file(output))
        self.assertEqual(manifest["source"]["lfs_sha256_status"], "UNVERIFIED_FULL_SOURCE_NOT_READ")

    def test_alias_candidates_are_not_reported_for_an_incomplete_synthetic_keyset(self) -> None:
        summary = extract.plan_summary(self._plan())
        aliases = summary["candidate_aliases"]
        self.assertEqual(aliases["aliases"], [])
        self.assertEqual(aliases["theoretical_payload_bytes_saved_if_a_future_loader_manifest_explicitly_allows_aliases"], 0)
        self.assertEqual(aliases["current_extractor_behavior"], "COMPLETE_210_TENSOR_SIDECAR; CANDIDATES_ARE_NOT_APPLIED")

    def test_reported_aliases_remain_materialized_and_only_state_theoretical_savings(self) -> None:
        plan = self._plan()
        records = tuple(
            replace(record, name=extract.ALIAS_CANDIDATES[index][0])
            for index, record in enumerate(plan.records)
        )
        summary = extract.plan_summary(replace(plan, records=records))
        aliases = summary["candidate_aliases"]
        self.assertEqual(len(aliases["aliases"]), 2)
        self.assertEqual(
            aliases["theoretical_payload_bytes_saved_if_a_future_loader_manifest_explicitly_allows_aliases"],
            2 * extract.ALIASED_TENSOR_BYTES,
        )
        self.assertTrue(all(item["extraction_behavior"] == "MATERIALIZED_IN_SIDECAR_NO_ALIASING" for item in aliases["aliases"]))

    def test_status_and_content_range_malformed_responses_fail_closed(self) -> None:
        self.server.mode = "status_200"
        with self.assertRaisesRegex(extract.ExtractionError, "must return 206"):
            self._plan()
        self.server.mode = "bad_content_range"
        with self.assertRaisesRegex(extract.ExtractionError, "Content-Range"):
            self._plan()

    def test_interruption_leaves_only_a_verified_resumable_prefix(self) -> None:
        plan = self._plan()
        output = self.root / "interrupted.safetensors"
        self.server.mode = "interrupted"
        with self.assertRaisesRegex(extract.ExtractionError, "truncated source body"):
            extract.execute_plan(plan, output)
        part, resume = extract._part_path(output), extract._resume_path(output)
        self.assertTrue(part.is_file())
        self.assertTrue(resume.is_file())
        state = json.loads(resume.read_text(encoding="utf-8"))
        self.assertEqual(state["downloaded_payload_bytes"], 0)
        self.assertEqual(part.stat().st_size, 8 + len(plan.sidecar_header))
        self.server.mode = "normal"
        extract.execute_plan(plan, output)
        self.assertTrue(output.is_file())
        self.assertFalse(part.exists())
        self.assertFalse(resume.exists())

    def test_resume_authenticates_every_committed_chunk_before_new_network_io(self) -> None:
        plan = self._plan()
        output = self.root / "tampered.safetensors"
        self.server.httpd.source_range_requests = 0  # type: ignore[attr-defined]
        with patch.object(extract, "TRANSFER_CHUNK_BYTES", 4):
            self.server.mode = "interrupted_after_one"
            with self.assertRaisesRegex(extract.ExtractionError, "truncated source body"):
                extract.execute_plan(plan, output)
            part, resume = extract._part_path(output), extract._resume_path(output)
            state = json.loads(resume.read_text(encoding="utf-8"))
            self.assertEqual(state["downloaded_payload_bytes"], 4)
            self.assertEqual(len(state["committed_chunks"]), 1)
            with part.open("r+b") as handle:
                handle.seek(8 + len(plan.sidecar_header))
                handle.write(b"\xff")
            source_requests_before_resume = self.server.httpd.source_range_requests  # type: ignore[attr-defined]
            self.server.mode = "normal"
            with self.assertRaisesRegex(extract.ExtractionError, "committed chunk SHA-256 mismatch"):
                extract.execute_plan(plan, output)
            self.assertEqual(self.server.httpd.source_range_requests, source_requests_before_resume)  # type: ignore[attr-defined]

    def test_symlink_leaf_and_no_resume_symlink_artifacts_fail_closed(self) -> None:
        target = self.root / "protected.bin"
        target.write_bytes(b"protected")
        output = self.root / "leaf.safetensors"
        try:
            output.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        plan = self._plan()
        with self.assertRaisesRegex(extract.ExtractionError, "symlink as sidecar output leaf"):
            extract.execute_plan(plan, output)
        output.unlink()
        part = extract._part_path(output)
        part.symlink_to(target)
        with self.assertRaisesRegex(extract.ExtractionError, "must not be symlinks"):
            extract._start_or_resume(output, plan, no_resume=True)
        self.assertTrue(part.is_symlink())
        self.assertEqual(target.read_bytes(), b"protected")
        part.unlink()
        resume = extract._resume_path(output)
        part.write_bytes(b"untrusted partial")
        resume.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(extract.ExtractionError, "does not match"):
            extract._start_or_resume(output, plan, no_resume=True)
        self.assertTrue(part.exists())
        self.assertTrue(resume.exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        plan = self._plan()
        output = self.root / "existing.safetensors"
        output.write_bytes(b"do not overwrite")
        with self.assertRaisesRegex(extract.ExtractionError, "refusing to overwrite"):
            extract.execute_plan(plan, output)
        self.assertEqual(output.read_bytes(), b"do not overwrite")

    def test_header_keyset_dtype_and_contiguity_fail_closed(self) -> None:
        broken_index = json.loads(self.index.decode("utf-8"))
        del broken_index["weight_map"]["model.layers.26.beta"]
        self.httpd_index = json.dumps(broken_index).encode("utf-8")
        self.server.httpd.index = self.httpd_index  # type: ignore[attr-defined]
        with self.assertRaisesRegex(extract.ExtractionError, "MTP keys"):
            self._plan()
        self.server.httpd.index = self.index  # type: ignore[attr-defined]
        bad_header = {
            "model.layers.26.alpha": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "model.layers.26.beta": {"dtype": "BF16", "shape": [3], "data_offsets": [8, 14]},
        }
        header = json.dumps(bad_header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        bad_source = len(header).to_bytes(8, "little") + header + bytes(14)
        bad_spec = extract.SourceSpec(
            **{**self.spec.__dict__, "source_header_bytes": len(header), "source_file_bytes": len(bad_source),
               "mtp_absolute_first": 8 + len(header), "mtp_absolute_last": 8 + len(header) + 13,
               "mtp_payload_bytes": 14}
        )
        self.server.httpd.source = bad_source  # type: ignore[attr-defined]
        with self.assertRaisesRegex(extract.ExtractionError, "expected BF16"):
            extract.fetch_plan(bad_spec, index_url=self.server.index_url, source_url=self.server.source_url)


if __name__ == "__main__":
    unittest.main()
