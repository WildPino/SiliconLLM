"""Offline synthetic checks for the bounded GigaChat MTP HF bundle builder."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmarks.donor_adaptation.density import strat01_gigachat_mtp_hf_bundle as bundle


def _source(tensor: bytes = b"\x01\x02\x03\x04") -> tuple[bytes, int, str]:
    header = json.dumps({"model.norm.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, len(tensor)]}}, separators=(",", ":")).encode()
    prefix = len(header).to_bytes(8, "little") + header
    return prefix + tensor, len(header), hashlib.sha256(prefix).hexdigest()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        return None

    def do_GET(self) -> None:  # noqa: N802
        body = self.server.body  # type: ignore[attr-defined]
        requested = self.headers.get("Range")
        if not requested or not requested.startswith("bytes="):
            self.send_error(400)
            return
        start_text, end_text = requested[6:].split("-", 1)
        start, end = int(start_text), int(end_text)
        end = min(end, len(body) - 1)
        if start > end:
            self.send_error(416)
            return
        data = body[start:end + 1]
        self.send_response(206)
        if self.server.bad_range:  # type: ignore[attr-defined]
            self.send_header("Content-Range", f"bytes 0-0/{len(body)}")
        else:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Server:
    def __init__(self, body: bytes) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.body = body  # type: ignore[attr-defined]
        self.httpd.bad_range = False  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_Server":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.httpd.server_close()

    @property
    def url(self) -> str:
        # The function receives a locally substituted URL only in offline tests.
        return f"https://127.0.0.1:{self.httpd.server_port}/source"


class _HttpsLocalOpener:
    """Map a synthetic HTTPS URL to stdlib HTTP without weakening production URL checks."""

    def __init__(self, server: _Server) -> None:
        self.server = server

    def __call__(self, request: object, timeout: int = 60) -> object:
        from urllib.request import Request, urlopen

        assert isinstance(request, Request)
        rewritten = Request(
            request.full_url.replace("https://127.0.0.1", "http://127.0.0.1"), headers=dict(request.header_items())
        )
        return urlopen(rewritten, timeout=timeout)


class GigaChatMtpHfBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="strat01-mtp-hf-bundle-test-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_pinned_config_normalization_is_numerically_exact_and_fail_closed(self) -> None:
        source = b'{"routed_scaling_factor": 1, "num_hidden_layers": 26}\n'
        local, proof = bundle.normalize_config_for_transformers(source)
        self.assertEqual(local, b'{"routed_scaling_factor": 1.0, "num_hidden_layers": 26}\n')
        self.assertEqual(proof["source_value"], proof["local_value"])
        with self.assertRaises(bundle.BundleError):
            bundle.normalize_config_for_transformers(b'{"routed_scaling_factor": 2}')
        with self.assertRaises(bundle.BundleError):
            bundle.normalize_config_for_transformers(b'{"routed_scaling_factor": 1, "routed_scaling_factor": 1}')

    def test_root_range_and_hash_fail_closed(self) -> None:
        source, header_bytes, header_sha = _source()
        with _Server(source) as server:
            opener = _HttpsLocalOpener(server)
            data, proof = bundle.fetch_root_tensor(
                server.url, source_file_bytes=len(source), source_header_bytes=header_bytes,
                source_header_sha256=header_sha, tensor_name="model.norm.weight", expected_shape=(2,),
                expected_sha256=hashlib.sha256(b"\x01\x02\x03\x04").hexdigest(), opener=opener,
            )
            self.assertEqual(data, b"\x01\x02\x03\x04")
            self.assertEqual(proof["absolute_range"][1] - proof["absolute_range"][0] + 1, 4)
            with self.assertRaisesRegex(bundle.BundleError, "SHA-256 mismatch"):
                bundle.fetch_root_tensor(
                    server.url, source_file_bytes=len(source), source_header_bytes=header_bytes,
                    source_header_sha256=header_sha, tensor_name="model.norm.weight", expected_shape=(2,),
                    expected_sha256="0" * 64, opener=opener,
                )
            server.httpd.bad_range = True  # type: ignore[attr-defined]
            with self.assertRaisesRegex(bundle.BundleError, "Content-Range"):
                bundle.fetch_root_tensor(
                    server.url, source_file_bytes=len(source), source_header_bytes=header_bytes,
                    source_header_sha256=header_sha, tensor_name="model.norm.weight", expected_shape=(2,),
                    expected_sha256=hashlib.sha256(b"\x01\x02\x03\x04").hexdigest(), opener=opener,
                )

    def test_index_has_exact_keys_and_rejects_duplicate_names(self) -> None:
        entries = [(f"mtp.{number}", bundle.SIDECAR_FILE, 2) for number in range(210)]
        entries += [("model.embed_tokens.weight", bundle.ROOT_FILE, 4), ("lm_head.weight", bundle.ROOT_FILE, 4), ("model.norm.weight", bundle.ROOT_FILE, 2)]
        index = bundle.build_index(entries)
        self.assertEqual(len(index["weight_map"]), 213)
        self.assertEqual(index["metadata"]["total_size"], 430)
        with self.assertRaisesRegex(bundle.BundleError, "duplicate"):
            bundle.build_index(entries + [("mtp.0", bundle.SIDECAR_FILE, 2)])

    def test_existing_output_is_never_overwritten(self) -> None:
        output = self.root / "already-there"
        output.mkdir()
        marker = output / "marker"
        marker.write_bytes(b"keep")
        with self.assertRaisesRegex(bundle.BundleError, "refusing to overwrite"):
            bundle._safe_new_directory(output)
        self.assertEqual(marker.read_bytes(), b"keep")

    def test_sidecar_materialization_never_mutates_source(self) -> None:
        source = self.root / "sidecar.safetensors"
        original = b"synthetic-sidecar-content"
        source.write_bytes(original)
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        target_dir = self.root / "target"
        target_dir.mkdir()
        result = bundle._materialize_sidecar(
            source, target_dir / "sidecar.safetensors", before, len(original), verify_sha256=True
        )
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        self.assertEqual((target_dir / "sidecar.safetensors").read_bytes(), original)
        self.assertTrue(result["sha256_verified"])


if __name__ == "__main__":
    unittest.main()
