"""Bounded public config/code acquisition; never fetch or execute model weights."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / "metadata"
MAX_FILE = 2 << 20
MAX_TOTAL = 16 << 20
MAX_CALLS = 20
MAX_SECONDS = 300


def main():
    began = time.monotonic()
    relative = Path(__file__).resolve().relative_to(ROOT).as_posix()
    frozen = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
    assert frozen.replace(b"\r\n", b"\n") == Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    OUT.mkdir(exist_ok=False)
    records = []
    total = 0

    def save(name, value):
        with (OUT / name).open("x", encoding="utf8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")

    def fetch(name, url):
        nonlocal total
        assert len(records) < MAX_CALLS and time.monotonic() - began < MAX_SECONDS
        assert url.startswith(("https://huggingface.co/", "https://api.github.com/", "https://raw.githubusercontent.com/"))
        with urlopen(Request(url, headers={"Accept-Encoding": "identity"}), timeout=30) as response:
            assert response.status == 200
            declared = response.headers.get("Content-Length")
            assert declared is None or int(declared) <= MAX_FILE
            assert response.headers.get("Content-Encoding") in (None, "identity")
            body = response.read(MAX_FILE + 1)
            assert len(body) <= MAX_FILE
            total += len(body)
            assert total <= MAX_TOTAL
            record = {"name": name, "url": url, "bytes": len(body),
                      "sha256": hashlib.sha256(body).hexdigest(), "etag": response.headers.get("ETag")}
        with (OUT / name).open("xb") as stream:
            stream.write(body)
        records.append(record)
        return body

    save("owned_process.json", {"pid": os.getpid(), "start_wall_time": time.time(),
                               "git_head": head, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    try:
        revisions = {}
        for slug, repo, files in (
            ("longcat_lite", "meituan-longcat/LongCat-Flash-Lite", ("config.json", "modeling_longcat_ngram.py", "model.safetensors.index.json", "README.md")),
            ("longcat_sparse", "meituan-longcat/LongCat-Flash-Lite-Sparse", ("config.json",)),
            ("switch_c", "google/switch-c-2048", ("config.json",)),
            ("switch_base256", "google/switch-base-256", ("config.json",)),
        ):
            meta = json.loads(fetch(slug + "_api.json", "https://huggingface.co/api/models/" + repo))
            revision = meta["sha"]
            assert re.fullmatch("[0-9a-f]{40}", revision)
            revisions[repo] = revision
            for filename in files:
                fetch(slug + "_" + filename, "https://huggingface.co/" + repo + "/resolve/" + revision + "/" + filename)
        ref = json.loads(fetch("transformers_tag.json", "https://api.github.com/repos/huggingface/transformers/git/ref/tags/v4.57.6"))["object"]
        if ref["type"] == "tag":
            ref = json.loads(fetch("transformers_tag_object.json", ref["url"]))["object"]
        assert ref["type"] == "commit" and re.fullmatch("[0-9a-f]{40}", ref["sha"])
        revisions["huggingface/transformers@v4.57.6"] = ref["sha"]
        fetch("modeling_longcat_flash.py", "https://raw.githubusercontent.com/huggingface/transformers/" + ref["sha"] + "/src/transformers/models/longcat_flash/modeling_longcat_flash.py")
        save("retrieval.json", {"status": "PASS_METADATA_ONLY", "git_head": head,
                                "revisions": revisions, "records": records,
                                "total_body_bytes": total, "calls": len(records),
                                "seconds": time.monotonic() - began,
                                "no_tensor_payload": True, "no_remote_code_execution": True})
        save("terminal.json", {"exit_code": 0, "seconds": time.monotonic() - began})
        print(json.dumps({"status": "PASS_METADATA_ONLY", "bytes": total, "calls": len(records)}))
    except BaseException as exc:
        save("terminal.json", {"exit_code": 1, "seconds": time.monotonic() - began,
                               "error": repr(exc), "traceback": traceback.format_exc(),
                               "completed_records": records, "body_bytes": total})
        raise


if __name__ == "__main__":
    main()
