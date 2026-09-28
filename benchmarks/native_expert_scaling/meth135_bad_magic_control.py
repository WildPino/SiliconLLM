#!/usr/bin/env python3
"""Verify sidecar-magic rejection and restore the exact third-tier bank."""

import argparse
import hashlib
from pathlib import Path
import subprocess


EXPECTED = "692db3dd1a647debb051ef66612263b5e16ef27c73978e9abd3beb6b3229abc9"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    args = ap.parse_args()
    assert sha(args.sidecar) == EXPECTED
    with args.sidecar.open("r+b") as file:
        old = file.read(1)
        assert old == b"M"
        try:
            file.seek(0)
            file.write(b"X")
            file.flush()
            result = subprocess.run(
                [str(args.exe.resolve()), str(args.bank.resolve()),
                 str(args.vectors.resolve()), str(args.sidecar.resolve())],
                capture_output=True, text=True, check=False)
        finally:
            file.seek(0)
            file.write(old)
            file.flush()
    assert sha(args.sidecar) == EXPECTED
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(f"exit_code={result.returncode}\n"
                        + result.stdout + result.stderr
                        + f"restored_sha256={EXPECTED}\n", encoding="utf-8")
    assert result.returncode == 2
    assert "invalid third-tier sidecar header" in result.stderr
    print(args.log.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
