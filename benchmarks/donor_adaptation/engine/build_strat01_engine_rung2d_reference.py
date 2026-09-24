#!/usr/bin/env python3
"""Build the pinned llama.cpp layer-2 depth-extension trace apparatus."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from build_strat01_engine_rung2a_reference import BuildError, build


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--config", default="Release")
    args = parser.parse_args()
    if args.build_dir:
        print(build(args.build_dir, args.config, rung2d=True))
        return 0
    with tempfile.TemporaryDirectory(prefix="strat01-rung2d-reference-") as temporary:
        print(build(Path(temporary), args.config, rung2d=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as error:
        raise SystemExit(f"build_strat01_engine_rung2d_reference: {error}")
