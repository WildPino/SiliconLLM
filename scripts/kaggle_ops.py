#!/usr/bin/env python3
"""Multi-account Kaggle driver for the phase-64 training runs.

Credentials live OUTSIDE this repo, one directory per account:

    %USERPROFILE%\\.kaggle_accounts\\<account>\\kaggle.json

which is what KAGGLE_CONFIG_DIR expects. Nothing here ever prints a key, and no
secret is ever written under the working tree -- the repo is public.

Subcommands are deliberately thin wrappers over the official CLI: the value this
file adds is (a) switching accounts without clobbering a global ~/.kaggle, and
(b) `whoami`, which is the one check that tells you a token is actually live
before a 12 h job is staked on it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ACCOUNTS_DIR = Path(os.environ.get("KAGGLE_ACCOUNTS_DIR", Path.home() / ".kaggle_accounts"))
ACCOUNTS = ("acct1", "acct2", "acct3")


def config_dir(account: str) -> Path:
    d = ACCOUNTS_DIR / account
    if not (d / "kaggle.json").is_file():
        sys.exit(f"missing credentials: {d / 'kaggle.json'}")
    return d


def kaggle(account: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["KAGGLE_CONFIG_DIR"] = str(config_dir(account))
    # A stray KAGGLE_USERNAME/KAGGLE_KEY in the environment silently outranks the
    # config dir, which would run every arm on one account without ever saying so.
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    cmd = [sys.executable, "-m", "kaggle", *args]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, check=check)


def username(account: str) -> str:
    return json.loads((config_dir(account) / "kaggle.json").read_text())["username"]


def cmd_whoami(args: argparse.Namespace) -> int:
    bad = 0
    for a in args.accounts:
        d = ACCOUNTS_DIR / a
        if not (d / "kaggle.json").is_file():
            print(f"{a:6} MISSING  ({d / 'kaggle.json'})")
            bad += 1
            continue
        u = username(a)
        r = kaggle(a, "kernels", "list", "--mine", "--page-size", "5", check=False)
        if r.returncode != 0:
            print(f"{a:6} {u:24} AUTH-FAIL  {r.stderr.strip().splitlines()[-1:] or ''}")
            bad += 1
        else:
            n = max(0, len(r.stdout.strip().splitlines()) - 2)
            print(f"{a:6} {u:24} ok  ({n} kernel(s) visible)")
    return 1 if bad else 0


def cmd_raw(args: argparse.Namespace) -> int:
    r = kaggle(args.account, *args.rest, check=False)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("whoami", help="verify every account's token is live")
    w.add_argument("--accounts", nargs="*", default=list(ACCOUNTS))
    w.set_defaults(func=cmd_whoami)

    r = sub.add_parser("raw", help="run a raw kaggle CLI command as one account")
    r.add_argument("account")
    r.add_argument("rest", nargs=argparse.REMAINDER)
    r.set_defaults(func=cmd_raw)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
