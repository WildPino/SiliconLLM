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
from typing import Dict, Optional

ACCOUNTS_DIR = Path(os.environ.get("KAGGLE_ACCOUNTS_DIR", Path.home() / ".kaggle_accounts"))
ACCOUNTS = ("acct1", "acct2", "acct3")

# A neutral HOME for every child process. authenticate() consults an OAuth access
# token BEFORE the per-account kaggle.json, and that token is read from sources that
# ignore KAGGLE_CONFIG_DIR entirely: the KAGGLE_API_TOKEN env var and the fixed path
# ~/.kaggle/access_token(.txt). On this machine both hold sirwildpino's token, so every
# account silently authenticated as sirwildpino until these were removed. We point HOME
# at an empty dir (so ~/.kaggle/access_token resolves to nothing) and drop the env var;
# the user's real global config is untouched -- only OUR children see the override.
_NEUTRAL_HOME = ACCOUNTS_DIR / "_neutral_home"


def config_dir(account: str) -> Path:
    d = ACCOUNTS_DIR / account
    if not (d / "kaggle.json").is_file():
        sys.exit(f"missing credentials: {d / 'kaggle.json'}")
    return d


def kaggle(account: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    _NEUTRAL_HOME.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["KAGGLE_CONFIG_DIR"] = str(config_dir(account))
    # Redirect ~ so the access-token FILE cannot be found, and drop every credential
    # source that outranks (or bypasses) KAGGLE_CONFIG_DIR. Order of precedence inside
    # authenticate(): KAGGLE_API_TOKEN env -> ~/.kaggle/access_token file -> legacy
    # KAGGLE_USERNAME/KEY env -> kaggle.json. We want it to fall all the way to the file.
    env["HOME"] = str(_NEUTRAL_HOME)
    env["USERPROFILE"] = str(_NEUTRAL_HOME)  # expanduser() on Windows reads USERPROFILE
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        env.pop(var, None)
    cmd = [sys.executable, "-m", "kaggle", *args]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, check=check)


def username(account: str) -> str:
    return json.loads((config_dir(account) / "kaggle.json").read_text())["username"]


def server_identity(account: str) -> Optional[str]:
    """The account the KEY actually authenticates as, per the server -- which is not
    necessarily the username written in kaggle.json. Derived from the owner prefix of
    the account's own datasets/kernels (--mine is filtered server-side by the caller)."""
    for kind in ("datasets", "kernels"):
        r = kaggle(account, kind, "list", "--mine", "--csv", "--page-size", "1", check=False)
        if r.returncode != 0:
            continue
        for line in r.stdout.strip().splitlines():
            line = line.strip()
            if "/" in line and not line.lower().startswith("ref"):
                return line.split("/", 1)[0]
    return None


def cmd_whoami(args: argparse.Namespace) -> int:
    bad = 0
    seen: Dict[str, str] = {}
    for a in args.accounts:
        d = ACCOUNTS_DIR / a
        if not (d / "kaggle.json").is_file():
            print(f"{a:6} MISSING  ({d / 'kaggle.json'})")
            bad += 1
            continue
        file_user = username(a)
        r = kaggle(a, "kernels", "list", "--mine", "--page-size", "1", check=False)
        if r.returncode != 0:
            print(f"{a:6} file={file_user:14} AUTH-FAIL  {(r.stderr.strip().splitlines()[-1:] or [''])[0]}")
            bad += 1
            continue
        real = server_identity(a) or "(unknown)"
        note = "" if real == file_user else f"  <-- file says '{file_user}', SERVER says '{real}'"
        dup = f"  !! same account as {seen[real]}" if real in seen else ""
        if real != "(unknown)":
            seen[real] = a
        print(f"{a:6} server={real:14} ok{note}{dup}")
        if dup:
            bad += 1
    if len(seen) < len([a for a in args.accounts if (ACCOUNTS_DIR / a / 'kaggle.json').is_file()]):
        print("\nWARNING: fewer distinct accounts than credential files -- parallel arms would collide.")
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
