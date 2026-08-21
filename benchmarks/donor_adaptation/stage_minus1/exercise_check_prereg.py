#!/usr/bin/env python3
"""exercise_check_prereg.py -- prove the guard FIRES before trusting its nulls.

Project law: "here we fail with the PLAUSIBLE artefact, not the wrong number -- a tool must be shown
to FIRE on a known-positive before its nulls are worth anything." check_prereg.py returning a clean
report is worthless until it has been shown to (a) catch a deleted field, (b) catch an UNPINNED
field, and (c) go silent on a complete file.

Three cases, all written to a scratch copy so the real prereg.yaml is never touched:

  CASE 1  the real file                       -> expect EXIT 1 (fields genuinely still open)
  CASE 2  one field DELETED + one set UNPINNED -> expect EXIT 1 with those two named
  CASE 3  every UNPINNED replaced by a value   -> expect EXIT 0, silent

Writes exercise_output.log next to this script.

USAGE: python exercise_check_prereg.py
"""

import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREG = HERE / "prereg.yaml"
CHECKER = HERE / "check_prereg.py"
LOG = HERE / "exercise_output.log"

DELETE_KEY = "licence"          # under donor:, indent 2 -- a field nobody would notice missing
DELETE_PARENT_HINT = 'licence: "apache-2.0"'
UNPIN_KEY = 'gated: false'      # under donor:
UNPIN_REPLACEMENT = ('gated: "UNPINNED: DELIBERATE NEGATIVE CONTROL injected by '
                     'exercise_check_prereg.py -- if the guard does not name this line, the guard '
                     'is broken and every clean report it has ever produced is void."')


def run(path, label, out):
    proc = subprocess.run([sys.executable, str(CHECKER), str(path)],
                          capture_output=True, text=True)
    out.write("\n" + "#" * 96 + "\n")
    out.write("# %s\n" % label)
    out.write("# command: python check_prereg.py %s\n" % path.name)
    out.write("#" * 96 + "\n")
    out.write(proc.stdout)
    if proc.stderr:
        out.write("[stderr]\n" + proc.stderr)
    out.write("\n>>> EXIT CODE: %d\n" % proc.returncode)
    return proc.returncode, proc.stdout


def main():
    text = PREREG.read_text(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp(prefix="prereg_exercise_"))
    results = []

    with io.open(LOG, "w", encoding="utf-8", newline="\n") as out:
        out.write("exercise_check_prereg.py -- bidirectional exercise of the stage -1 guard\n")
        out.write("source: %s\n" % PREREG)

        # ---- CASE 1: the real file ------------------------------------------------------------
        code1, stdout1 = run(PREREG, "CASE 1 -- THE REAL FILE (expect EXIT 1: fields still open)",
                             out)
        results.append(("CASE 1 real file", code1, 1))

        # ---- CASE 2: one field deleted, one field un-pinned -----------------------------------
        assert DELETE_PARENT_HINT in text, "fixture drift: %r not in prereg.yaml" % DELETE_PARENT_HINT
        assert UNPIN_KEY in text, "fixture drift: %r not in prereg.yaml" % UNPIN_KEY
        broken = "\n".join(
            line for line in text.split("\n") if DELETE_PARENT_HINT not in line
        ).replace(UNPIN_KEY, UNPIN_REPLACEMENT, 1)
        broken_path = tmp / "prereg_broken.yaml"
        broken_path.write_text(broken, encoding="utf-8")
        code2, stdout2 = run(broken_path,
                             "CASE 2 -- 'donor.licence' DELETED, 'donor.gated' set UNPINNED "
                             "(expect EXIT 1 naming BOTH)", out)
        saw_absent = "[ABSENT]   donor.licence" in stdout2
        saw_unpinned = "[UNPINNED] donor.gated" in stdout2
        out.write("\n>>> guard named the DELETED field 'donor.licence' as ABSENT : %s\n"
                  % ("YES" if saw_absent else "NO -- GUARD IS BROKEN"))
        out.write(">>> guard named the UNPINNED field 'donor.gated'            : %s\n"
                  % ("YES" if saw_unpinned else "NO -- GUARD IS BROKEN"))
        results.append(("CASE 2 deleted+unpinned", code2, 1))
        results.append(("CASE 2 named the deleted field", 1 if saw_absent else 0, 1))
        results.append(("CASE 2 named the unpinned field", 1 if saw_unpinned else 0, 1))

        # ---- CASE 3: complete file -- every UNPINNED replaced by a value -----------------------
        # Only single-line 'key: "UNPINNED: ..."' scalars exist in this artefact, by construction.
        filled, n_filled = re.subn(
            r'^(\s*[A-Za-z0-9_]+): "UNPINNED:.*"$',
            r'\1: "FILLED-BY-EXERCISE (positive control: a complete file must produce silence)"',
            text, flags=re.MULTILINE)
        filled_path = tmp / "prereg_complete.yaml"
        filled_path.write_text(filled, encoding="utf-8")
        out.write("\n[fixture] CASE 3 replaced %d UNPINNED scalars with placeholder values\n"
                  % n_filled)
        code3, stdout3 = run(filled_path,
                             "CASE 3 -- EVERY UNPINNED FIELD FILLED (expect EXIT 0, silent)", out)
        results.append(("CASE 3 complete file", code3, 0))

        # ---- verdict --------------------------------------------------------------------------
        out.write("\n" + "=" * 96 + "\n")
        out.write("EXERCISE VERDICT\n")
        out.write("=" * 96 + "\n")
        ok = True
        for label, got, want in results:
            good = (got == want)
            ok = ok and good
            out.write("  %-36s got %d, expected %d   %s\n"
                      % (label, got, want, "PASS" if good else "FAIL"))
        out.write("\n%s\n" % ("ALL CASES PASS -- the guard fires on a deleted field, fires on an "
                              "UNPINNED field, and is silent on a complete file. Its nulls are "
                              "now worth something."
                              if ok else
                              "AT LEAST ONE CASE FAILED -- the guard is not trustworthy."))

    print(LOG.read_text(encoding="utf-8"))
    return 0 if all(g == w for _, g, w in results) else 1


if __name__ == "__main__":
    sys.exit(main())
