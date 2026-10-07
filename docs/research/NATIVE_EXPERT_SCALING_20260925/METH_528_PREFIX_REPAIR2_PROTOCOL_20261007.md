# Prefix528 execution repair2: detach console, unchanged science and budget

7 October2026. Preserve first and repair1 faults, protocols, SHA/exit receipts.
[Diagnostic](METH_528_PREFIX_COMPILER_DIAG_PROTOCOL_20261007.md) freeze
7e0396d2692ec4a901713b5415eba0cff75a0c03, actual chunk279a89 EXIT0.
[Diagnostic raw](meth528_prefix_compiler_diag_result.json) SHA
d6592c623c07178068cd7400f09981395498c16204cbcffc0ca7cd54b350f955:
3.922s,54,927,360B parent,29,536,256B observed compiler-family peak;30s/512MiB.
Its ONE driver plan completed exit0 and produced no object/C controls/responses.
Actual clang21 PID21312/create1791372614.9052856,20,819,968B OS peak.
Observed child is **Windows conhost.exe**, PID19384/create1791372614.9506328,
8,716,288B OS peak, parent21312. Plan explicitly records in-process cc1.
This identifies a hidden console host in these same execution conditions;
first failed attempts' unretained descendant identities remain UNKNOWN.

Execution repair now uses **DETACHED_PROCESS=8**, with stdout/stderr directed to
files and no console/window. Microsoft
[creation flags](https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags)
distinguishes detaching a console from CREATE_NO_WINDOW. Retain all no-descendant
checks, direct cc1/LLD and actual retained-handle OS peaks from repair1. Compiler
PID/create/argv are serialized BEFORE waiting, so a later failure preserves
entered process identity. No hypothetical family peak/admission is granted.

Same [issued scientific inquiry](METH_528_PREFIX_NECESSARY_FAILURE_PROTOCOL_20261007.md),
code/numerical bodies/15 first C response calls/energy controls/proof/thresholds.
No completed scientific operation exists in first/repair1: zero response or
energy calls are replayed. Same main180s/audit120s,CPU10/BLAS1,512MiB parent+
MAX sequential compiler peak,465MiB all prefix/diag new folders/2GiB ALL528 cap.
This is an evidence-driven console execution fix, no budget/selection ladder.
Four compiler subprocesses per DLL (plan/cc1/link-plan/LLD); one DLL per process.

Sources are new `_repair2` namespaces, ALL prior sources untouched. Manifest
additionally binds both prior fault raws/logs and diagnostic raw/plan. Freeze
BEFORE first scientific observations. Commands: pinned Python original flags,
`meth528_prefix_bound_repair2.py --manifest-sha <repair2-SHA> --freeze <commit>`;
after actual main EXIT0/hash, `meth528_prefix_energy_audit_repair2.py` with same
arguments AND `--main-sha <repair2-main-SHA>`. Stop/retain first remaining fault.
Never promote original528 resource/whole-chatbot eligibility.
