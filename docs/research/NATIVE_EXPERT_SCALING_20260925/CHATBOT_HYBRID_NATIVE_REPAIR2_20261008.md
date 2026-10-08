# Native apparatus repair2: hidden compiler console host

Repair1 freeze bd318b4/native binding5311d857b2721863b8239e3d5d8efdea64262882de05bb91865c5f421a1fa739
was stopped before executable/query/inference output because the compiler driver
created Windows `conhost.exe`. Namespace has only original extracted header,
body receipt and empty compile log; no C model calls. Preserve launcher failure
and external first-fault record. Live compiler/native/worker inventory is empty;
unrelated existing system console hosts are not touched.

Use CREATE_NO_WINDOW for compiler/native process and permit only the system
conhost executable resolved to SystemRoot/System32/conhost.exe, with its exact
file bytes bound, if the installed driver creates it. Keep all compiler/parent/
descendant monitoring; no broad whitelist or numeric/data/format/gate change.
Packed export is complete and reused byte-for-byte; no export replay. Existing
native failures were process-apparatus faults, not C numerical observations.
