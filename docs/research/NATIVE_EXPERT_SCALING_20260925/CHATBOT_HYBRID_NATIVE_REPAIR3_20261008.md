# Native apparatus repair3: original kernel parameter macro collision

Repair2 completed the bounded compiler process/descendant setup. Compiler failed
because defining top-k macro K=8 BEFORE the verbatim original kernel header
substituted its legitimate `int K` matrix-width arguments in bc_tm/ref_t3.
Move target K definition AFTER header inclusion; original kernel bodies untouched.
No numeric/data/format/gate change. Compiler exit1/diagnostics/process receipt,
worker first_failure and launcher_failure remain. No executable/query/native
model outputs, so first C observation still pending. Packed export is reused.

Repair2 freeze c2459c2acdebc4756fdcd3feeb3ed36f4f79e3f3, binding
30598973c6f35cd402d580a30009d652f7b3d1fd43cb48f1eb5fbbb52561a13f.
Original two earlier process-monitor failures retained. Scientific gates and
the full goal stay unchanged; apparatus compile faults are not model failures.
