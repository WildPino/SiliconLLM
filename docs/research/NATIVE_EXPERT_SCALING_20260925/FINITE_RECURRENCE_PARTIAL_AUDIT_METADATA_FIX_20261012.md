# Partial-audit metadata lookup fault retained

Partial audit46036 closed1 after all24 CPU autograd backward checks and V1/V2
array-bit comparisons passed. Its slope comparison attempted r['tangent'],
which does not exist in the deliberately failed-prefix report (no final producer
report was saved). No complete audit/adjudication claim from that attempt.

AuditV2 compares independently derived stored-gradient/direction slope with the
already frozen transpose-check receipt;adds that receipt to hash custody. No
missing producer observation is invented. Same data,autograd/tangent/FD checks,
tolerances/caps;new source/output namespace. Original audit terminal retained.
