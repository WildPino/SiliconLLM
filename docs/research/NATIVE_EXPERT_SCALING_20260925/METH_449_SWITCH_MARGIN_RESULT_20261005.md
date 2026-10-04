# METH-449: first serialization stop, no retained margin result

Frozen b5c5586, exact resumption94aa4e5 before first import. One command,
terminal chunk7f22dc, exit1; observed tool wall time9.3187747s. Admission
reported8.25s/8,893,394,459B hashed. No active process/session or rerun.

After the row-I4 arm loop, printing its summary raised
TypeError: Object of type int64 is not JSON serializable (controller line158).
The failure writer then raised the SAME TypeError at json.dumps (line31,
called from line176), so no controller-generated raw/failure JSON was saved.
The source boolean produced by a NumPy sqrt comparison propagates through
sum into a NumPy integer; stdlib JSON cannot encode that scalar directly.

[First failure retention](meth449_switch_margin_result.failure.json) is an
explicitly EXTERNAL terminal/metadata record, SHA256
c0549100c0b7b1ed1404f130bdec4dcb65f5b0891e1859f317bac32ed5dcb1e8.
It records terminal error/stack/admission, fresh frozen scientific hashes and
the empty created output directory. This is NOT a reconstructed scientific
margin result: the computed arm values were not retained and are unavailable.
No complete diagnostic, paired-classification, new quality or source-model claim.

Keep controller/math/protocol unchanged. NEW450 will correct ONLY native scalar
serialization of NumPy scalar leaves, retain SAME formulas/data/resources,
bind this first failure and frozen449 sources, and use NEW numbered paths.
Freeze before import. Do not rerun449 or reinterpret its first stop as successful.
447/448 predictive failures remain immutable. Full goal ACTIVE / INCOMPLETE.
