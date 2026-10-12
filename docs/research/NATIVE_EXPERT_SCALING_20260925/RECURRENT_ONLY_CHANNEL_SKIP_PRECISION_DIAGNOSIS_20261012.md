# Original precision bridge failure retained

V2 capture closed exit1 at the first case/site0/fixed8/source denominator.
Old F32 metrics reproduce original report exactly (max delta0); F64/BF16 versus
F32/BF16 output RMS .001982562385296028 exceeds original .001 bridge gate.
First fault and terminal are retained. No output arrays were saved by that run;
do not claim complete V2 output audit.

V3 retains identical .001 threshold and records each precision bridge flag,
including false flags, instead of terminating on that numerical observation.
Source reconstruction/old metric/algebra/resource assertions remain unchanged.
The F64 controlled comparison still isolates skip treatment within one precision
regime; it cannot qualify equivalence to the old F32/BF16 operator when bridge
flags fail. All F64 arm outputs and decomposition receive full independent audit.
This is a diagnostic continuation, not repaired precision or a threshold change.
No native/quality/speed admission. Separate first fault custody remains required.
