# Preparation V2: valid ABI arguments for the negative-control test

Original numerical freeze319f5de. Preparation29496/create1791972961.48448
failed exit1/20.266s/OS50,249,728 bytes before binding or any timings/trajectories.
The first defect was an access violation in the unbound row-guard test.
rs_block constructs f/bits output views before invoking state_row_step;
the test had passed null output pointer arrays. This is an invalid ABI call,
not evidence that the table values or state-row binding guard were wrong.
Finite table comparisons occur earlier in the exact original control flow;
their assertions had passed, but complete preparation was NOT qualified.

Preserve original adapter, failed receipt, generated source, all4 DLLs/logs
and all3 tables. Inventory/hash record bf_activation_preparation_fault_20261014.json
is included in V2 input pins together with every original family artifact.
No deletion, overwrite or reinterpretation of the original receipt.

Separate native_bf_activation_lut_v2.py and ordered E family
native_bf_activation_lut_v2_20261014. Replace both null-argument negative
tests with the canonical site0 parameter packet and real qualified output
buffers/pointer arrays. The state is still fresh/zero, guard must reject5
and all state/ring/seen remain zero. Original C emitter/reference/consumer/
support, operator substitutions, tables, input data, FP controls and every
exactness/speed/resource criterion unchanged. No timing choice based on data.
Freeze V2 adapter and repair before its preparation.
