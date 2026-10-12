# Prelaunch coefficient dtype correction

Original frozen producer `0d72234` uses `tensor_reader`, which returns F64.
The controlled F64 arms are appropriate, but the old F32 control must explicitly
cast source coefficients back to F32 before normalization and output projection.
Found through source inspection during binding preparation, before any assay.
Original producer/binding are retained and not launched.

V2 producer makes only explicit `.float()` casts on D/norm/out_proj loading.
Actual BF16/F32 source values are exactly representable, so F64 arm coefficients
are unchanged. Same protocol, operands, tolerances, gates and resource bounds;
new source, binding and output namespace. No relaxed numerical gate.
