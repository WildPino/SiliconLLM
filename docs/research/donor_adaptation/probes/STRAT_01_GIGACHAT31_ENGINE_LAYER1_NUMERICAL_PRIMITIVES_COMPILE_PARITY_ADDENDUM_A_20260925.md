# STRAT-01 layer-1 numerical-primitives compile-parity — addendum A

**Recorded after the initial scientific attempt and before repair:** 2026-09-25

**Disposition of the initial attempt:**
`VOID_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY`

## Observed apparatus defects

The first scientific invocation completed its one local binary and wrote all
12 registered outputs, but failed the frozen replay/control gates. It makes no
scientific claim and its raw directory must remain immutable.

Source inspection established two apparatus defects:

1. production Rung-2C encloses its scalar F32 matrix loop in
   `#pragma STDC FP_CONTRACT OFF`; the standalone probe omitted that pragma,
   so Clang lowered the nominal scalar replay differently and 373/512 values
   differed from the captured C output;
2. the input, weight and routed-gate mutations moved one value by only one
   ULP. The first two changes were rounded away entirely at the final F32
   output, so both router mutation controls were dead.

These are validation defects. They do not change the frozen estimand,
candidate primitive, independent oracle, immutable references, tensor
descriptor, accepted artifact, or decision thresholds. Descriptive inspection
of the VOID outputs found candidate/oracle and all three candidate/reference
comparisons exact, but those observations cannot be promoted while the frozen
replay/control gates are invalid.

## Narrow repair and authorization

The repair must only:

- apply `#pragma STDC FP_CONTRACT OFF` to the standalone production replay;
- replace each one-ULP mutation with a finite one-value `+1.0f` mutation;
- add static tests binding those changes and preserve all other calculations;
- qualify a new model-free apparatus directory with every execution counter
  at zero.

**REPAIR 1 AUTHORIZED:** only after that repaired apparatus is documented and
the exact sources are committed, one scientific repair invocation may run in a
new directory. It may read the same accepted matrix and immutable payloads but
must execute no producer or model graph. The original raw directory remains
VOID and must never be overwritten or retrospectively adjudicated.
