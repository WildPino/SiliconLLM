# 491 operator semantics checked before numerical observations

6 October2026. Primary documentation read before code freeze and any491
numerical control/SVD/witness. Local scientific runtime is Python3.12.10,
NumPy2.4.6; its exact installed files,BLAS and _decimal extension are bound
separately. The documentation defines operator semantics,not this runtime's
observed outputs or candidate validity.

- Python [Decimal.exp](https://docs.python.org/3.12/library/decimal.html#decimal.Decimal.exp)
  and [Decimal.ln](https://docs.python.org/3.12/library/decimal.html#decimal.Decimal.ln)
  use correctly rounded half-even results. The next representable values on
  either side provide an enclosing interval. Basic arithmetic uses explicit
  floor/ceiling contexts. Every negation uses copy_negate,which preserves the
  operand without applying the ambient precision. Source floats convert exactly.
  These semantics are documented in the same [Decimal reference](https://docs.python.org/3.12/library/decimal.html).
- [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
  returns full orthonormal bases when full_matrices=True. The constructor uses
  the last left vector of a770x769 feature matrix. Its diagnostic singular
  values do not constitute an independently proved numerical rank.
- [NumPy rint](https://numpy.org/doc/stable/reference/generated/numpy.rint.html)
  rounds halfway values to even. A fixed2^40 scaling/rounding creates integer
  weights; the validator checks quantization with Python round. Exact dyadic
  residual validity holds for any saved integer weights,regardless of SVD quality.

Independent verification uses200digit alternate logit formulas and a different
F32 decoder,float.as_integer_ratio. The main uses100digit probability-first
intervals and bit-field decoding. Both rely on the separately bound Decimal
runtime for correctly rounded exp/ln. Exact residuals require only integers.
No statistical or whole-model evidence follows from these operator sources.
