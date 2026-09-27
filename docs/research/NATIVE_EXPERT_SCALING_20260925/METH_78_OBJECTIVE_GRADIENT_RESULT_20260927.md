# METH-78: balance dominates local product-key gradients

The [frozen protocol](METH_78_OBJECTIVE_GRADIENT_PROTOCOL_20260927.md)
was committed at `9daa1e2` before the measurement. The exact METH-71
update-64 E128 and E1280 checkpoints were evaluated on one bound raw
training window and one bound chat training example per arm, with no
optimizer step or quality-set query. The [E128 JSON](meth78_e128_gradients.json)
and [E1280 JSON](meth78_e1280_gradients.json) contain all loss values,
per-group gradient norms, sample identities and resource measurements.

| Arm and example | CE router norm | weighted KL router norm | 0.02×balance router norm | balance / CE | balance / KL |
|---|---:|---:|---:|---:|---:|
| E128 raw | 0.00945 | 0.00681 | 3.37590 | 357× | 496× |
| E128 chat | 0.03091 | 0.01776 | 1.92374 | 62× | 108× |
| E1280 raw | 0.00985 | 0.00772 | 4.90817 | 498× | 636× |
| E1280 chat | 0.03068 | 0.01907 | 2.77625 | 90× | 146× |

The balance term's B-factor gradient is only 0.03–0.04× the CE B-factor
gradient on raw inputs and under 0.01× on chat inputs. Thus the measured
imbalance is concentrated in the product-key router. Since METH-71/75
globally clip the combined gradient to norm 1.0, the balance component
can materially determine the clipping scale. These are local component
norms before vector addition and do not prove the direction of the
actual joint update or explain the donor-top-1 decline causally.

Two apparatus attempts stopped before obtaining gradients because the
donor helper leaves global autograd disabled; the runner now explicitly
enables it, as METH-71 does. The completed E128/E1280 reads took about
19/25 seconds and used training-source inputs only. The next controlled
test should change the balance rule at a frozen checkpoint, preserve the
same optimizer/data stream, and use fresh train-disjoint development
prompts. METH-72/74/75 failures remain in force; this diagnostic does
not promote either bank.
