# Frozen active-depth FIT family: no reduced candidate admitted

15 October 2026. Goal INCOMPLETE. All four stored numerical audits are complete;
the unchanged V2 FIT-only selector exited0. No reduced candidate passes.

| Kept blocks | FIT mean case KL | FIT disagreement | All FIT gates |
| --- | ---: | ---: | --- |
| 24 control | 0.00577542 | 2.79708% | PASS |
| 3 | 15.61598524 | 99.86659% | FAIL |
| 6 | 14.68219940 | 98.94408% | FAIL |
| 12 | 5.44700801 | 81.87035% | FAIL |

These are the selector's producer metrics; independent audit recomputation
agrees within declared numeric tolerance. Source operators at kept sites and
residual width2048 remain unchanged; omitted whole blocks are identities.
This rejects these unadapted schedules. It does not rule out learned merging,
joint adaptation, different representations or useful conditional capacity.
No reduced artifact is promoted, and none has a same-artifact CPU rate pass.

The final FIT12 CPU audit exited0/732.532s/OS1,147,056,128B/no GPU. All56
sequences/672 kept-site sequences/145,944 native positions,298,893,312 BF
words and the same number of integer words replay exactly, including state
and ring. F64 maximum errors: contextual generator8.374853e-5,
B/C1.459882e-7/1.663063e-7, scan9.018208e-7/state1.850093e-6;
fields/norm exact. Actual prefix KV digests are retained. Source GPU
attention/FFN/head remain witnesses, not independently replayed GPU arithmetic.

FIT12 closer18424 exited0/112.0s/OS38,428,672B, sealing all1428 phase files/
4,917,709,527B, original411 source parameters and pinned inputs. Producer,
holder and auditor terminal results match. Original FIT24/FIT3 posthash
failures and recovery/prefix-attestation limitations remain historical facts;
the successful later FIT6/FIT12 closures do not erase them.

Authoritative evidence:

- [FIT12 audit](fixed_depth_fit12_audit_20261014.json), SHA256
  `00b9821e258fa2bf04660340013477509eec83fb42cfa933177ced4a48732f6b`.
- [FIT12 closure](fixed_depth_fit12_session_closure_20261015.json), SHA256
  `6162f86b893768302e6e159b50ab029babce0441278aa250c436f59930dd0008`.
- [FIT-only selection](fixed_depth_fit_selection_20261015.json), SHA256
  `ca64235eb2ef6093243cac423ab2cce21adb73a0d21718542c940d4e775bdf47`,
  selected_depth=null, evaluation_depth=6, nonselected_defect_diagnostic=true,
  DEV_queries=0/task_choice_queries=0. Selector0.562s/OS52,776,960B.

The frozen protocol requires one NONSELECTED six-block defect evaluation on
all24 known DEV histories,16 canonical tasks, own-answer followup and16 prefix
pairs, followed by complete stored audit/closure even on failure. It cannot
be promoted from a DEV pass. RESERVED remains unopened. After this diagnostic,
use the observed defects to motivate source-function adaptation rather than
repeat unchanged identity-drop schedules or small runtime tuning grids.

Owner wait preference updated15 October: pause only for runs expected to take
at least one hour. Shorter jobs continue in this thread with progress updates.
