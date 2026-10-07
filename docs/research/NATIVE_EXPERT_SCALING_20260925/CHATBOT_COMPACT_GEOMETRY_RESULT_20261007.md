# Finite joint SwiGLU geometry: complete necessary budget passes

New [specification/actual Torch block](../../../benchmarks/native_expert_scaling/chatbot_compact_geometry.py)
and [protocol](CHATBOT_JOINT_SWIGLU_TRANSFER_PROTOCOL_20261007.md) frozen
`1e4b2b5c9bbffe3ba33cc76e646b9b3db4fd51c3` BEFORE this new finite arithmetic.
Executor9948a6 exit0,0.3102502s tool wall span. No Torch/scientific packages
imported, values/census/model calls replayed or hardware timing/peak admission.
[Raw ledger](chatbot_compact_geometry_budget_20261007.json) SHA256
`4aad875680fc2fb0ea35e275d395c71a8542ced41ffca6e70fc613bb6dc5cd37`.
Specification SHAa5850df0172d9f9152e8391a3588ed5cfe9fb184d049955a161e28b4786cd9d0;
completed source census SHAa4293870... reused unchanged.

| Complete proposed batch1 decode | Original donor | E16 | E160 |
| --- | ---: | ---: | ---: |
| Matrix products/token |493961216|246935552|246966272|
| Active FFN products |313786368|66060288|66060288|
| Router products |0|700416|731136|
| Full head products |136134656|136134656|136134656|
| Logical coefficient bytes/token |988067328|494017536|494082816|
| Proposed stored tensor payload bytes |988065536|692196608|3070628096|
| Gated channels active/layer |4864|1024|1024|

Both fixed necessary<=3/5 source complete-matrix and logical-coefficient gates
PASS. More functions increase stored payload while keeping active FFN fixed;
child choice still adds30720 matrix products/token. Head/attention remain
complete, context attention43008*context extra, C float32 KV allocation100663296B
at4096. Nonlinear/RMS/scalar-distance/topK/softmax/accumulation/cache/control work
is additional. Matrix coefficients BF16, centroid norms F32. The implemented
router cancels query norm algebraically; the generic extra-work label does not
charge a new computed query norm to its executable score convention.

These are DIMENSION DEDUCTIONS, not executed weights/unique useful capacity,
physical DRAM or accepted throughput. Initial block requires explicit source-row
initialization; its zero constructor is deliberately guarded from evaluation.
Joint shared/leaf G/U/B fitting and BF16 export/native arithmetic are missing.
No automatic50/s inference from half matrix/byte counts is permitted.

Decision: proceed with the [pure original source-capture prerequisite](CHATBOT_SOURCE_CAPTURE_PROTOCOL_20261007.md)
for a bounded finite fitting trial, not another width/precision ladder. The
capture stage is implemented with48 frozen self-authored conversations,32 fit/
16 reserved development, exact original24-layer x/y BF16 and own-history maps.
Freeze runtime/weights/inputs before its first execution; qualify actual exits/
peak through exit and retain first faults. Fitting objectives/region exposure/
finite updates/held-out pilot criteria still need a separate prospective freeze.
The complete pretrained-chatbot conversion/quality+rate goal remains open.
