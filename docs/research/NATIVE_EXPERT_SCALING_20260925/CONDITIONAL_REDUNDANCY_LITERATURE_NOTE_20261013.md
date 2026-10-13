# Two primary upcycling sources checked during the bounded conditional pilot

13 October2026. Focused abstract-level primary-source check, not a systematic
review or independent replication. No article downloaded or executable code run.
Local conditional pilot evidence is separate from these authors' results.

## Expert Upcycling

[Dwivedi et al., Expert Upcycling: Shifting the Compute-Efficient Frontier of
Mixture-of-Experts, arXiv2604.19835v2](https://arxiv.org/abs/2604.19835v2),
submitted21 April2026/revised10 May2026. Authors describe expanding an already
trained MoE by duplicating experts/extending its router while keeping top-K
fixed. Continued pretraining breaks duplicate symmetry. Their abstract reports
7B--13B experiments and reduced GPU-hour training cost versus a fixed-size
baseline. These are authors' claims, not replicated here.

Inference for this project: duplication may supply a useful initialization,
but specialization still requires evidence. This source does not demonstrate
dense chatbot conversion to ternary engine.c functions, physical CPU DRAM
bandwidth, preserved OWN-history quality or our same-artifact50-token/s gate.
Do not equate retained top-K with constant physical CPU latency across n.

## DivMoE

[Lou et al., DivMoE: Fine-Grained MoE Upcycling via Cross-Domain Expert
Composition, arXiv2610.11317v1](https://arxiv.org/abs/2610.11317v1),submitted
8 October2026. Abstract reports a fine-grained single-source slicing/routing
failure, then derives domain-specialized experts through domain-adaptive
continued pretraining and constrains routing to distinct domain groups.
Later continued pretraining and supervised reasoning fine-tuning contribute
to the reported benchmark results. No zero-training conservation theorem
for arbitrary partitioned donor functions is provided by the abstract.

Inference for this project: how information is distributed among functions
and how routes select them must both be measured. Our current tree uses only
actual normalized input features; routing ignores response/domain labels.
The bank is a redundant affine operator assay, not this paper's trained MoE.
Its result can diagnose local representation/routing error, but cannot inherit
these authors' chatbot claims. Full paper methodology remains unreviewed here.
