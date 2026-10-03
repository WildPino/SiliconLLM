# METH-323: current Switch reference fails capacity and cached model contract

Freeze `16fdde9`, exit0,12.750s/endRSS1,257,435,136bytes. No network/pretrained
weights read; all planned cases executed. Same322 seed/config/source weights.
This completes the interrupted322 diagnostic without making failed gates pass.

Direct3D router returns probability1x4x1, mask1x4x1x2, third output1x4x1.
Mask admits all four chosen tokens instead of first token/expert under capacity1.
F32 selected probabilities and repeated eval remain exact; mask/shape gate FAILS.
Installed keepdim+one_hot creates the extra axis before capacity cumsum.

| Sparse input | Returned finite/matching shape | Error versus logical masked top1 | Norm on logically dropped tokens |
| --- | --- | ---: | ---: |
|1x1x8|PASS|0|0|
|1x4x8|PASS|0.8327527046|0.2191708386|
|2x2x8|PASS|0.8327527046|0.2191708386|

Sparse forward DOES execute, contradicting an inferred forward-crash hypothesis
from static mask rank alone. It flattens tokens, which allows dispatcher rank,
but loses specified per-sequence capacity behavior. Zero-reference faults
detected. No1e-6 capacity-semantic waiver or favorable token subset.

Actual tiny encoder-decoder/head/cache call FAILS: ValueError(use_cache can
only be true if stack is a decoder), raised on the encoder stack. No full
model logits/cache returned. This is a measured reference runtime contract
failure; it does not imply pretrained Switch weights are unusable.

Gates: direct-router/capacity FAIL, sparse shape/finite PASS, full cached model
FAIL, logical capacity semantics FAIL. **Decision:** installed5.13.1 Switch
reference unsuitable before large source acquisition/quality scoring. Existing
project environment remains unchanged. Next qualify isolated official4.57.6
reference with its explicit mask/prob/raw-logit API, unchanged logical capacity
oracle and full tiny cached/prefill controls, then verify321 metadata namespaces.

[Raw complete diagnostic](meth323_switch_runtime_diagnostic_result.json)
SHA `295c2c7855a4f7f8655600545b253a6c6b5e428ed9a7d6f7e0c6ef729ae0e9dc`.
[Official4.57.6 implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/switch_transformers/modeling_switch_transformers.py)
is a candidate reference, not yet installed/validated. No source quality/cost,
accepted>=50 or useful real expert-count scaling established by tiny controls.
