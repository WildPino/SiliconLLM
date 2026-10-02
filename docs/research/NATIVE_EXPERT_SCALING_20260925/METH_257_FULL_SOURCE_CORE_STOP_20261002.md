# METH-257: existing effective bank violates the all10-sibling prerequisite

The [frozen export](METH_257_FULL_SOURCE_CORE_PROTOCOL_20261002.md) executes
after commit cd80e83, session16232. All361 source FFN segments pass their
exact hashes;220 unchanged non-FFN/head-proposal fields are assembled.
Existing centered bank reconstruction reaches the first layer but the
requirement that every sibling group contain10 distinct effective BF16 B
matrices fails. No artifact is saved, no component/full-model outputs or
quality targets scored. Stop85.063s after imports, exit1.

[Preserved stop record](meth257_full_source_core_export_result.failure.json)
SHA `fc0fb83cb370a165b89fc3616224a47648077aefa42c13aa766e531329507fcb`.
No completed bank rows/parameter counts were emitted before the guard.
The result establishes at least one duplicate group, not its extent or
cause: this may preexist in raw trained children, merge during FP32
centering, or merge during the BF16 execution cast. Do not attribute it
to quantization until those three stages have been compared.

This closes the fixed all1280-distinct prerequisite for this export. It
does not invalidate the old model's scoped quality measurements, invent
new independent functions, or turn1280 routing labels into1280 genuinely
different functions. Keep the stopped recipe and threshold unchanged.

Next is a frozen read-only METH-258 audit of raw/centered/effective functions
in all24 layers, plus common original-state probes. It must report actual
unique counts and canonicalize exact zero functions, with no training,
perturbation or quality screening. A subsequent changed storage format
can store actual unique functions and alias maps while preserving old
execution; it must distinguish routing labels, parameter signatures and
observed function variation and qualify all saved/core controls anew.
No complete saved candidate/native full quality/accepted rate or arbitrary
RAM/10B/100B/second-family capacity proof is obtained here.
