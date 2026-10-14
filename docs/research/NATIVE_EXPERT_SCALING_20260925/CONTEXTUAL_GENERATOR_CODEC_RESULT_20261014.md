# Full23 source generator codecs: arithmetic passes, passive cost exceeds budget

14 October2026. Goal ACTIVE/INCOMPLETE. Pipeline33444/session31208 exit0,
201.640s/OS3,349,409,792B. Full export, production, independent audit and
102-file/800,171,408B input/output family closure completed. No GPU/RESERVED.

| Codec | Coefficients + scales bytes | Resident median ms | Evicted median ms | Largest source-field relative error |
|---|---:|---:|---:|---:|
|BF16 control|631570432|33.49805|40.40070|Original archived control exact|
|I8 row-scale|316401984|28.95015|30.90460|1.188455%|
|I4 group128|167760896|22.80625|23.84020|15.283569%|

Three physical workers, same binary/passive-zero-blocktime profile, all23
source matrices retained. New BF control outputs match every prior raw/GG/
field word. Full coefficient audit checks631,570,432 codec coefficients,
all codes/scales/manual nearest-even;5,888 candidate calls/39,473,152 raw
and78,946,304 GG/field words replay exactly,2,944 BF control calls exact.
Decoded-weight F64 maximum relative8.30263e-8<1e-4. Complete384-group order/
timing/accounting passes. Audit122.828s; source-only conversion25.875s.

Native correctness is relative to the specified decoded approximate weights.
It does not mean source outputs are preserved. Both formats miss the historical
1% source-field screen somewhere: I8 worst B1.188455%/dt1.164789%; I4 worst
B15.283569%, with DEV B13.681605%. Those errors use supplied original own-history
h, not an evolving quantized-generator model. No whole-quality verdict or
artifact selection follows. No old threshold is loosened or result reclassified.

Every I8 measured group exceeds20ms; I4 some individual groups are below20ms
(minimum15.4064/15.7884ms), but both medians exceed20ms BEFORE recurrence,
readout, FFN, attention, routing and head. Full-depth quantization is not a
validated50-token solution. Fields, quantizer/decoder overhead and dispatch are
charged; cache eviction is not a physical DRAM counter. Shared-desktop tails
remain, including I4 maxima107.1591/146.9594ms.

Keep the reproducible source-only compiler and exact decoded-weight C kernels.
Before an own-history GPU phase, test one explicit runtime waiting intervention
on the SAME packets with actual generator+readout interleaving and all outputs
checked. M307's separate apparatus motivates this hypothesis but did not itself
qualify stable active waiting. A passed partial budget still requires actual
remaining operators and own-model quality; it is not a recipe admission.

[Binding](contextual_generator_codec_20261014_binding.json),
[report](contextual_generator_codec_20261014_result.json),
[audit](contextual_generator_codec_20261014_audit.json),
[closure](contextual_generator_codec_20261014_closure.json),
[pipeline](contextual_generator_codec_20261014_pipeline.json),
[terminal](contextual_generator_codec_20261014_pipeline.terminal.json).
