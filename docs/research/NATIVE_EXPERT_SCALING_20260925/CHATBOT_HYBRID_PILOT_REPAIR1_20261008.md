# Pilot apparatus repair1, before first target observation

Original freeze b7a47e15e7979f6a2a06421da60d202fb3bce99f, binding
f484251254db3c0f31eb9157f6a62c1d414971a6eb1b1cfac9c8147f30c9c280.
Original worker saved ALL6 new source calibration cases/32 generated labels, then
failed in target_config before target construction/basis/forward/training:
`FalconH1Config.layer_types` maps to read-only `layers_block_type`.
Original first_failure and launcher_failure plus complete packets remain intact.

Repair uses the source's derived property after setting num_hidden_layers=12;
the custom target Block explicitly chooses SSM/SWA. No geometry, loss, data,
precision, threshold or budget changes. Source cache is not used by the learner.
Retain source calibration records in the complete result for external reused
score extents and saved-only audit; no source calls repeated.

Repair binding includes ALL original partial-directory bytes; source may be
materialized again solely to initialize missing target weights. Reuse original
six teacher packets, not a new selection or generation. Any completed later
basis/shard/gradient/output observations remain missing-only on further repairs.
