# PQT experiment preregistration

Status: template; no scientific run admitted.

Before execution, fill and commit every field below. A missing choice prevents
launch; do not substitute an implicit default.

| Required item | Record |
| --- | --- |
| Experiment ID, question and falsifiable hypothesis | Explicit PQT ID and question |
| Scientific code commit and file hashes | Exact immutable source |
| Donor/checkpoints, input manifests and file hashes | Actual source and target identities |
| Calibration/development/holdout splits and prior exposure | IDs, exclusions, overlap checks |
| Baselines and single changed variable | Equal scale/storage/computation accounting |
| Ternary codes, scale grouping, packing and decoder | Explicit final representation |
| Selection rule, progressive schedule and compensation | Fixed algorithm and stopping rule |
| Metrics, per-bank coverage, gates and aggregation | Fixed before observing results |
| Seeds and replication policy | No post-result seed selection |
| CPU/GPU/RAM/disk/wall-time stops | Explicit limits and failure behavior |
| Kaggle account, live quota/reservations and resource ownership | Fresh admission evidence |
| Hardware/software versions, two-GPU strategy | Observed devices and pinned dependencies |
| First invocation and immutable output namespace | Exact command and new output paths |
| Apparatus verification and independent audit | Decode/scale/finite/identity checks |
| Artifact retrieval and hashes | Complete logs/results/checkpoints before release |
| Decision and remaining project requirements | Distinguish measure, inference, hypothesis |

Log failures and interrupted runs as outcomes. A repair gets a new numbered
namespace retaining the first failure. Resume only from a verified checkpoint,
with actual applied optimizer steps, RNG and input position preserved.
