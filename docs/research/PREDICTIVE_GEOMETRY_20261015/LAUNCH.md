# Three mathematical diagnostics launched

15 October 2026. The owner authorized all three new experiments. This is a
dated launch record, not an observation of numerical success or a model admission.
The [protocol](PROTOCOL.md) was frozen before any push.

| Experiment | Account | Private kernel | Observed first status |
| --- | --- | --- | --- |
| Predictive sensitivity geometry | acct1 / wildpino | [predictive-geometry-20261015](https://www.kaggle.com/code/wildpino/predictive-geometry-20261015) | RUNNING |
| Sensitivity persistence | acct2 / giggio253 | [predictive-temporal-20261015](https://www.kaggle.com/code/giggio253/predictive-temporal-20261015) | RUNNING |
| Cross-context frame reuse | acct3 / sirwildpino | [predictive-reuse-20261015](https://www.kaggle.com/code/sirwildpino/predictive-reuse-20261015) | RUNNING |

Each API push accepted version 1. Subsequent authenticated status queries
returned `KernelWorkerStatus.RUNNING` for all three. Queuing was not required at
that observation. These observations do not establish that cloud smoke tests,
source acquisition, finite-difference controls or numerical studies completed.

## Resource boundaries

Each kernel has a 7,200-second platform cap, a lower bootstrap deadline and a
normally 6,000-second numerical cap. Conservative initial allocation is at most
4 T4-hours per account, 12 total, counting both allocated devices even though
GPU 0 performs the work. No sweep or automatic retry is configured. All accounts
had at least four directly verified remaining GPU quota hours before their push.

The three jobs fit inside Kaggle's documented 12-hour CPU/GPU session limit, so
no GitHub cron, workflow modification or checkpoint relay was needed. If the
platform queues a job, the accepted batch remains queued without another push.
The previously discussed 80-hour envelope is not fully allocated.

## Preparation and scope

Seven new standard-library checks passed locally, and the preceding pilot's
nine structural/receipt checks still passed. No local tensor tests, numerical
model work, checkpoint downloads or training were performed. Cloud numerical
smoke tests validate the shared hook/autograd implementation before acquisition.

All three source packages contain the same code, previous-article exclusions,
and protocol, with mode-specific frozen configurations. The cohort excludes all
176 articles from the preceding pilot; fresh partitions are fixed at 64 FIT,
12 DEV and 12 TEST articles. The previous negative result remains sealed.

This campaign measures local derivative geometry on a small donor. It does not
demonstrate compact predictive dynamics, autonomous future equivalence, cheap
routing, preserved large-donor abilities or useful CPU throughput.

## Receipts and continuation

The [launch receipt](launch_receipt.json) retains exact source/config pins,
launch attempts, API responses and dated status observations. The physical
control/output root is
`E:/SiliconLLM_Artifacts/campaigns/predictive_geometry_20261015`.
It is outside the ongoing artifact migration's source/destination packages.

```text
python -B scripts/predictive_geometry_kaggle.py status --mode all
python -B scripts/predictive_geometry_kaggle.py output --mode all
```

Inspect each bootstrap terminal, numerical status, source/data custody and
module report. Geometry additionally requires its actual-source derivative
checks. Rank-deficient temporal frames and finite-probe archive oracles have
explicit limitations; do not turn them into evidence of scalable execution.
The launcher refuses duplicate attempts. Inspect an ambiguous outcome before
any separately recorded repair or successor decision.

## Subsequent execution update

The temporal and reuse numerical studies completed, with bootstrap wall times
169.205 and 155.837 seconds respectively. Their JSON readbacks matched their
cloud manifests. COMPLETE is not a positive scientific conclusion; the temporal
primary held-out rows have insufficient rank-qualified articles, and the reuse
effects still require the prescribed paired-interval interpretation.

The initial geometry job failed an orthogonality control after both source
derivative checks passed, before held-out score sampling. The first fault is
preserved. A [qualified numerical QR repair](REPAIR.md) was launched as a new
[private kernel](https://www.kaggle.com/code/wildpino/predictive-geometry-qr-repair-20261015).
Its cap is reduced by a conservative bound on the original allocation already
used, preserving the twelve-T4-hour wave ceiling. The original source packages,
negative pilot and numerical tolerance are unchanged. Inspect the
[repair launch receipt](repair_launch_receipt.json) and current API status.
