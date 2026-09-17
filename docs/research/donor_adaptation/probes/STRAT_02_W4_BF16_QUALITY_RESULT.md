# STRAT-02 W4-BF16-v2 — risultato quality gate paired

**Eseguito il 17 settembre 2026. Stato finale: `FAIL_BPB`.** Il run ha
completato il protocollo preregistrato sul payload W4-v2 completo: 48/48
documenti di calibration e 96/96 documenti heldout, ciascuno una sola volta
nell'ordine congelato. Il primo tentativo, distinto, resta
`VOID_RESOURCE` prima di qualsiasi forward e non viene fuso con questo
risultato: [tentativo 1](STRAT_02_W4_BF16_QUALITY_ATTEMPT1_VOID_RESOURCE.md).

## Gate e risultato

Il gate preregistrato era `upper_one_sided_ci95_delta_bpb <= +0.02`, con
20.000 draw bootstrap, seed `20260916`, resampling paired per documento
stratificato 32/32/32 e quantile superiore unilaterale al 95%.

| insieme | documenti | token | teacher BPB | candidate BPB | Δ BPB | upper CI95 Δ BPB |
|---|---:|---:|---:|---:|---:|---:|
| heldout totale | 96 | 181.385 | 0,604406515337738 | 0,630431372965746 | +0,026024857628008 | +0,027177891834178 |
| code | 32 | 57.004 | 0,384223718120425 | 0,404914168555279 | +0,020690450434854 | +0,022325908375791 |
| technical_general | 32 | 60.561 | 0,557407126777526 | 0,582411679141243 | +0,025004552363717 | +0,027566276535542 |
| prose | 32 | 63.820 | 0,853600576543356 | 0,885555166698682 | +0,031954590155327 | +0,033702299448568 |

Il limite è superato sia sul totale (+0,027177891834178 > +0,02) sia in
ognuna delle tre categorie. Il verdetto operativo è quindi
`FAIL_BPB`; `task_rollout` è `NOT_RUN_BPB_FAIL`.

## Identità, protocollo e audit

Il candidato è il medesimo export completo W4-v2:

`benchmarks/donor_adaptation/density/results/strat02_w4_bf16_full_export_20260917_143753`

Il run write-once e i suoi log sono in
`benchmarks/donor_adaptation/density/results/strat02_w4_bf16_quality_20260917_162329`.

Il denominatore teacher è quello congelato nel brief: 96 documenti, 746.161
byte e 0,604406515337738 BPB. Il run ha usato il candidate loader only,
chunk da 128, EOS-prefix, nessun troncamento; il processo non ha aperto i
valori F32 sorgente. La calibration è stata completata prima dello heldout
come controllo operativo, senza selezione basata sul suo BPB. Non sono stati
inclusi testo raw, token ID o logit; gli ID documento sono le chiavi di
binding.

Hash SHA-256 registrati dagli artefatti finali:

| oggetto | SHA-256 |
|---|---|
| `artifact_manifest.json` (control) | `06924c13b5ad2106929ee51c84442bf63b4b47c6528e90b1ea91de891d4a0a5c` |
| `plan_manifest.json` (control) | `1c1512b3735f0f9a80b638908801dfba7894b781047c767078a8b0cba54c3e42` |
| export `supervisor_result.json` (control) | `d52e09748d594b26af89b8223dfcd372343e7b344bc9f10199d379e5e4c7faca` |
| teacher `teacher_scores.jsonl` (denominatore) | `96d284da8f223d613039cb9e2a87b24e7b6130701c0873eea0854c3e8290b224` |
| `calibration_checks_sha256` | `ef1d7f42ad30ff5727432bcc4f0c1a5299ff681f52a5a7690f55602f0171a142` |
| `candidate_scores_sha256` | `0ecbac5406b926240ddba3dc9eb25ca5239eb37ba717a5970aee7fb192a8039f` |
| quality `supervisor_result.json` | `b910dec8f07a497448c2c1400eb936513c3ad78c6551d8cc1347225fa2419e3e` |
| quality `worker_result.json` | `ed97f9816e7498d169c747220da3098e4d0dfad14a338707f232693a51d1eec2` |
| quality `supervisor_log.jsonl` | `ce4ba6d0a69dde82a6c93146b7425cb420a30a11caf427cf13a57783d69df514` |
| `worker_stdout_sha256` | `cb5b3864ebe2704859f495c491a7af8037571be104d759dda9c14f3e27d3eff2` |
| `worker_stderr_sha256` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Audit indipendente del run: durata `4283,766 s`, 771 campioni supervisor,
massimo private commit child `52,61 GiB`, massimo working set `51,58 GiB`,
minimo RAM fisica disponibile `12,48 GiB`; tutti entro i limiti congelati.
Hash audit: `worker_result.json`
`ed97f9816e7498d169c747220da3098e4d0dfad14a338707f232693a51d1eec2`,
`supervisor_result.json`
`b910dec8f07a497448c2c1400eb936513c3ad78c6551d8cc1347225fa2419e3e`,
`supervisor_log.jsonl`
`ce4ba6d0a69dde82a6c93146b7425cb420a30a11caf427cf13a57783d69df514`.
Il denominatore teacher e gli hash di controllo dell’artefatto coincidono
indipendentemente con quelli registrati dal worker.

Il risultato del supervisor riporta `ok=false`, exit code del worker `1`,
771 campioni di monitoraggio e durata `4283,766 s`; ciò è coerente con un
failure conclusivo del gate, non con un run incompleto o con un arresto per
risorse.
Il picco di private commit del worker è stato 52,61 GiB, il working set
51,58 GiB e la RAM fisica disponibile minima 12,48 GiB: tutti entro i
limiti di stop preregistrati.

## Cosa è e cosa non è stato misurato

Questo è esclusivamente un gate di fedeltà BPB paired del donor ricostruito
dal formato W4-v2. Non è stato eseguito alcun task o rollout successivo,
né HumanEval; non è una misura di `engine.c`, di un runtime C nativo, di
token/s, di banda o di un traguardo end-to-end a 50 tok/s. Il precedente
`VOID_RESOURCE` resta una classificazione separata e non è un secondo
failure BPB.

## Prossimo passo

Il risultato chiude il gate W4-v2 secondo il brief. La scelta della
continuazione della ricerca resta da decidere; questo documento non
prescrive un nuovo esperimento né autorizza automaticamente il passaggio a
un altro formato o donor.
