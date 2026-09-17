# STRAT-02 W4 g128 BF16-scale v2 — formato definito su tutto il donor

**17 settembre 2026. Verdetto ristretto `FORMAT_VALID_FULL`.** Il
[brief preregistrato](../briefs/BRIEF_STRAT_02_W4_BF16_FULL_FORMAT_CENSUS.md)
ha fissato in anticipo la scansione count-only, il primo controesempio
come stop e i cap. Il [pilot v2](STRAT_02_W4_BF16_SCALE_V2_PILOT_RESULT.md)
aveva coperto solo cinque slice. Questo risultato **non** è un modello
quantizzato né una misura di qualità, `engine.c` o tok/s.

## Audit del run completo

[Raw write-once](../../../../benchmarks/donor_adaptation/density/results/strat02_w4_bf16_census_20260917_134519/):
11/11 worker sequenziali `SHARD_COMPLETE`, ciascuno con SHA-256 dello
shard assegnato uguale al manifest pin-nato. I record JSONL per-tensore
coprono **6.225/6.225** lineari distinti e **104.398.848/104.398.848**
gruppi g128. Conteggi aggregati: **0** gruppi sorgente realmente zero,
**0** source-nonfinite, **0** scale BF16 underflow, **0** scale
overflow/nonfinite. La contabilità indipendente dei JSONL ha verificato
unicità dei nomi, forma→gruppi e SHA dei 11 file di conteggi: tutti
coerenti col supervisor. I 256 gruppi delle righe shared del router
sono non nulli e validi; `expert.0.down_proj[0,3]` è non nullo e valido.
Il runner ha anche confrontato col codec scalare completo due gruppi
reali non nulli per tensore, **12.450** controlli.

Il file dei conteggi è complessivamente **2.077.440 B** rispetto al cap
16 MiB; non sono state emesse coordinate massive. Dal log: 11 lanci,
11 completamenti e 53 campioni risorse; circa 236,726 s dal primo
all'ultimo evento. Private commit massimo di un worker 6.238.343.168 B
(<8 GiB), RAM fisica disponibile minima 64.391.151.616 B (>4 GiB).
Nessun cap è stato superato. Nessun heldout, forward del donor,
conversione, download o T4. Commit del runner finale `2ae9191`, non
firmato; il codice v1 è rimasto immutato.

SHA-256 dei file di controllo completi:

| File | SHA-256 |
|---|---|
| `census_manifest.json` | `2986a98d5268a495097ad7885d11ec2bcadd8d9ada5bc9f95f42c42798c3e1eb` |
| `supervisor_result.json` | `87c2fe1e0d140e91909717df630419c88d3bdc9517e60c7ec8402b08dba2c4af` |
| `supervisor_log.jsonl` | `672838cc2305e0a6f5b6e64795b62e8746fb86a47b767472829883268c7f0ffe` |

I singoli SHA degli 11 shard e dei relativi file JSONL sono nel
`supervisor_result.json`; i 11 `shard_XX_worker_result.json` conservano
SHA dello shard verificato, conteggi, controlli e fingerprint metadata.

## Primo tentativo `VOID_APPARATUS`, conservato

Il [run precedente](../../../../benchmarks/donor_adaptation/density/results/strat02_w4_bf16_census_20260917_134344/)
si è fermato **prima di leggere pesi** e prima dell'hash del primo shard:
il worker confrontava le *stringhe* del percorso cache `C:` e della
junction risolta `E:` e riportava erroneamente snapshot diverso. Zero
gruppi/tensori completati; non è evidenza negativa sul formato. Il fix
`2ae9191` usa l'identità filesystem `Path.samefile`; test sintetico e
controllo diretto sulla junction reale passati prima di una directory
nuova. SHA del `supervisor_result.json` del run void:
`846ef95495f2a4e97f853b690616486c97e93e6761269abea8a069d4fa312728`.
Non è stato cancellato né sovrascritto.

## Decisione e limite

Il formato W4-v2 con scala BF16 di due byte è **rappresentabile su ogni
gruppo lineare F32** di questa revisione del checkpoint. Il risultato
autorizza a **preregistrare** un export completo e la successiva prova
paired BPB/task, non a chiamarla superata. La scala BF16 ha meno precisione
relativa di F16 e il codec W4 può alterare fortemente gruppi singoli;
la validità delle scale non misura tale danno. Il ledger W4-all resta
661.782.528 B di pesi/token, già oneroso per 50 tok/s prima di
attention, routing, stato e glue. Anche un gate qualitativo positivo
non dimostrerebbe ancora il bridge dall'operatore Transformer MoE al
piccolo SSM nativo in `benchmarks/phase60/engine.c`.
