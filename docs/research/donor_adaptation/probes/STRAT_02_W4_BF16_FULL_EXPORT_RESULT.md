# STRAT-02 W4 g128 BF16-scale v2 — export completo

**17 settembre 2026. Verdetto ristretto `COMPLETE/EXPORT_VERIFIED`.** Il
[brief preregistrato](../briefs/BRIEF_STRAT_02_W4_BF16_FULL_EXPORT.md) autorizzava
solo l'export del formato W4-v2 a scala BF16 del donor fissato, senza forward,
calibrazione, heldout o T4. Il risultato qui riportato è quindi un esito di
formato/export, non un esito di caricamento o di qualità del modello.

## Esito dell'export completo

[Raw write-once](../../../../benchmarks/donor_adaptation/density/results/strat02_w4_bf16_full_export_20260917_143753/):
il supervisor ha terminato `COMPLETE` con verdetto `EXPORT_VERIFIED`, e l'audit
separato del verifier ha riportato `PASS` sullo stesso ledger. Tutti gli 11/11
shard sono stati completati e verificati; il manifest copre i nomi sorgente da
`model-00001-of-00011.safetensors` a `model-00011-of-00011.safetensors`.

| Voce | Risultato |
|---|---:|
| Record/tensori totali | 6.259 |
| `nn.Linear` | 6.225 |
| Altri tensori F32 | 34 |
| Parametri lineari | 13.363.052.544 |
| Parametri non-lineari | 205.588.480 |
| Gruppi g128 | 104.398.848 |
| Byte lineari W4-v2 | 6.890.323.968 |
| Byte altri tensori F32 | 822.353.920 |
| Payload totale | 7.712.677.888 B |
| Scalar checks | 12.450 |

I 12.450 confronti scalari coprono primo e ultimo gruppo di ogni lineare;
non sono un confronto scalare esaustivo di tutti i 104.398.848 gruppi.
Il verifier ha controllato invece layout, scale/codici validi, lunghezze,
offset e hash di **tutti** i record esportati.

L'audit indipendente Windows ha verificato per tutti gli 11 payload la
corrispondenza tra SHA-256 e dimensione dichiarati nel manifest; la somma delle
dimensioni verificata è `7.712.677.888` B. Questa pass documentale non ha
riletto né ricalcolato gli SHA dei payload.

## Apparatus e risorse

Il [run smoke](../../../../benchmarks/donor_adaptation/density/results/strat02_w4_bf16_export_smoke_20260917_143357/)
resta `PARTIAL_APPARATUS`: solo shard 1, 495 record, 491 lineari,
8.146.944 gruppi, 1.359.806.464 B di payload, 982 scalar checks e 33 campioni
di risorsa. Il massimo private commit è stato 6.892.064.768 B e la RAM fisica
disponibile minima 63.264.813.056 B; non è un risultato full.

Per il full export, l'audit riporta 324 campioni di risorsa, massimo private
commit `6.892.445.696 B`, RAM disponibile minima `60.487.970.816 B` e massimo
worker wall `160.484 s`. Non sono stati superati i cap del brief.

## Controlli riproducibili

La lettura dei JSON/log piccoli ha confermato `schema:
strat02_w4_bf16_full_export_v1`, la revisione sorgente
`d2a4949c9d4ad6cf47fbac131f7e020077332b21`, gli 11 descriptor shard e i conteggi
aggregati sopra. SHA-256 dei controlli del full, verificati contro i file
presenti:

| File | SHA-256 |
|---|---|
| `supervisor_result.json` | `d52e09748d594b26af89b8223dfcd372343e7b344bc9f10199d379e5e4c7faca` |
| `artifact_manifest.json` | `06924c13b5ad2106929ee51c84442bf63b4b47c6528e90b1ea91de891d4a0a5c` |
| `plan_manifest.json` | `1c1512b3735f0f9a80b638908801dfba7894b781047c767078a8b0cba54c3e42` |
| `supervisor_log.jsonl` | `5bee7a2526ddfb060f2c6ca85d0b303fc86ec72f657ee9c20c3867da8eba2472` |

Per confronto, i controlli piccoli del run smoke sono:

| File | SHA-256 |
|---|---|
| `supervisor_result.json` | `7f9e42544ae8615229e75c4de583c59cb547aedddd23933f3dd139221890a034` |
| `artifact_manifest.json` | `7b54fe0774a3853f2991d0b047095ea828d8b012f58bd04ab616d0da6d9a4af7` |
| `plan_manifest.json` | `66000af49b7ea774c92838f48c567d342c20b2385ee64e9e55a47d1cafae5b07` |
| `supervisor_log.jsonl` | `7a0281c38b52c30ae90b355184cf1d0afec422f93685319c8899c75dfa81847e` |

## Limiti e decisione

`EXPORT_VERIFIED` significa che il contenitore W4-v2 e il suo ledger sono stati
scritti e verificati. **Non** sono stati misurati: caricamento del modello dal
nuovo artifact, parity o fedeltà del teacher, BPB/qualità W4, accesso heldout,
`engine.c`, tok/s, rate o qualità nativa SSM. Non è stata eseguita alcuna
conversione/forward in questa documentazione e non è un'autorizzazione a
presentare l'export come pass qualitativo. Il passo successivo resta la prova
operativa di load/parity/calibrazione, seguita solo dalle valutazioni
condizionali preregistrate.
