# STRAT-02F — W4 con router F32: heldout result

**Eseguito il 18 settembre 2026. Stato: `FAIL_BPB`.** Questo è il follow-on
selezionato della [preregistrazione congelata](../briefs/BRIEF_STRAT_02F_W4_ROUTER_F32_HELDOUT.md)
e segue l'[attempt 1 `VOID_APPARATUS`](STRAT_02F_ATTEMPT1_VOID_APPARATUS.md).
È stato testato un solo candidato: export W4-v2 completo con soltanto le 16
matrici router ripristinate F32 dalla sorgente pin-nata.

## Adjudication

Le 96/96 righe heldout sono state scored una volta, nell'ordine originale,
con ID unici e pairing esatto contro teacher e W4 control. Il corpus è di
746161 byte e 181385 token.

| arm | BPB | Δ vs teacher | upper one-sided CI95 | confronto W4 |
|---|---:|---:|---:|---:|
| teacher | 0.604406515337738 | — | — | — |
| W4 control | 0.630431372965745 | +0.0260248576280077 | — | — |
| W4 + router F32 | 0.623938577445144 | +0.0195320621074062 | +0.0203953142210874 | gain +0.00649279552060154 |

Il gate preregistrato era `upper one-sided CI95 <= +0.02`; l'upper CI è
`+0.0203953142210874`, quindi il candidato è `FAIL_BPB`. Il punto migliora
il W4 control di `0.00649279552060154` BPB, ma non supera il gate paired
contro il teacher.

| categoria | Δ candidato-teacher | upper one-sided CI95 |
|---|---:|---:|
| code | +0.015605334254582751 | +0.016919110595079407 |
| prose | +0.024140060971026913 | +0.025231667316572674 |
| technical_general | +0.018513499745342797 | +0.020541660893876008 |

Il bootstrap è quello congelato: 20.000 draw, seed `20260916`, paired
document-stratified con 32 documenti per categoria e quantile lineare; ordine
di categoria per la riproduzione indipendente: `code`, `technical_general`,
`prose`. La somma indipendente delle righe paired ha riprodotto l'upper CI
`0.020395314221087417`.

## Integrità dell'apparato

Il supervisor ha riportato `ok=false` e child exit code `1` intenzionale per
il gate `FAIL_BPB`: `CHILD_EXITED`, 4406.36 s e 787 campioni. Questo non è un
`VOID_RESOURCE` né un errore di apparato: tutte le 96 righe e le verifiche
scientifiche necessarie al gate sono complete. La RAM fisica minima
disponibile osservata è stata 15.107 GB (circa 14.07 GiB), sopra il cap
di stop a 8 GiB.

Sono stati verificati 11/11 shard sorgente, i 16 SHA dei tensori router,
`other_organs_mutated=false`, `one_arena=true`, sentinel W4 finale e rollback
W4 finale. Nessun testo, token ID o logits è stato riportato; stdout e stderr
sono vuoti.

| oggetto | SHA-256 |
|---|---|
| `candidate_scores.jsonl` / parent candidate | `244a443f7ad1cc5e7afb752ecd2ecc55edd9f5ccedbb9ab2681d0dc2092aaf75` |
| `worker_result.json` | `dd2d2d1fa5a4d8e03cb68c0d3c193961543530f2c5890f5c5b31e5918cc37714` |
| `supervisor_result.json` | `7cde6f2bd3a4852e76262da41ee666b916348e84ce6e2131b75b1a5313941e9c` |
| stdout/stderr vuoti | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Gli hash dei file elencati dal parent coincidono con quelli ricalcolati;
lo SHA del supervisor è stato calcolato separatamente, poiché non può
essere incluso nel proprio risultato.

## Limiti

Questo è un follow-on selezionato su heldout già usato per l'adjudication W4,
non un test finale vergine dell'intero programma. Il risultato riguarda
questa precisa arm W4-v2 + router F32: non dimostra l'impossibilità di un
router F32 in generale, né l'impossibilità di W4 o di altri formati.

Task, rollout, generazione, C, rate e T4 non sono stati eseguiti. Il ledger
attivo teorico è `676397056 B/token` (`661782528 + 14614528`), pari a circa
33.82 GB/s di solo payload a 50 tok/s; anche un BPB pass non avrebbe quindi
implicato una dimostrazione di velocità. Non seguono automaticamente altri
bracci o una promozione.
