# STRAT-02 W4 g128 v1 — pilot su pesi reali

**17 settembre 2026. Esito: `VOID_FORMAT` per il braccio W4 v1 congelato;
non è una misura di BPB o una prova contro la quantizzazione W4 in generale.**
Il [brief del pilot](../briefs/BRIEF_STRAT_02_W4_CONVERSION_APPARATUS.md)
e il [formato v1](STRAT_02_STAGE0_WEIGHT_FORMAT.md) precedono questo run.
Nessuna T4, heldout, generazione o forward del donor è stata usata.

## Apparato ed evidenza grezza

Runner del commit `6fc8005`, checkpoint StdMoE pin-nato revisione
`d2a4949c9d4ad6cf47fbac131f7e020077332b21`. Il preflight senza pesi
ha confermato 6.259/6.259 chiavi index/modello, 6.225 lineari e
13.363.052.544 pesi lineari; tassonomia 64 attention, 16 router, 6144
expert e una head. Il worker ha verificato SHA-256 di tutti gli 11 shard
una volta, poi ha letto solo le slice preregistrate in ordine.

Gli [artefatti write-once](../../../../benchmarks/donor_adaptation/density/results/strat02_w4_pilot_20260917_122525/)
registrano `status=INCOMPLETE`, `failure_classification=WORKER_ERROR`,
worker exit code 1 dopo 62,172 s e 13 campioni. La classificazione raw è
un errore Python (`ValueError`); l'adjudication scientifica è
`VOID_FORMAT`, perché il codec rispetta la specifica congelata ma questa
non rappresenta alcuni gruppi non nulli. Massimo private commit osservato
1.659.260.928 B, minimo RAM disponibile 69.204.525.056 B: nessun cap
di risorsa è stato superato. Non si interpreta il tempo come throughput
di inferenza o di conversione completa.

- `attention`: `model.layers.0.self_attn.k_proj.weight`, prime 256 righe
  `[256,2048]`. Blob 270.336 B, SHA-256
  `59cb7ece7da2b81fe88c938518a314640d8c193755db4bd05c62cf57b00d0b86`;
  byte identici al riferimento row-wise, decode scalar/vector F32 esatto,
  parità di `nn.Linear` sintetico da pesi **ricostruiti dal blob**.
- `router`: `model.layers.0.mlp.gate.weight`, prime 128 righe. Il blob
  `router.w4g128.bin` è stato preallocato a 135.168 B ma è **parziale e
  invalido**: non va letto come tensore W4 riuscito. Nessun record router
  `matrix_complete` compare nel log; expert/head non sono stati aperti.

SHA-256 degli artefatti di controllo: `pilot_manifest.json`
`245a57ff828ef92ee5a17c826a7fa469fb9065fe018b761bed5da9392031ddd9`,
`worker_result.json`
`cb5d86a83aa46d9aec4932010cd7df93c79afe7aca3fd7e1a240fee32e51b962`,
`supervisor_result.json`
`f7b53d72498049d0fce0c01c7b961c1dd27645143fb66b41dcdd221feb4de7e3`,
`supervisor_log.jsonl`
`36030f1b42cbabd9fe77c54fc2dc94f54895f3a624d7778b26b7ee763792d6b2`.
I log stdout/stderr worker sono vuoti; il traceback completo è nel
`worker_result.json`.

## Diagnosi limitata, successiva al fallimento

Il W4 v1 calcola per ogni gruppo non nullo `s=float16(max(abs(w))*c/7)`
per undici `c` fra 0,50 e 1,00; se `s==0` il formato impone errore, non
una scala sostitutiva. Una lettura **solo diagnostica** dei 16 router F32
pin-nati, senza score, ha trovato 256 gruppi che underflowano a scala F16
zero per *tutti* gli undici candidati: esattamente i 16 gruppi della riga
127 di ciascuno dei 16 layer, e nessun'altra riga router. Sono gruppi
non nulli: la massima grandezza assoluta della riga 127 è circa
`9,82e-25…1,03e-24`. Sulla riga 127 del layer 0 il minimo `absmax`
per gruppo è `7,10e-25`. Non abbiamo scandito gli altri 6.209 tensori
lineari per questo difetto: l'assenza di ulteriori underflow non è provata.

Nella configurazione pin-nata `num_shared_experts=1` e
`always_active_experts=null`. Il codice sorgente pin-nato del MoE divide
i logits router fra 127 routed e **un** shared, fa softmax separati e
top-k separati: il softmax su un elemento dà coefficiente shared 1.
Questo suggerisce che la riga 127 sia funzionalmente morta per l'output
di inferenza; non basta tuttavia a riutilizzare retroattivamente il
formato v1. I logits diagnostici del router possono differire e la
parità dell'output va verificata con un controllo piantato prima di un
eventuale formato successore. Anche la funzione di auxiliary load-balance
nel sorgente rimuove i logits degli expert shared, ma qui non si è
eseguito training.

**Controllo funzionale successivo, senza donor:** il
[self-test tiny sul codice pin-nato](../../../../benchmarks/donor_adaptation/density/strat02_shared_gate_zero_selftest.py),
seed `20260917`, passa dopo aver azzerato la sola riga shared del gate:
hidden state e logits finali sono bit-identici, routed top-k e pesi sono
identici, coefficiente shared è esattamente 1. I raw router logits
diagnostici **differiscono**, come atteso. Il controllo negativo azzera
invece la routed row 2 e cambia il top-k. Questa è evidenza dell'invarianza
della funzione di inferenza nel caso tiny e della struttura matematica
del ramo con un solo shared; non prova ancora che tutti i 6.225 lineari
del donor siano codificabili dopo la canonicalizzazione.

**Decisione:** fermare l'arm `W4_ALL_LINEAR` v1. Nessun heldout W4,
nessun W2 automatico, nessun rate C. Un possibile successore deve avere
nuovo ID e nuova preregistrazione: o canonicalizzazione esplicita e
dimostrata delle sole 16 righe shared non operative, oppure una precisione
diversa per il router. Prima di scegliere, verificare l'esattezza
funzionale e scandire l'underflow sugli altri lineari; non cambiare il
quantizer v1 in place.
Il [censimento completo preregistrato](../briefs/BRIEF_STRAT_02_W4_FORMAT_CENSUS.md)
è il gate successivo prima di scegliere quel nuovo ID.
