# STRAT-02 W4 g128 BF16-scale v2 — export completo del donor

**Preregistrato il 17 settembre 2026, prima dell'export completo. Stato:
PROPOSED.** Il [pilot v2](../probes/STRAT_02_W4_BF16_SCALE_V2_PILOT_RESULT.md)
ha verificato cinque slice; il [censimento completo](../probes/STRAT_02_W4_BF16_FULL_FORMAT_CENSUS_RESULT.md)
ha trovato zero gruppi con scale non rappresentabili su tutti i 104.398.848
gruppi. Questo autorizza l'export, **non** prova fedeltà al teacher, qualità,
velocità o compatibilità con lo SSM nativo. Il codec v1 F16 resta chiuso e
immutato. Il [brief di precisione](BRIEF_STRAT_02_STDMOE_PRECISION_GATE.md)
e il [formato v2](BRIEF_STRAT_02_W4_BF16_SCALE_V2_PILOT.md) restano vincolanti.

## Domanda e artefatto

Si può convertire ogni `nn.Linear` della revisione pin-nata
`allenai/StdMoE_1b14b_1T_Preanneal` (`d2a4949c9d4ad6cf47fbac131f7e020077332b21`)
in byte W4-v2 esatti, preservare tutti gli altri tensori F32 byte-identici,
e caricare il modello dal **solo** artefatto convertito? Questo stage non
legge calibrazione o heldout, non esegue forward e non usa T4. Non è una
ricerca di quantizer o una selezione per organo.

L'inventario viene dall'esatta classe pin-nata istanziata su `meta`, con
keyset identico all'indice safetensors: 6.259 tensori, 6.225 lineari
(13.363.052.544 pesi), 34 altri tensori (205.588.480 parametri).
L'export non può usare regex come autorità per decidere quali chiavi
quantizzare. Ogni chiave ha un solo record `source_shard`, `name`, `shape`,
`source_dtype`, `encoding`, `offset`, `length`, `sha256`; nessuna chiave
omessa, ripetuta o extra. Il riferimento F32 ricostruito dovrà leggere
solo questi record, mai il lineare sorgente per correggere errori.

## Contenitore congelato

Un file raw `shard_01.payload` ... `shard_11.payload` per shard sorgente,
nello stesso ordine del manifest pin-nato. Nel file, i tensori dello shard
sono concatenati in **ordine lessicografico del nome**, senza header,
allineamento, padding o compressione aggiuntiva. Il manifest separato
definisce offset e lunghezze; ogni offset è il precedente più la lunghezza,
iniziando da zero. Per un lineare `[out,in]`, il singolo record contiene
prima tutte le scale BF16 little-endian `[out,in/128]`, poi i codici W4
packed `[out,(in/128)*64]` del codec
`strat02_w4_g128_bf16scale_v2`. Lunghezza esatta
`out*(in/128)*66` byte. Per ciascun altro tensore, il record contiene
il payload F32 IEEE-754 little-endian, C-contiguous e shape originale,
lunghezza `numel*4`; nessuna modifica ai bit, inclusi eventuali zeri
signed. I confini dei record non sono un formato di runtime C ancora.

Byte attesi: codici lineari 6.681.526.272, scale BF16 208.797.696,
F32 passthrough 822.353.920, totale payload **7.712.677.888 B**. Il
manifest e i log non contano in questa cifra. Il byte ledger per token
W4-all resta **661.782.528 B**, non una misura di DRAM o tok/s.

## Esecuzione e verifiche

Prima di aprire i valori: verificare hash dei file di protocollo/codec,
runtime pin-nato, config/codice remoto/tokenizer e identità del modello su
`meta`; confrontare forme e keyset con l'indice. Nessun download. Il parent
crea una directory nuova/write-once, scrive piano, revisioni, hash del brief,
codice e sorgenti, ledger atteso e cap. Ogni worker fresco riceve **un solo
shard**, in sequenza: verifica il suo SHA-256 pin-nato una volta prima dei
valori; visita tutte e sole le chiavi indicizzate dello shard. Le slice
F32 lineari sono al più 16 righe per lettura e il tile codec è al più
16 righe × 16 gruppi. Scrivere ogni record nel layout congelato mediante
offset espliciti; non materializzare né scrivere un secondo modello F32
intero. Flush, size e SHA-256 di ogni record e del contenitore dello shard
sono obbligatori. Il worker rilegge il blob, verifica scala/codici/layout e
confronta i byte tile con il riferimento scalare per **due gruppi fissati
per tensore lineare**: primo e ultimo gruppo distinti, oppure uno solo
se coincidono. Nessuna scelta basata sull'errore o sui dati heldout.
Verificare inoltre bit-esattezza dei 34 F32 passthrough.

Selftest sintetici precedono pesi reali: offset/concatenazione, tensor
keyset, scale-then-codes, BF16 RNE, `-8` proibito, truncated/extra bytes,
record SHA alterato, source SHA errato e decoder scalar-vettoriale.
Il verifier indipendente rilegge ogni record del payload dal manifesto,
ricontrolla offset, size, SHA, set di chiavi, organ count, numero di
gruppi/parametri e checksum dei contenitori **senza** consultare i valori
lineari originali. Il completamento richiede 11/11 shard, 6.259/6.259
record e payload totale esatto. Un loader candidato dovrà poi istanziare
il medesimo modello su `meta`, ricostruire tensore per tensore in F32
esclusivamente da questo artefatto, fare `load_state_dict(strict=True,
assign=True)` e verificare nessun tensor `meta` residuo. La parità
dell'operatore e il gate BPB/task sono stage successivi; **non** si usa
il solo checksum come prova di qualità.

Prima della conversione intera è permesso un *apparatus smoke* sul primo
shard, stessi controlli/codec/contenitore, directory separata write-once.
È etichettato `PARTIAL_APPARATUS`, non è un candidato né un sottoinsieme
per BPB. Il run completo parte da una directory **nuova**, non riprende
implicitamente il parziale e rifà la verifica SHA dello shard 1 nel suo
run. Il costo duplicato è dichiarato, non usato per scegliere una variante.

## Cap, stop e decisione

Preflight: RAM fisica disponibile >=8 GiB; output su volume con >=20 GiB
liberi (l'eventuale ricostruzione F32 per scoring è in RAM, non sul disco
di export). Parent monitora solo il PID del worker avviato ogni 5 s;
stop a RAM disponibile <4 GiB, private commit >8 GiB, wall >90 min per
shard o >12 h complessive. Nessun worker concorrente, auto-resume,
eliminazione di parziali o processo estraneo terminato. Cap superato:
`VOID_RESOURCE`; hash/keyset/offset/layout/parità errati:
`VOID_APPARATUS`; gruppo reale non rappresentabile contro il censimento:
`VOID_FORMAT` con coordinate; interruzione generica: `INCOMPLETE`.
Ogni parziale e log resta write-once, senza promozione a `COMPLETE`.

Se l'export e il verifier passano, la prossima autorizzazione è
**teacher repeatability + parità del modello ricostruito + calibration
operativa**, poi una sola misura heldout W4-v2 paired secondo il gate
CI95 superiore di ΔBPB <=+0,02 e i task/rollout condizionali del brief
principale. Non usare il risultato quality per cambiare scale, formato o
allocazione. W2 non è un fallback automatico. Anche un successo W4-v2
rimane un gradino R1 del donor Transformer, non il target su
`benchmarks/phase60/engine.c` né una claim >=50 tok/s.
