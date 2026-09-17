# STRAT-02 — W4 g128 con scale BF16, nuovo braccio v2

**Preregistrazione: 17 settembre 2026, prima di leggere valori dei pesi con
questo formato. Stato: PROPOSED.** Il [pilot W4-v1](../probes/STRAT_02_W4_PILOT_V1_RESULT.md)
e il [censimento parziale](../probes/STRAT_02_W4_FORMAT_CENSUS_PARTIAL_RESULT.md)
mostrano che la scala F16 congelata nel v1 è zero per gruppi reali non nulli
anche fuori dal router. Non si corregge il v1 retroattivamente. La domanda
qui è solo se un esponente più ampio in **due byte di scala** rende
rappresentabili gli organi campionati mantenendo l'apparato verificabile.
Non è un gate BPB, una prova di rate, un export o una conversione all'SSM.

## Formato candidato `strat02_w4_g128_bf16scale_v2`

Stessi tensori lineari, orientamento `[out,in]`, gruppi contigui di 128,
codici signed 4-bit `[-7,7]`, nibble basso prima, `-8` proibito e nessun
padding/outlier/fallback del [W4-v1](../probes/STRAT_02_STAGE0_WEIGHT_FORMAT.md).
La **sola** variabile scientifica è la scala: ogni gruppo ha un valore
IEEE bfloat16 little-endian (2 byte), non binary16. Per `m=max(abs(w))`
calcolato in F32, un gruppo realmente zero ha scala e codici zero. Per un
gruppo non nullo, provare in ordine gli undici `c=0,50…1,00` a passo 0,05:
formare `m*c/7` con le stesse operazioni F32 del v1; convertire il bit
pattern F32 positivo a BF16 con round-to-nearest-even (RNE) sui 16 bit
bassi; ri-espandere a F32. Una scala zero/non finita è **errore di
formato**, mai sostituita con epsilon o con un secondo dtype. Calcolare
`q=clip(rint(w/s),-7,7)` e la MSE F32 della ricostruzione; scegliere la
MSE minima, a parità esatta il `c` maggiore. Nessun dato di calibrazione
o heldout entra nella scelta. Il blob per tensore è l'array BF16
row-major/group-major seguito dai codici packed row-major, come nel v1.
Il decoder deve trattare i 16 bit della scala come la metà alta del bit
pattern F32, poi moltiplicare per i codici. Le lunghezze restano **66
byte/gruppo**; cambiano i byte di scala e potenzialmente i codici.

Le implementazioni scalare e a tile devono essere indipendenti quanto
basta a scoprire errori di layout/rounding. Verificare il cast BF16 RNE
su casi sintetici halfway, carry, zero/subnormal, finiti estremi,
overflow e nonfinite contro `torch.bfloat16` sul runtime CPU pin-nato;
se il confronto indipendente differisce, `VOID_APPARATUS`. I test non
possono modificare il codec v1. Decoder scalar da blob e vettoriale
devono ricostruire F32 bit-identici; `nn.Linear` sui F32 ricostruiti deve
concordare con `X @ W_hat.T`, sulla stessa CPU e con lo stesso input F32,
per ogni elemento entro `atol=1e-5, rtol=1e-6` (non coi pesi originali).

## Pilot reale limitato, senza heldout

Usare esattamente il checkpoint/revisione, gli undici hash e il loader
pin-nati nel [brief v1](BRIEF_STRAT_02_W4_CONVERSION_APPARATUS.md); tutti
gli hash degli shard vengono verificati una sola volta nello stesso run.
Inventario da modello su `meta`, non da regex. Campionare le prime
`min(256,out)` righe complete di cinque matrici, fissate prima dei valori:

1. `model.layers.0.self_attn.k_proj.weight` (attention);
2. `model.layers.0.mlp.gate.weight` (router, inclusa riga shared 127);
3. `model.layers.0.mlp.experts.0.down_proj.weight`;
4. `model.layers.0.mlp.experts.0.gate_proj.weight` (molti underflow v1);
5. `lm_head.weight`.

Il test deve registrare per ciascuna sorgente shape, shard, hash della
slice F32, byte/hash del blob v2 e conteggi di gruppi zero/invalidi;
confrontare i byte tra scalar e tile; decodificare **soltanto dal blob**;
confrontare decode scalar/vettoriale e applicare almeno un input sintetico
deterministico ai `nn.Linear` ricostruiti. Non serve che l'output quantizzato
coincida con l'F32 originale: quello è il successivo gate di fedeltà.
Nessun testo heldout, generazione, download o T4.

Directory risultato nuova/write-once e parziali conservati come
`INCOMPLETE`. Preflight: RAM disponibile ≥8 GiB, spazio libero output
≥2 GiB; supervisor del solo PID worker, campioni ogni 5 s; stop a RAM
disponibile <4 GiB, private commit >8 GiB o wall >30 min inclusi gli
hash. Resource breach è `VOID_RESOURCE`; mismatch di inventario,
rounding, byte o operatore è `VOID_APPARATUS`; scala BF16 non valida su
un gruppo non nullo è `VOID_FORMAT`. Nessun risultato parziale diventa
`PASS`.

Se il pilot completo passa, **non** promuovere direttamente la qualità:
preregistrare un censimento count-only per tutti i 104.398.848 gruppi
senza l'enorme file di coordinate v1, poi l'export completo e il gate
BPB/task paired del brief principale. Se il censimento trova scale
BF16 invalide, arrestare questo formato e progettare una precision map
nuova. Un eventuale pass qualitativo resta solo R1.1: il ledger
W4-all di 661.782.528 byte/token richiede 47,27 GB/s nei 14 ms
ipotetici **per i soli pesi**. Non autorizza il claim 50 tok/s né il
bridge a `engine.c`.
