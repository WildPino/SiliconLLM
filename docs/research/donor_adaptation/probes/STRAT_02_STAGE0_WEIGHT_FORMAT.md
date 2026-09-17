# STRAT-02 Stage 0 — formato weight-only W4/W2 preregistrato

**Data:** 2026-09-16. **Stato:** specifica prima di scaricare pesi o
misurare qualità. Non è una prova di BPB, tok/s, kernel C o compatibilità
con l'SSM nativo. [Brief](../briefs/BRIEF_STRAT_02_STDMOE_PRECISION_GATE.md),
[tokenizer/scoring](STRAT_02_STAGE0_TOKEN_SCORING.md).

## Ambito degli operatori

Ogni `nn.Linear` grande è trattato come matrice F32 `[out, in]`, partizionata
in gruppi contigui di **128 colonne input per riga output**, nell'ordine del
checkpoint. `in % 128 == 0` è obbligatorio; non si aggiungono colonne o
padding implicito. Si quantizzano **solo** i pesi: bias, embedding lookup,
norm, attivazioni, KV, ordine dei layer, attention, router, selezione top-k,
coefficienti 7 routed +1 shared e somma delle uscite non cambiano. Un
checksum dell'inventario dei tensori dovrà confermare che nessuna matrice
lineare interessata viene saltata o duplicata.

Controllo strutturale offline: il modello della revisione pin-nata istanziato
su `meta` con Transformers 4.57.1 espone **6225** moduli `nn.Linear`, tutti
con `in % 128 == 0`: 6144 matrici expert (12884901888 pesi), 64 attention
(268435456), 16 router (4194304) e una head (205520896). Totale lineare
13363052544; i rimanenti 205588480 parametri sono embedding e piccoli
organi. Nessun peso reale è stato caricato da questo controllo. La parità
tra questi moduli e i nomi del safetensors index sarà ricontrollata dal
loader prima dell'uso.

`W4_ALL_LINEAR` usa W4 per attention, router, head e **tutti** gli expert
memorizzati. `MIXED_W4_W2` usa lo stesso W4 per attention/router/head e W2
per **tutti** gli expert memorizzati, inclusi i 127 routed e lo shared per
layer. L'embedding resta F32 in entrambi. Il checkpoint teacher resta F32;
nessun adapter, fine-tuning, QAT, cambio di tokenizer o selezione di un
formato per organo è ammesso in questa cella.

## Codec W4, versione `strat02_w4_g128_v1`

Per ogni gruppo di 128 valori F32:

1. `m = max(abs(w))`, in F32. Se `m==0`, scala fp16 zero e tutti i codici zero.
2. Altrimenti provare undici candidati `c = 0.50, 0.55, ..., 1.00` in ordine
   crescente. Per ciascuno `s = float16(m*c/7)`, poi ri-espanso in F32. Una
   scala non finita o zero dopo l'arrotondamento fp16 è un errore di formato,
   non una correzione silenziosa.
3. `q = clip(round_to_nearest_even(w/s), -7, 7)`. Calcolare la MSE F32 di
   `w - s*q`; scegliere il candidato con MSE minima. A parità esatta
   scegliere il **c maggiore**. Non si usano input di calibrazione o loss
   heldout per selezionare la scala.
4. Salvare una scala IEEE binary16 per gruppo e i codici signed 4-bit in
   complemento a due, con nibble basso prima. Il codice `-8` è riservato e
   deve essere rifiutato dal decoder. Decodifica: `w_hat = s*q`.

Payload 64 byte e metadata 2 byte per gruppo. Tutti gli array di scale sono
row-major/group-major e little-endian. Nel blob di **ciascun tensore** vengono
prima tutti i valori `scale` fp16 contigui `[out, in/128]`, poi tutti i
codici byte contigui `[out, (in/128)*64]`; non sono interleavati per gruppo e
non c'è padding. Non sono ammessi outlier bypass,
zero-point, metadata nascosti o riordino dei gruppi in questa versione.

## Codec W2, versione `strat02_w2_sym4_g128_v1`

Per ogni gruppo di 128 valori F32, il codebook simmetrico a quattro livelli è
`{-a, -b, +b, +a}`, con `a>=b>=0`. Questo è **quattro livelli a due bit**,
non ternario codificato in due bit e non il packed E63. Il fit usa solo i
valori assoluti dei pesi:

1. Inizializzare `b` e `a` con i quantili 0.25 e 0.75 di `abs(w)`, usando
   interpolazione lineare tra le statistiche d'ordine. Eseguire esattamente
   otto iterazioni Lloyd di assegnazione al centro più vicino e media per
   cluster. In un pareggio assegnare al centro piccolo `b`; se un cluster è
   vuoto conservare il suo centro precedente. Ordinare i due centri se
   necessario dopo ogni aggiornamento.
2. Convertire `a,b` a IEEE binary16 e ri-espanderli in F32; un overflow o
   valore non finito è errore. Assegnare una volta i codici finali ai centri
   così memorizzati, non a quelli F32 precedenti. Per `w<0` usare il segno
   negativo; per `w==0` scegliere `+b`. In caso di centri equidistanti
   scegliere il livello di magnitudine piccolo.
3. I codici unsigned sono `0:-a`, `1:-b`, `2:+b`, `3:+a`. Un gruppo tutto
   zero ha `a=b=0`, codici `2`. Impacchettare quattro codici per byte, dal
   meno significativo al più significativo. Decodifica tramite la tabella
   del gruppo senza altre correzioni.

Payload 32 byte e metadata 4 byte (`a,b` fp16) per gruppo, row-major/
group-major e little-endian. Il blob del tensore concatena l'array `a`
fp16 `[out, in/128]`, poi l'array `b` della stessa forma, poi i codici byte
`[out, (in/128)*32]`, senza interleaving né padding. Il codebook è locale al
gruppo e occupa L1/L2
solo se il kernel futuro ne dimostrerà il riuso. La decodifica W2 comporta
costo addizionale da misurare: il quoziente dei byte non predice da solo il
tok/s. Non esiste una garanzia che W2 preservi la qualità; il gate W4 deve
passare prima di autorizzarne la misura heldout.

## Byte ledger prima di qualunque misura

I conteggi derivano dall'inventario tensoriale pin-nato di StdMoE, non da
un download o da un benchmark. Per token: 478150656 pesi sempre attivi
(attention/router/head), 805306368 expert (7 routed +1 shared). Con gruppi
esattamente da 128:

| Braccio | Payload peso/token | Scale/token | Totale codici+scale/token | GB/s per soli pesi in 14 ms |
|---|---:|---:|---:|---:|
| W4-all | 641728512 B | 20054016 B | **661782528 B** | **47.27** |
| Mixed W4/W2 | 440401920 B | 32636928 B | **473038848 B** | **33.79** |

I 32636928 byte di scale mixed sono 7471104 B per always-on W4 e
25165824 B per expert W2 selezionati. Tutti gli expert memorizzati pesano
12884901888 parametri, quindi il solo storage W2 expert vale 3221225472 B
di codici +402653184 B di codebook =3623878656 B. Always-on W4 occupa
239075328 B di codici +7471104 B di scale. I restanti 205588480 parametri
(embedding e piccoli organi non quantizzati) a F32 richiedono 822353920 B:
storage derivato totale **4692779008 B**, prima di header, alignment,
manifest, eventuale riferimento F32 offline e RAM temporanea di loading.

I 33.79 GB/s necessari al mixed sono già maggiori del punto ipotetico
28 GB/s della roadmap, ma vicini al lato packed dell'E63 sintetico per
*byte caricati*; non trasferiamo quell'efficienza a questo decoder, routing
o full attention. W4-all è un gradino qualitativo: 47.27 GB/s richiederebbe
efficienza e banda molto alte **prima** di compute/glue, quindi non è
promosso come candidato deployment a 50 tok/s. A 100 tok/s il mixed avrebbe
bisogno di 67.58 GB/s nei 7 ms streaming ipotizzati, prima dell'overhead:
questa geometria non offre oggi un percorso credibile a 100 senza ridurre
ulteriormente byte attivi o costo compute. Nessuna di queste cifre è un rate
misurato. Il formato finale richiederà un envelope aggiornato con byte
effettivamente mossi, padding e costo del kernel esatto.

## Controlli e stop

Prima dei pesi: selftest sintetici per segno, zero, tie/rounding, saturazione,
codice invalido, ordine dei bit, MSE scelta, scala fp16 e byte count. Dopo i
pesi: quantizzazione streaming, checksum di input/output per tensore,
dequant reference indipendente, confronto numerico dell'operatore con il
runtime scelto, e solo allora scoring. Un mismatch è `VOID_APPARATUS`, non
un risultato di quantizzazione. Non si cambia questo codebook o il gruppo
dopo il heldout; una nuova proposta avrebbe un ID diverso e dati di
calibrazione/heldout ancora protetti.

Il contenitore globale dei tensori non è definito dal codec: un manifest
write-once dovrà associare nome originale, shape, arm, offset/lunghezza e
SHA-256 di ogni blob, mantenendo i blocchi sopra identici. Un cambio del
contenitore che non muta questi byte è lossless ma richiede parity; una
quantizzazione differente non è un semplice cambio di contenitore.

**Diagnostico di costo dell'implementazione di riferimento, non benchmark
scientifico:** su una singola matrice gaussiana sintetica F32 `256×2048`,
seed `20260916`, il codec Python row-by-row ha impiegato 1,163 s per W4
(0,451 Mpesi/s) e 1,291 s per W2 (0,406 Mpesi/s). Nessun warmup, controllo
di occupancy o replica; non sono tempi di quantizzazione del donor, né
timing dell'inferenza C. Segnalano soltanto che il riferimento scalare
potrebbe richiedere ore sul checkpoint intero: prima di convertire 13,6B
pesi serviranno un cap esplicito, salvataggio incrementale e, se necessario,
un encoder vettoriale verificato byte-per-byte contro questo riferimento.

### Diagnostico successivo dell'encoder W4 a tile (17 settembre 2026)

`strat02_weight_codec.py:iter_encode_w4_tiles` aggiunge un percorso a tile
limitati senza modificare il formato W4 congelato. I test sintetici
confrontano **byte identici** con `encode_w4`/`w4_to_blob` su gruppi random,
zeri, valori halfway, outlier, scale da `1e-4` a `1e3`, viste NumPy non
contigue e diversi confini tile/batch; i casi non finiti e scale F16
invalide sono rifiutati. Sul runtime locale pin-nato, NumPy 2.5.3, un
diagnostico di sola conversione sintetica di 4.194.304 pesi F32 ha
richiesto 0,3238442 s, pari a **12,95 Mpesi/s**. Non ci sono repliche o
controllo occupancy: non è un throughput scientifico, né un tempo di
conversione dell'intero donor, né un rate di inferenza. Il prossimo apparato
deve ancora dimostrare scrittura write-once dei tensori, hash, decodifica
indipendente e confronto dell'operatore prima di qualsiasi heldout W4.
