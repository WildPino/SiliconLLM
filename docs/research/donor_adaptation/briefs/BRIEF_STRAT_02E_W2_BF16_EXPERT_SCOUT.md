# STRAT-02E — expert W2-BF16 con router preservato: scout di calibrazione

**Preregistrazione del 17 settembre 2026, prima di codificare o misurare
pesi W2 del donor in questa cella.** Questa è una nuova coordinata
diagnostica CPU: il vecchio Stage 2 W2 era subordinato al passaggio di W4,
mentre W4-v2 ha concluso `FAIL_BPB` heldout. Non si riapre quel gate e non
si esegue lo Stage 2 precedente. Nessuna T4, heldout, task, rollout, port C
o misura di token/s è autorizzata da questo brief.

## Domanda e ipotesi

Se riduciamo **solo** i 6.144 expert lineari del donor StdMoE a quattro
livelli/2 bit per peso, quanto peggiora il BPB della stessa calibration
rispetto al W4-v2? Il recupero economico del router F32 visto in STRAT-02D
persiste quando gli expert sono W2? È un test del preciso codebook sotto,
non un giudizio universale su W2, QAT o carving strutturale.

Il W4-all già fallito carica 661.782.528 B/token di codici+scale. Sostituire
gli expert W4 con W2 porta il ledger **teorico** a 473.038.848 B/token;
ripristinare anche il router F32 porta a 487.653.376 B/token. Questo entra
nel budget di streaming ipotetico di 560 MB a 40 GB/s × 14 ms, ma non in
quello di 392 MB a 28 GB/s, e ignora compute, gather, cache, KV, glue e
sampling. Non è un rate misurato né una promessa di 50 tok/s.

## Identità congelata e controlli

- Donor `allenai/StdMoE_1b14b_1T_Preanneal`@
  `d2a4949c9d4ad6cf47fbac131f7e020077332b21`, 11 shard F32 verificati
  dal manifest sorgente esistente. Un solo modello/arena F32 in RAM; al più
  uno shard sorgente safetensors aperto alla volta. Stesso tokenizer,
  operatori, routing top-7+shared, EOS-prefix, scoring FP32/eager e chunk
  head 128 dei run STRAT-02 precedenti.
- Export W4-v2 già verificato in
  `results/strat02_w4_bf16_full_export_20260917_143753`: SHA-256 di
  `artifact_manifest.json`
  `06924c13b5ad2106929ee51c84442bf63b4b47c6528e90b1ea91de891d4a0a5c`,
  `plan_manifest.json`
  `1c1512b3735f0f9a80b638908801dfba7894b781047c767078a8b0cba54c3e42`,
  export `supervisor_result.json`
  `d52e09748d594b26af89b8223dfcd372343e7b344bc9f10199d379e5e4c7faca`.
- I 48 documenti `calib.jsonl` e le 48 righe W4 di controllo restano
  nell'ordine originale; SHA-256 di `calibration_checks.jsonl`
  `ef1d7f42ad30ff5727432bcc4f0c1a5299ff681f52a5a7690f55602f0171a142`.
  Come secondo controllo paired, le 48 righe `ROUTER_F32` di STRAT-02D
  hanno SHA-256
  `f7913767de5c65396d90473cfd7302961c615ec09bcc0eeb65b3ff428a78550b`.
  ID, ordine, categoria, byte e token devono coincidere, senza aprire
  alcun file heldout o i suoi score.

## Formato W2-BF16 g128, nuovo e distinto dal W2-F16 storico

Ogni riga `[out,in]` è divisa in gruppi contigui di 128 input, senza
padding. Per ogni gruppo, il codebook simmetrico è `{-a,-b,+b,+a}` con
`a>=b>=0`. Il fit **F32** replica l'algoritmo W2 g128 congelato in
[STRAT-02 Stage 0](../probes/STRAT_02_STAGE0_WEIGHT_FORMAT.md): quantili
lineari 0,25/0,75 di `abs(w)`, esattamente 8 iterazioni Lloyd, tie al
centro piccolo `b`, cluster vuoto conserva il centro precedente. La sola
variazione dichiarata è il salvataggio di `a,b` come raw IEEE BF16
round-to-nearest-even, anziché F16. Si ri-espandono i centri BF16 in F32
**prima** dell'assegnazione finale dei codici. In caso di parità finale si
sceglie `b`; `w==0` usa il segno positivo. I codici sono `0:-a, 1:-b,
2:+b, 3:+a`, quattro per byte dal bit meno significativo. Gruppo tutto
zero: `a=b=0`, tutti i codici `2`. Centro non finito, `a<b`, o `a=0` per
un gruppo sorgente nonzero è `VOID_FORMAT`, non un silent fallback.

Il blob di ogni tensore è `[a BF16][b BF16][codici]`, tutti row-major,
little-endian, rispettivamente 2+2+32 = 36 B per gruppo. Si contabilizzano
e si verificano sia codici sia metadata; gli expert inattivi non sono
addebitati per token, ma tutti i 128 expert/layer devono essere codificabili
perché ciascuno può essere scelto. Il test è weight-only: embedding, norm,
bias, attention, head, router e attivazioni restano esattamente quelli W4-v2
salvo il router F32 nel secondo braccio. Nessun fit sulle attivazioni o
sulla loss dei documenti.

Prima dei pesi reali: reference scalare indipendente, encoder tiled bounded
e decoder dal **bitstream realmente impacchettato** devono concordare su
gruppi random, zeri, tie, tiny/large, confini tile, endian, bit order e
input invalidi. La velocità dell'encoder tiled va misurata su sintetico:
se la proiezione conservativa del lavoro completo supera il cap di 6 ore,
non lanciare il donor, registrare `NOT_RUN_RESOURCE_PROJECTION`. Niente
secondo modello F32 né output W2 completo persistito: il payload W2
transita in tile limitati, poi si decodifica nella vista F32 già allocata.
I byte del tile e hash del tensore decodificato restano auditabili; questa
è una prova di qualità del formato, **non** del futuro kernel packed C.

## Bracci, ordine e verifiche

1. Caricare l'esatto W4-v2. Verificare che una sentinella W4 riproduca lo
   score write-once entro `1e-5` bit.
2. Sostituire solo i 6.144 expert con W2-BF16, da sorgente F32 pin-nata,
   sempre attraverso pack/decode. `W2_EXPERTS_ROUTER_W4`: score di tutti i
   48 documenti, una volta, nell'ordine congelato.
3. Ripristinare le 16 matrici router F32 dalla stessa sorgente verificata.
   `W2_EXPERTS_ROUTER_F32`: score degli stessi 48 documenti, una volta.
   Poi ripristinare il router W4 bit-per-bit e riscorare la sentinella
   W2 entro `1e-5` bit.
4. Ripristinare tutti gli expert W4 dai byte dell'export e verificarne
   SHA-256 per tensore; riscorare la sentinella W4 entro `1e-5` bit.

Hash di tutti gli 11 shard F32 e dei payload W4 devono essere validi prima
di leggere valori o assegnare score. Il runner non deve consultare punteggi
parziali per cambiare bracci, subset, scala o stopping rule. Registrare 48
righe per braccio senza testo raw, token ID o logits, BPB totale e per
code/prose/technical_general; differenze paired contro W4 e contro
`ROUTER_F32` di STRAT-02D, e il costo in byte/token. I gain tra bracci non
si assumono additivi. Un peggioramento W2 dimostra solo il costo locale di
questo preciso PTQ W2, non l'impossibilità di training o altri formati.

## Stop e adjudication

Output nuovo write-once con manifest di codice/dati/shard/artefatto e hash
dei risultati. Preflight prima dei valori: RAM fisica disponibile ≥55 GiB,
spazio output ≥1 GiB, versioni/operatori pin-nati; stop durante worker se
RAM libera <8 GiB, private commit o working set >70 GiB, oppure wall >6 h.
Supervisor ogni 5 s. Ogni mismatch di identità/parità/rollback è
`VOID_APPARATUS`; centro W2 invalido è `VOID_FORMAT`; limite risorse è
`VOID_RESOURCE`. Righe parziali restano auditabili ma **non** orientano una
decisione. `COMPLETE_DIAGNOSTIC` richiede due bracci 48/48 e tutti i
controlli finali. Nessun BPB calibration promuove direttamente a heldout,
task, T4 o port C: la decisione seguente richiede un altro brief e dati
disgiunti, e va presa rispetto al budget end-to-end del goal.
