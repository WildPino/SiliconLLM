# H4 — Aggressive-rank step zero

**Stato al 2026-09-16:** `CPU STEP-ZERO COMPLETE / STAGE A NOT RUN`
**Provenienza della preregistrazione:** commit `6c5665b`.

## 1. Oggetto e scope

H4 prepara e misura lo step zero della cella di training nel formato ternario low-rank
q/o a frazione aggressiva `r/D = 1/32` sul donor reale `Qwen/Qwen2.5-1.5B`.
Sono coinvolti tutti i 28 layer, solo `q_proj` e `o_proj`, rank `48`, `D=1536`, per
`8,260,224` master. La revisione del donor è pinned; il training Stage A su T4 **non è
stato eseguito**.

Questo record non riapre E65: E65 misura fattori fp32 redensified post-hoc, mentre H4
misura l'inizializzazione ternaria del formato destinato al training. L'anchor H4 init
è quindi il risultato qui sotto, non il QO-48 fp32 di E65.

## 2. Provenienza e controlli CPU

- Factorization H4 rank 48: G-H4a `FIRES`, 56 organi, `bad_organs=[]`;
  `worst_scale_rel=0` e `worst_prod_rel=0`.
- Artefatto `h4_factors.npz`: `33,468,422` byte, SHA-256
  `7ae0d2ceeff628ff9b076b5df5806def54340f193e6cae5eb8e8013817959371`.
- Self-test CPU FULL: 56 organi, `PASS`; G-H4b/c/d/f/g sono tutti veri.
  G-H4c: `worst_relative=1.0266176531104065e-7`.
- Pack `_h4_bundle` costruito con 15 file hashati e `MANIFEST.json`; verifica
  indipendente dei 15 digest: **0 mismatch**. Dimensione complessiva **97,563,800 byte**
  (manifest incluso). I file CPU di output sono sotto
  `benchmarks/donor_adaptation/s1/results/h4`.

## 3. Controllo dense intact

Controllo CPU fp32, slice frozen `24x512`, 5 prompt E6:

| arm | BPB | free exact | teacher-forced exact | mean rank | median rank | rank ≤5 |
|---|---:|---:|---:|---:|---:|---:|
| dense intact | 0.7675949641196625 | 160/160 | 160/160 | 1 | 1 | 160/160 |

Il risultato passa rispetto all'anchor E65 dense `0.7675949641196624`; la differenza
è `1e-16` nell'ultima cifra riportata.

## 4. H4 INIT ternary rank 48

Valutazione CPU fp32 dello step zero, con gli stessi dati/slice/prompt del protocollo:

| arm | BPB | free exact | teacher-forced exact | mean rank | median rank | rank ≤5 | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| H4 init ternary rank48 | 2.8035765393907672 | 5/160 | 21/160 | 3094.775 | 52.5 | 39/160 | 325.95 s |

L'hash dei fattori init è quello dell'artefatto H4 sopra indicato.

## 5. P1 preregistrata: score

P1 prevedeva che lo step zero ternario fosse peggiore del QO-48 fp32 post-hoc E65
(`2.473430292224694` BPB, `33/160` teacher-forced) e indicava una banda BPB
`3.2–5.5`.

- Predizione qualitativa BPB: **confermata**; H4 init è peggiore di
  `+0.3301462471660732` BPB.
- Predizione qualitativa teacher-forced: **confermata**; `21/160 ≤ 33/160`.
- Predizione numerica BPB `3.2–5.5`: **FALSIFICATA**; `2.8035765393907672 < 3.2`.
- Free `5/160` contro `4/160` di E65 non è un miglioramento generativo interpretabile:
  è vicino al floor e i due percorsi non sono lo stesso oggetto sperimentale.

Il confronto E65 resta descrittivo e non sostituisce l'anchor H4 init.

## 6. Prossimo gate

Anchor init, pack e manifest sono congelati insieme. Il prossimo gate — soltanto con
comunicazione esplicita prima dell'uso della risorsa — è eseguire Stage A
su una sola T4 secondo il brief preregistrato. L'adjudication dovrà usare il checkpoint
terminale e riportare separatamente `TRAINING_HELPS`, `BEATS_QO96`, `BEATS_QO192` e
`GENERATOR_PARTIAL`. Le bande non sono ancora state lette: Stage A non è iniziato.

## 7. Non-promozione

H4 step zero non dimostra training, qualità finale, generazione utile o recupero del rank.
Non autorizza claim su 10B, tok/s, width/scale transfer, export `engine.c`, FFN carve,
router o one-byte. Non autorizza inoltre la riapertura di H2I né modifica lo stato di
`G-E63d`, che resta separatamente owed.
