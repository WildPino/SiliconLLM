# H4 — Aggressive-rank step zero

**Stato al 2026-09-16:** `CPU STEP-ZERO COMPLETE / STAGE A RUNNING`
**Provenienza della preregistrazione:** commit `6c5665b`.

## 1. Oggetto e scope

H4 prepara e misura lo step zero della cella di training nel formato ternario low-rank
q/o a frazione aggressiva `r/D = 1/32` sul donor reale `Qwen/Qwen2.5-1.5B`.
Sono coinvolti tutti i 28 layer, solo `q_proj` e `o_proj`, rank `48`, `D=1536`, per
`8,260,224` master. La revisione del donor è pinned; prima del launch del 2026-09-16 il
training Stage A su T4 **non era stato eseguito** (stato storico pre-launch).

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
- Pack v1 `_h4_bundle` costruito con 15 file hashati e `MANIFEST.json`; verifica
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

## 6. Stato operativo Stage A

Al controllo del 2026-09-16 alle ~09:21 UTC, l'account `acct3` risultava autenticato come
`sirwildpino`. Il dataset Kaggle privato `sirwildpino/siliconllm-h4-rank48-v2` è stato
creato una sola volta, risultava server `READY`, e il suo inventario flat di 16/16 file e
byte combaciava esattamente con il manifest congelato; SHA-256 del manifest:
`a97e444359684b2bd769932248493873fe7a7f4b4e63f3f811cf3caf17f2c331`. La quota GPU era
`0/30h` usata, verificata via API diretta. Il problema di quota della CLI è un difetto del
parser `kagglesdk` per `Duration` in secondi interi; il launcher ora include la correzione
in memoria e i 16 test passano.

Il kernel `sirwildpino/h4-rank48-stagea-v2` è stato sottomesso una sola volta; `kernels status`
ha restituito `KernelWorkerStatus.RUNNING`. La configurazione Stage A è pinned a 4000 step,
batch size 2, accumulation 8, learning rate `2e-4`, checkpoint ogni 250 step, limite
`max 2.8h`, seed 1717, una T4.

Questo aggiorna soltanto lo stato operativo: non è ancora disponibile alcun checkpoint terminale
né alcuna adjudication CPU fp32, e tutte le bande di outcome H4 restano **non valutate**.

## 7. Prossimo gate

Anchor init e pack v2 sono congelati insieme. Il prossimo gate — soltanto con
comunicazione esplicita prima dell'uso della risorsa — è completare Stage A
su una sola T4 secondo il brief preregistrato. L'adjudication dovrà usare il checkpoint
terminale e riportare separatamente `TRAINING_HELPS`, `BEATS_QO96`, `BEATS_QO192` e
`GENERATOR_PARTIAL`. Le bande non sono ancora state lette: Stage A è in esecuzione.

## 8. Non-promozione

H4 step zero non dimostra training, qualità finale, generazione utile o recupero del rank.
Non autorizza claim su 10B, tok/s, width/scale transfer, export `engine.c`, FFN carve,
router o one-byte. Non autorizza inoltre la riapertura di H2I né modifica lo stato di
`G-E63d`, che resta separatamente owed.

## 9. Correzione dell'apparato prima della T4 — checkpoint periodici

Una verifica del sorgente prima del launch ha trovato una discrepanza con il brief §5.2:
`h4_qat.py` registrava metriche ogni 250 step ma salvava solo il checkpoint terminale.
Prima di qualunque uso T4, il trainer è stato corretto per scrivere una coppia
`h4_trained.stepNNNN.npz/.json` a ogni multiplo di 250 inferiore allo step finale.
Ogni coppia è hashata, vincolata agli input, marcata `INTERIM_NONTERMINAL`, rifiuta
collisioni ed è esclusa dall'adjudication. Salva i pesi, non lo stato Adam: un
eventuale resume riavvierebbe l'ottimizzatore e richiederebbe una decisione separata.
Rimangono invariati seed, dati, fattori,
schedule, soglie, limite di 2.8 ore e selezione del solo checkpoint terminale CPU fp32.

Il self-test completo v2 in `results/h4/h4_selftest_v2.json` passa, compreso il
controllo sintetico `G-H4h` su nome, metadata, hash e rifiuto di sovrascrittura.
Il bundle originale è preservato; `_h4_bundle_v2` contiene 15 file con **0 mismatch**
alla verifica indipendente, `MANIFEST.json` SHA-256
`a97e444359684b2bd769932248493873fe7a7f4b4e63f3f811cf3caf17f2c331`.
Gli hash dei fattori (`7ae0d2ceeff628ff9b076b5df5806def54340f193e6cae5eb8e8013817959371`)
e dell'anchor init (`140a8968ccc49ebe4bdf8518bd6f901b0b5c16e76ed5df87a069bc2f7c1e461c`) nel
manifest sono invariati. Questa è una correzione di persistenza dell'apparato,
non una nuova misura scientifica; Stage A resta **in esecuzione**.
