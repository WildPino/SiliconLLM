# STRAT-02 — gate di precisione weight-only per StdMoE

**Stato: preregistrazione; Stage 0 e Stage 1 sono PROPOSED, non eseguiti.**
Questo è il brief R1 per una sola cella nuova: fedeltà di precisione
weight-only di un MoE pretrained reale, prima di download, training, export,
port C o timing. Non esistono risultati STRAT-02 e nessun numero in questo
documento è una qualità o una velocità misurata.

## Domanda, identità e non-duplicazione

La domanda è se il checkpoint esatto `allenai/StdMoE_1b14b_1T_Preanneal`,
revisione `d2a4949c9d4ad6cf47fbac131f7e020077332b21`, conserva la qualità del
suo teacher quando si sostituiscono **solo** i pesi lineari con i formati
preregistrati sotto, lasciando invariati tokenizer, operatori del modello e
semantica del router. Non è una domanda su chirurgia, retraining, conversione
SSM, rate CPU o 8K/32K.

Identità nota dal record publisher e dal config/remote implementation pin:

| Proprietà | Valore congelato |
|---|---|
| Stored params | 13,568,641,024 F32 distinti |
| File publisher | 11 shard, 54,275,336,216 byte |
| Architettura | `D=2048`, `L=16`, full MHA, `F=1024`, `E=128`, `V=100352`, embedding/head untied |
| Routing FFN | top-8 = **7 routed + 1 shared**; lo shared è valutato in ogni layer. Con `always_active_experts=null`, softmax routed-127 e softmax shared-1 sono separati: coefficiente shared `1`, coefficienti routed non rinormalizzati dopo top-7 (`norm_topk_prob=false`) |
| Contesto nativo dichiarato | massimo 4096 posizioni; nessuna claim 8K/32K |
| Licenza dichiarata | Apache-2.0 |

Hash SHA-256 dei byte remoti alla revisione sopra: `configuration_emo.py`
`f1bd419d8dd926cf7d15131b8192938a0feda1b003586320fa56de20360e4bf2`
(11,686 B); `modeling_emo.py`
`26f57354940655db673b35d47e9c6b1a85900068f41d91cb08afed4ec53219d6`
(55,477 B). Il caricamento futuro deve verificare i file effettivamente
eseguiti contro questi hash, non solo la stringa della revisione.

Il nearest prior è lo screen metadata [STRAT-01](../audits/STRAT_01_SPARSE_HYBRID_METADATA_SCREEN.md)
e l'addendum StdMoE di
[TARGET_DONOR_FOLLOWUP_2026-09-16](../audits/TARGET_DONOR_FOLLOWUP_2026-09-16.md).
STRAT-01 non ha scaricato pesi né misurato quality/speed; E63 è un artefatto
sintetico mixed-format e non un oracle di layout, banda o qualità StdMoE;
H2I e H5 riguardano rispettivamente una cella Qwen one-byte `SCORE-ONLY` e
una composizione congelata negativa. Questo brief non li ripete, né trasferisce
le loro conclusioni a questo donor o quantizer.

## Dati, tokenizer e separazione calibrazione/heldout

**Limitazione scoperta del corpus legacy:** `density/build_calib.py` prima
taglia ogni documento originale in chunk da 8192 byte, poi li mescola
globalmente e li concatena. Di conseguenza calibration e heldout possono
contenere chunk dello **stesso documento originale**, e gli span casuali possono
attraversare confini fra chunk non correlati. `density/corpus/heldout.txt` e le
relative slice non sono pertanto un heldout document-disjoint e non possono
sostenere il bootstrap per documento del gate primario.

Prima di **qualsiasi download di pesi, lettura di qualità o quantizzazione**,
creare e congelare manifest write-once nuovi, document-preserving e disgiunti
a livello di documento per una valutazione **code** e una **general**. Ogni
manifest deve dichiarare fonti, licenze/accesso, SHA-256 degli input, encoding
e normalizzazione confermati, ID documento stabili, ordine, offset byte degli
span, assegnazione calibration/heldout e la prova che nessun ID documento
appare in entrambi gli split. Nessun dataset è scelto implicitamente qui e non
sono ammessi documenti non collegati solo per riempire la suite: fonti e
documenti concreti devono essere approvati/fissati nel manifest prima del
download. Il manifest, lo script che lo genera e i loro hash sono artefatti di
protocollo. Il completamento di questi due manifest è un prerequisito bloccante
di Stage 0 per il primary gate document-bootstrap.

La slice BPB primaria, il regression suite e la generazione usano questi nuovi
manifest document-level. Se un confronto storico è utile, il legacy
`density/corpus/heldout.txt` può essere valutato soltanto come diagnostico di
comparabilità, esplicitamente etichettato non-primario e senza bootstrap per
documento, gate o decisione di promozione.

Per ogni manifest, il runner tokenizza gli stessi span di testo UTF-8 con il
tokenizer StdMoE e calcola BPB sui byte UTF-8 originali. Non confrontare né
riutilizzare token ID Qwen. In particolare `density/common.py:get_slice` può
cacheare per `part/n_seq/seq_len/seed` e rifiutare un cambio tokenizer, e
`h1_heldout.npz` contiene ID Qwen: la sua slice 24×512 non è una slice StdMoE
identica e non è un input ammesso al gate. Il runner deve quindi avere un
namespace/cache key che includa almeno tokenizer identity/revision e manifest
hash, oppure bypassare tale cache con una lettura verificata degli span grezzi.

Prima del download sono da congelare, senza inventare soglie empiriche nuove:

- confini esatti del primary BPB slice e la formula (NLL in bit divisa per i
  byte UTF-8 del testo valutabile), inclusi BOS/EOS, truncation, stride e
  gestione degli span troppo corti;
- composizione del regression suite generalista, prompt, decode settings,
  metriche, criterio di regressione e definizione operativa di
  `catastrophic`; nessun task o threshold non già pin sarà scelto dopo aver
  visto il heldout;
- suite di rollout/generazione, prompts, lunghezza, seed, sampling/greedy,
  controlli di validità e regole di giudizio per degenerazione, ripetizione,
  errore sintattico e fallimento catastrofico;
- seed, versione delle librerie, hardware/software, tokenizer artifacts e
  ordine paired teacher/candidate. Le statistiche per dominio saranno sempre
  riportate separatamente oltre all'aggregato.

Il teacher e ogni arm confrontano lo stesso testo UTF-8 retokenizzato; token
IDs, token count e BPB non sono confrontabili tra tokenizer diversi, ma qui il
tokenizer resta quello StdMoE in tutti gli arm.

## Bracci, invarianti e formato da specificare prima delle misure

Prima del primo benchmark/forward, una futura implementazione deve scrivere e
hashare una specifica format per ogni arm: quantizer esatto, simmetria/zero
point, unità di grouping per organo, scale e loro dtype/layout, rounding e tie
rule, clipping/outlier policy, metadata, padding/alignment, decoder e formato
di file effettivamente letto. La specifica non può essere scelta o alterata
dopo il heldout. “W4” e “W2” senza questi campi non sono formati misurabili.

| Arm | Pesi grandi interessati | Precisione |
|---|---|---|
| `TEACHER` | checkpoint F32 originale | F32, riferimento esatto |
| `W4_ALL_LINEAR` | tutti i grandi organi lineari, inclusi attention, router, routed/shared experts e head | W4 weight-only |
| `MIXED_W4_W2` | attention, router e head sempre attivi; routed e shared experts selezionati | W4 sempre-attivi; W2 per tutti gli esperti selezionati, incluso lo shared |

Restano identici tokenizer, embedding lookup, RMSNorm/altre operazioni,
attention, softmax, KV/cache, ordine dei layer, top-k, coefficienti e
normalizzazione del router, selezione stabile/tie behaviour e somma
routed+shared. Non comprimere attivazioni, KV/cache o operatori in questa
cella; non fare finetuning, QAT, adapter, retraining, pruning, changing
router, context extension o sostituzione del modello.

## Stage 0 — autorizzazione di download (PROPOSED)

Stage 0 è una checklist documentale e di capacità; non scarica né esegue pesi.
Prima di scaricare anche uno shard, registrare write-once:

1. conferma Apache-2.0/accesso, publisher/revision esatti, config e hash del
   remote/trusted model code da usare, con ogni dipendenza `trust_remote_code`;
   verificare in ambiente isolato l'import della configurazione e del codice
   dopo revisione di sicurezza, senza pesi: il config riporta Transformers
   `4.57.1`, la `.venv` locale osservata usa `5.13.1`. Un controllo che ha
   importato **solo i simboli della libreria locale**, senza codice remoto,
   ha già fallito su `OutputRecorder` da `transformers.utils.generic`:
   la `.venv` corrente non è un runtime valido per questo checkpoint.
   Verificare un ambiente isolato pin-nato compatibile oppure una modifica
   di compatibilità sottoposta a nuova parity. Il codice remoto applica
   `use_kernel_forward_from_hub("RMSNorm")`; per il primo import/teacher
   reference impostare `USE_HUB_KERNELS=NO` prima di importare Transformers,
   così non compare una dipendenza runtime non pin-nata da hub kernels;
2. inventario dei tensori/operatori e verifica che l'implementazione remota
   realizzi realmente 7 routed + 1 shared, top-8 e la combinazione dichiarata;
3. RAM e spazio disco disponibili al momento, cap di download esplicito
   (massimo gli 11 shard/54,275,336,216 byte più il margine fissato nel piano),
   e un piano di loading bounded-memory che non materializzi implicitamente
   una seconda copia F32 completa; il piano indica peak-RAM stimato, streaming
   per shard, ownership e cleanup non distruttivo;
4. i manifest code e general document-preserving, le loro fonti/licenze/hash,
   la prova di split document-disjoint e tutti i protocol choices elencati
   sopra, già hashati; e
5. versione/hash di brief, runner previsto, environment lock e destinazioni
   output write-once.

**Nota di avanzamento Stage 0, non risultato di qualità:** dopo la stesura
iniziale del brief, un target temporaneo isolato con Transformers `4.57.1`
ha importato `AutoConfig` e `EmoForCausalLM` con `USE_HUB_KERNELS=NO`; i file
di codice importati corrispondono agli hash sopra. Nessuno shard peso è stato
scaricato. Questo soddisfa il solo controllo d'import; non soddisfa ancora
parità del forward, loader bounded-memory, dati document-level o gli altri
punti Stage 0. La `.venv` locale 5.13.1 rimane non compatibile as-is.

Se licenza, revision/code hash, capacità, manifest document-level, formato o
piano di loading non sono verificabili, lo stage è `VOID_PRE_DOWNLOAD`: nessun
download e nessun numero di qualità. Un cambio di qualunque artefatto congelato
richiede una nuova preregistrazione, non la sostituzione silenziosa del file.

## Stage 1 — teacher e W4 (PROPOSED)

Dopo Stage 0, caricare soltanto secondo il piano bounded-memory e stabilire
prima il `TEACHER` oracle: due run deterministiche dello stesso testo devono
concordare secondo la tolleranza numerica preregistrata; devono inoltre
produrre NLL/BPB, rollout e task artifacts completi. Fallimento di
determinismo, hash/provenienza, tokenizer, loader, router parity o controlli
piantati è `VOID_APPARATUS`, non un esito quantization.

I planted tests devono precedere sia il teacher sia il heldout: tiny tensors
con scale/rounding/clipping e tie costruiti per coprire estremi, zeri, valori
sui confini e padding; un blocco MoE piantato deve verificare 7 routed + 1
shared, pesi router, tie top-k e output combinato. Per ogni organo/formato,
la decodifica weight-only deve riprodurre il riferimento di quantizzazione
indipendente e il kernel/loader deve leggere precisamente metadata e payload
registrati. La parità fallita non può essere “corretta” con tuning sul heldout.

Solo dopo teacher/parità validi, misurare `W4_ALL_LINEAR` su calibration per
verifiche operative prefissate, poi esattamente una volta sul heldout. Nessun
quantizer, grouping, clipping, scale, organ allocation o checkpoint viene
scelto dal heldout. Il confronto primario è paired, per documento: bootstrap
dei documenti con upper CI95 della differenza `BPB_candidate - BPB_teacher`;
il gate obbligatorio è **upper CI95 <= +0.02 BPB**. Riportare delta, CI,
numero documenti/byte, bootstrap seed e distribuzione per dominio; non fare
bootstrap di token correlati.

In aggiunta, eseguire la regression suite e il rollout/generazione congelati,
con risultati per dominio. Un failure `catastrophic` nella definizione
congelata fa fallire l'arm anche se BPB passa. Se il teacher è esso stesso
scarso in un dominio rispetto al criterio preregistrato, registrare il fatto
e non chiamare un eventuale pareggio del quantizzato un successo dell'obiettivo
di qualità: è soltanto fedeltà al teacher su quel dominio.

Se W4 non passa tutti i controlli/parità/fedeltà/regressione/rollout richiesti,
**non procedere automaticamente a W2**. Il ramo si ferma, oppure un successivo
brief può proporre una nuova coordinata dichiarata; non reinterpretare il
fallimento come motivazione per scegliere W2 dopo il test.

## Stage 2 condizionale — mixed W4/W2 (PROPOSED, non autorizzato da solo)

Questo stage esiste solo se Stage 0 e Stage 1 sono validi e l'arm W4 soddisfa
i gate richiesti. `MIXED_W4_W2` usa la format specification già congelata per
W4 e W2; tutti i 128 esperti memorizzati sono rappresentati W2 perché
qualunque routed può essere scelto, e ogni token legge esattamente sette
routed più lo shared. Lo shared non si omette né si fonde nel router.
Ripete calibration e una sola
valutazione heldout paired contro **lo stesso teacher**, con identici gate BPB,
regression/rollout, reporting per dominio, provenance e no-hidden-test-tuning.
Un fallimento chiude questo arm; non autorizza training, una ricerca di bit
width, un rerun selettivo o una promozione C.

## Byte ledger e confine di un futuro port C

Per il layout specifico del donor, l'aritmetica derivata (non misura) è:

| Elemento | Pesi caricati/token | Payload mixed |
|---|---:|---:|
| Attention + router + head, W4 | 478,150,656 | 239,075,328 B |
| 7 routed + 1 shared expert, W2 | 805,306,368 | 201,326,592 B |
| **Totale** | **1,283,457,024** | **440,401,920 B/token** |

Il payload mixed implica 31.46 GB/s per soli pesi nel budget streaming di
14 ms (e 22.02 GB/s nel periodo intero di 20 ms, senza altro lavoro). Scale,
metadata, padding, read amplification, cache/KV, routing, compute, dispatch,
sampling e tempo di caricamento non sono inclusi. Questi quozienti non sono
un benchmark e non trasferiscono E63: il suo 49.37 tok/s e la sua banda
descrittiva appartengono a pesi sintetici, formato/layout diversi e 88.6% di
pesi packed, non a questo full W4/W2 donor.

Un futuro port C può essere proposto **solo** con un nuovo brief, dopo quality
passata per l'arm candidato e dopo una plausibilità dimostrata di payload
440,401,920 B/token, metadata e computazione entro il budget. Non nasce da
E63 né da una velocità sintetica. Tale brief deve mantenere artefatto,
contesto nativo <=4096, format e operatori identici, poi stabilire parity e
misurare rate end-to-end separatamente; questa preregistrazione non autorizza
quel lavoro.

StdMoE rimane un MoE Transformer con full attention, mentre
`benchmarks/phase60/engine.c` è un SSM nativo con operatori e shape diversi.
Un runtime C che preservi StdMoE richiederebbe un'estensione/port e
dimostrerebbe al massimo quella milestone. Se “nostra architettura” significa
esattamente l'SSM nativo senza estensioni, resta obbligatoria una successiva
conversione dei mixer/operatori con gate di qualità e parità propri; il
solo STRAT-02 non la prepara né la certifica.

## Artefatti, adjudication e stop/VOID

Ogni output è write-once e include: hash del brief; revision, config, remote
code e tokenizer; manifest raw-text; elenco shard e hash effettivi; hardware
e environment; format specification e tutti i scale/metadata; runner/source
hash; comandi/config effettivi; seed; log determinismo/parità; risultati
calibration/heldout/task/rollout completi, inclusi fallimenti; e una decisione
arm per arm. Nessun file risultato esistente si sovrascrive.

Un problema di download/integrità, licenza, remote-code trust, memoria,
manifest/cache tokenizer, determinismo, planted parity, data leakage o
provenienza produce `VOID_*` e conserva l'audit senza stima scientifica.
Un problema di gate su un apparato valido è `FAIL_*` e conserva tutte le
metriche; non si ritesta cambiando quantizer, protocollo o heldout. La
sequenza di stop è: Stage 0 failure -> niente download; teacher/parity failure
-> niente W4; W4 failure -> niente W2 automatico; W2 failure -> niente C;
quality pass senza plausibilità byte/metadata/compute -> niente C. Nessun
esito di STRAT-02 è un claim di 50/100 tok/s, di contesto oltre 4096, o di
successo del programma R1.
