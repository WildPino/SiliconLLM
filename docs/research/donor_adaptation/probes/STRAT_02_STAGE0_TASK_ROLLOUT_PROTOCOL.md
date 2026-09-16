# STRAT-02 Stage 0 — task e rollout preregistrati

**Data:** 2026-09-16. **Stato:** protocollo congelato prima di qualsiasi
download dei pesi o lettura della qualità del donor. Nessun task o rollout è
stato eseguito. Questo documento non autorizza da solo Stage 1 o T4.

Il confronto è sempre fra `TEACHER`, `W4_ALL_LINEAR` e, solo se W4 passa,
`MIXED_W4_W2` dello stesso checkpoint, tokenizer e contesto nativo <=4096.
I dati e gli esiti di calibrazione non possono modificare prompt, subset,
metriche, soglie o decoding qui fissati. Il BPB primario rimane quello del
[protocollo token](STRAT_02_STAGE0_TOKEN_SCORING.md); questi task sono gate
aggiuntivi, non sostituti del BPB.

## Fonti e set immutabili

| Suite | Fonte primaria, revisione e file | SHA-256 byte originali | Unità |
|---|---|---|---:|
| Codice | [OpenAI HumanEval](https://github.com/openai/human-eval/tree/6d43fb980f9fee3c892a914eda09951f772ad10d), `data/HumanEval.jsonl.gz`, MIT | `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef` | 164 task |
| Generalista | [PIQA degli autori](https://github.com/ybisk/ybisk.github.io/tree/21edab439af693b961be2f069e8690a88d3b4e37/piqa), `data/valid.jsonl`, AFL-3.0 dichiarata nel README | `93503cc97c679e459b065c3d13e848282e44b2a25213985bed3e5d458abef72d` | 1838 domande |
| Etichette PIQA | stesso repository/revisione, `data/valid-labels.lst` | `b4192dc3a2a0363d9d60ccf79800cbbe2f32ebb17726efdde6970e0b8131bceb` | 1838 etichette |
| Rollout documentale | [heldout document-level locale](../../../../benchmarks/donor_adaptation/density/corpus/strat02_document_holdout_v1/manifest.json), SHA manifest `56f3d707040785b21e657d7ba721814fb32fe63e85feb418fafe38889d8ca749` | JSONL heldout `450da27e25755bb7c71215e148f5863d197af6893385033e6100bb212deafd0e` | 96 documenti |

Gli URL pin-nati e gli hash sono stati controllati senza conservare i file
benchmark nella repository. Prima di usarli, acquisire i tre file esterni
con un'operazione write-once, verificare SHA-256 e numero di record, e
registrare path, comando e hash nell'output. Non usare copie derivate o
split alternativi. Nessun benchmark esterno entra nel training/calibrazione.
Per gli span documentali l'[inventario delle fonti selezionate](STRAT_02_STAGE0_SOURCE_PROVENANCE.md)
è disponibile; non equivale a clearance universale dei singoli libri PG-19
fuori dagli USA e non autorizza la ridistribuzione del testo.
Un preflight **senza forward** con il tokenizer pin-nato ha contato 164
HumanEval (prompt massimo 391 token: EOS+prompt+512 <=904) e 1838 PIQA
(sequenza EOS+prefisso+soluzione massima 245 token; suffisso da 1 a 234).
Tutti entrano nel contesto nativo 4096; il runner deve comunque rifiutare
qualsiasi violazione, non troncare.
Il validatore [task/rollout](../../../../benchmarks/donor_adaptation/density/strat02_task_rollout.py)
ha inoltre letto i tre byte-stream in una directory temporanea e accettato
gli SHA e conteggi 164/1838/1838 sopra; nessun file benchmark è stato
conservato o eseguito. Questo valida le fonti, non il modello.
Sul modello casuale miniaturizzato della vera classe `EmoForCausalLM`
(D64, 1 layer, 4 esperti/top2, seed 20260916) il runner ha passato la
parità cache/full-forward, generato 4 token con un solo prefill e update KV,
e calcolato la NLL di un'opzione PIQA fittizia. Lo stesso controllo diretto
sul codice del donor ha visto delta massimo cache/full di
`3.13e-7` negli hidden e `1.79e-7` nei logits. Non è un test del
checkpoint pretrained, della qualità del benchmark o del rate CPU.

## Regole comuni di inferenza

- Usare esclusivamente `GPT2TokenizerFast` con Transformers `4.57.1` e
  tokenizers `0.22.2`, hash e audit di ID del protocollo token. `model.eval()`,
  batch 1, nessun sampling, penalty, instruction wrapper, chat template,
  few-shot o prompt-specific tuning. Il prefisso di documento/modello è
  l'ID EOS `100257` in tutti i task.
- Per generazione, scegliere a ogni passo il minimo ID fra i massimi logit
  finiti (greedy, tie stabile); fermarsi all'EOS o al cap indicato. Il seed
  registrato è `20260916`, ma il percorso primario non usa RNG. Registrare
  token ID, testo decodificato, motivo dello stop, lunghezza e tempi per ogni
  item; un logit non finito o un ID fuori vocabolario è `FAIL_NUMERIC`, non un
  completamento vuoto.
- Il runner deve verificare su un input piantato la parità numerica del
  forward pieno e incrementale/cache per il medesimo arm, e su un item
  piantato il confine fra prompt e risposta. Una discrepanza di apparato
  sospende tutti i gate (`VOID_APPARATUS`). Per i logits dell'ultima
  posizione di ciascun input piantato, la tolleranza è
  `atol=1e-5`, `rtol=1e-5` con `allclose` e
  finitezza di tutti i valori: è fissata ora, prima del primo forward reale.
- Ordinare gli item secondo l'ordine dei file pin-nati; teacher e candidato
  ricevono esattamente gli stessi ID di prompt. Pubblicare gli output grezzi
  di **tutti** gli item, incluso ogni fallimento, senza selezione ex post.

## HumanEval: capacità di completare codice

Per ciascuno dei 164 record in ordine originale, usare il campo `prompt`
letterale; input `[100257] + encode(prompt, add_special_tokens=False)`.
Generare al massimo **512 token nuovi**, fermandosi soltanto all'EOS. La
completion è `decode(ids_nuovi_senza_EOS,
clean_up_tokenization_spaces=False)` senza taglio euristico a fine funzione,
post-processing o riparazioni. `prompt + completion` è l'unico programma da
valutare; il testo aggiuntivo o un syntax error contano come fallimento.

La metrica primaria è `pass@1 = successi/164` sulla suite di test ufficiale,
un campione greedy per task. Registrare anche compilabilità AST, errori,
timeout e pass per task. Il gate di regressione è
`successi_candidate >= ceil(0.98 * successi_teacher)`; il riferimento è il
teacher misurato **sugli stessi 164 task**. Se il teacher passa zero task,
la fedeltà del codice è `INCONCLUSIVE_NO_BASELINE_ABILITY`, non PASS.
Riportare intervallo paired bootstrap al 95% della differenza di pass@1
(20.000 ricampionamenti dei 164 task, seed `20260916`, quantili 0.025/0.975
con interpolazione lineare); l'intervallo è descrittivo e non rimpiazza il
gate di conteggio fissato.

**Sicurezza vincolante:** HumanEval esegue codice generato e il repository
ufficiale [avverte esplicitamente](https://github.com/openai/human-eval#usage)
di non lanciarlo senza sandbox robusta. Su questo host Windows non eseguire
né importare `prompt + completion`, nemmeno tramite il runner ufficiale.
L'esecuzione dei test richiede un ambiente effettivamente isolato, senza
rete/credenziali/mount scrivibili dell'host, con limiti di CPU, RAM, tempo e
output, verificato con programmi piantati benigni e ostili. La semplice
compilazione AST può essere fatta senza `exec`, ma **non** dà pass@1: finché
la sandbox non è verificata, il gate task è `PENDING_SANDBOX` e nessun arm è
promosso.

## PIQA: regressione generalista a scelta binaria

Usare tutte le 1838 righe `valid.jsonl` e le etichette nell'ordine originale.
Per ogni domanda `goal` e soluzione `sol1`/`sol2`, definire il prefisso UTF-8
esatto `Question: {goal}\nAnswer:`. Tokenizzare **separatamente** prefisso
e suffisso ` {sol}` (spazio iniziale incluso), con
`add_special_tokens=False`. Il modello riceve `[100257] + prefix_ids +
suffix_ids`; sommare la NLL causale solo sui `suffix_ids`, incluso il primo
token predetto dal prefisso. Questa separazione esplicita rende stabile il
confine BPE; non retokenizzare la concatenazione con regole implicite.

La scelta primaria è l'opzione con **minore NLL media per token di suffisso**;
tie esatto -> opzione 0. Vietati EOS nella risposta, few-shot e generazione
libera. La metrica primaria è accuracy su 1838; riportare anche accuracy con
NLL totale non normalizzata, NLL/media per opzione e ID della scelta. Il gate
è `correct_candidate >= ceil(0.98 * correct_teacher)`. Se il teacher non
supera il 50% di accuracy, la suite non dimostra capacità generalista del
donor (`INCONCLUSIVE_NO_BASELINE_ABILITY`, mai PASS). Riportare bootstrap
paired 95% della differenza accuracy (20.000 ricampionamenti delle 1838
domande, seed `20260916`, quantili 0.025/0.975 lineari), senza usarlo per
selezionare item o soglia. Il caso di parità esatta al 50% è inconclusivo.

## Rollout: stabilità fuori dal teacher forcing

Per **ognuno** dei 96 documenti heldout, nell'ordine del JSONL, usare
`[100257] + primi 256 token payload` già verificati dall'audit tokenizer.
Generare al massimo **256 token nuovi** greedily, fermandosi solo all'EOS.
Non usare la continuazione reale come target: serve a testare drift e
degenerazione in free-run, non a calcolare BPB. Registrare categoria
(`code`, `technical_general`, `prose`), token e byte emessi, EOS/cap, NLL
eventuale solo come diagnostica e i controlli seguenti sul *solo output*:

1. `run32`: almeno 32 ID di token consecutivi identici;
2. `loop8x3`: una sequenza di 8 ID ripetuta tre volte consecutivamente,
   senza spazi fra le copie;
3. `empty`: zero token non-EOS emessi.

Un documento è degenerato se passa almeno uno di questi controlli; il flag
viene conservato separatamente per ciascun criterio. `catastrophic` per un
arm significa **almeno tre** documenti candidati degenerati che non lo sono
nel teacher agli stessi ID-documento. `catastrophic` anche se accuracy PIQA
perde almeno 10 punti percentuali assoluti o HumanEval perde almeno il 25%
dei successi del teacher (quando il teacher ha almeno un successo). È una
stop rule aggiuntiva; un calo minore può comunque fallire il più stretto
gate relativo del 2%. Un teacher che degenera va riportato: eguagliarne il
difetto non è evidenza di qualità assoluta. Sintassi del codice è misurata
su HumanEval, non sugli span `code` ritagliati a metà documento.

## Adjudication

Ogni arm valido richiede: BPB primario con upper CI95 <=+0.02,
HumanEval/PIQA senza regressione oltre il 2%, nessun `catastrophic` nuovo e
nessun problema di apparato, hash, provenance o sicurezza. `FAIL_*` è un
esito scientifico su apparato valido; `VOID_*`/`PENDING_SANDBOX` non sono
promozioni. Se W4 fallisce, W2 non si prova automaticamente. Pubblicare
sempre risultati per dominio e per item, CI e numero di casi, compresi gli
inconcludenti. Nessuna delle suite prova 50/100 tok/s o la conversione
all'SSM nativo.

**Stop per evitare compute inutile, fissato prima dei numeri:** dopo un
fallimento conclusivo del gate BPB di un arm valido, non si eseguono per
quell'arm i costosi HumanEval/PIQA/rollout; il manifest registra
`NOT_RUN_BPB_FAIL`, mai `PASS_TASK`. Analogamente una failure di calibrazione
operativa già prevista dal brief ferma l'arm prima dell'heldout. Il teacher
resta il riferimento da eseguire sulle suite quando il primo candidato
raggiunge il relativo gate BPB; il differimento non permette di cambiare
suite, prompt o soglie dopo il risultato. Un esito BPB inconclusivo non si
risolve scegliendo un subset o leggendo task per promuoverlo.
