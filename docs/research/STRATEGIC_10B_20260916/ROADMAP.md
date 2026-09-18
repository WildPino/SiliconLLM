# Roadmap strategica: pretrained ~10B, 50–100 tok/s, CPU AVX2

**Stato aggiornato:** [18 settembre 2026](STATUS_20260918.md). La [decisione sul perimetro di `engine.c` del 17 settembre](STATUS_20260917.md) resta valida. Il testo seguente resta lo snapshot progettuale del 16 settembre.

**Data: 16 settembre 2026. Solo analisi e progettazione. Nessun codice, test, benchmark, download di pesi o training eseguito.** Base della lettura: HEAD `f1f9cfd59d34071a004248bc04c9d56e6038c596`. Registro delle fonti, scope e risultati: [EVIDENCE.md](EVIDENCE.md). Le sigle L/W rimandano a quel registro; ulteriori fonti primarie sono linkate nel testo.

## Sintesi decisionale

**Il percorso più plausibile è conservare il pretraining e cambiare il minimo indispensabile, partendo preferibilmente da un donor già sparso/ricorrente.** La conversione deve essere addestrata sulla geometria realmente eseguibile: precisione per organo, mixer, rank, shared path e routing. Se si mantiene il donor denso già studiato, il prossimo passo informativo è un pilot di adattamento congiunto, non un'altra composizione di componenti allenati separatamente. La distillazione verso l'SSM nativo rimane una terza strategia, coerente con l'engine ma economicamente più incerta.

Non esiste oggi nella repository una dimostrazione congiunta di pretrained ~10B, qualità entro +0.02 BPB e 50 tok/s. E63 misura 49.37 tok/s su pesi sintetici mixed-format; E66 dimostra un donor reale 7B con BPB quasi invariato ma fallisce il suo gate di accordo generativo. H4/H2I mostrano che il training recupera punteggio; H5 mostra che assemblarne i checkpoint non preserva quel recupero. Il vincolo dominante è quindi **la qualità alla geometria che entra nei 10–20 ms/token**, non la sola capacità di muovere un file da 10B. [L10–L18]

Propongo tre roadmap, con budget separati: R1 preservare un donor sparse/hybrid; R2 trasformare congiuntamente un donor denso; R3 distillare in un SSM nativo con capacità condizionale. Prima di settimane di T4, un gate comune deve fissare identità del modello, byte effettivi, semantica engine, qualità e unità di throughput. Una prima tranche di **24–72 ore effettive di T4×2**, dopo lo screening documentale, può decidere fra le direzioni; non è una promessa di conversione completa. Le tranche successive sono condizionali a risultati nuovi, senza riaprire automaticamente esperimenti chiusi.

Il successo richiede **lo stesso artefatto** per qualità e velocità, parametri distinti realmente appresi, contesto dichiarato e nessuna sostituzione tacita del target con uno student piccolo o con pesi duplicati. 50 tok/s è il primo obiettivo; 100 resta una seconda frontiera, da affrontare solo preservando il gate qualitativo.

## 1. Cosa cambia dopo la lettura della repository

| Punto | Evidenza attuale | Conseguenza |
|---|---|---|
| Architettura | `phase60/engine.c` è SSM con dimensioni compile-time; `donor_adaptation/engine/donor_engine.c` è un runtime Transformer distinto | “Gira sul nostro C” e “è convertito nell'SSM nativo” sono due milestone diverse [L01,L02] |
| Packing | Il default storico usa 4 bit/peso; P1 ha già aggiunto `--pack nibble` a **2 bit/peso** | Il prossimo passo lossless 2→1.6 bit offre al massimo 1.25× sul traffico codici, non 2.5× [L07] |
| Qualità | +0.00004 BPB riguarda ottimizzazioni di inferenza del sandbox | Non è il costo di ternarizzare un pretrained; neppure il costo della ricetta nativa [L06] |
| Cache | t6 sfrutta LLC aggregata; spill graduale e pollution | 16 MB non è un divieto fisico di esecuzione a 10B; è un obiettivo del pool caldo [L03] |
| Prestazioni 10B | E36 ~50; E40 R128 oltre100; E63 mixed 49.37, CI[49.18,50.49] | Tutti artefatti sintetici: nessuna qualità da ereditare [L08–L10] |
| Donor7B | E66 A2 +0.000378 BPB; 137/160 accordi, gate150 fallito | Una precisione fedele esiste; non è ancora il target qualità+rate [L12] |
| Compressioni strutturali | Rank aggressivo, carve e composizione post-hoc degradano | Serve apprendimento congiunto oppure cambiare natura del donor [L13–L18] |

Il loader nativo legge anche copie fp32 complete delle MLP e rappresentazioni di controllo int8 prima dei packed kernels. A 10B, la sola copia fp32 sarebbe circa 40 GB: **il file ternario compatto non implica RAM compatta**. Un export per deployment dovrà separare il riferimento di debug dalla rappresentazione eseguibile, senza eliminare il riferimento come strumento di verifica. È una necessità ingegneristica futura, non codice scritto qui. [L01]

Non uso il vecchio indice come prova superiore ai probe più recenti. Un esempio: T3 mantiene formalmente `VOID`; non lo trasformo in una prova universale contro Hadamard. Analogamente E38/E67 non sono oracoli del miglior selector possibile. [L19,L20,L16]

## 2. Contratto quantitativo e fronte di Pareto

### 2.1 Definizione di successo

Fissare prima di qualunque nuova prova: checkpoint e revisione; numero di parametri **distinti**, totali/attivi/stored separati; tokenizer; corpus UTF-8 identico e split immutabile; dominio primario code più suite di regressione generalista; contesti 2K/8K/32K e, separatamente, eventuale128K; batch1; decode greedy/hygiene coerente con il ramo, output lungo almeno256 token per prompt. Questi sono protocolli proposti, non risultati esistenti.

Gate finale proposto: ΔBPB rispetto al teacher sul medesimo corpus ≤+0.01 preferito, **limite superiore CI95%≤+0.02** obbligatorio; bootstrap per documento, non token correlati. Non trasferire σ≈0.005 del sandbox a donor o domini diversi. Aggiungere task di codice eseguibile e completamento, regressione relativa proposta≤2% sui task primari con intervalli dichiarati, errori sintattici e degenerazione su rollout. I task e le soglie vanno fissati prima dei numeri. Agreement greedy/rank sono diagnostici e, se preregistrati, gate propri; non equivalgono alla capacità generale. Il loro fallimento storico resta un fallimento. [L06,L12,L18]

Per velocità: tok/s end-to-end dei **token emessi e accettati**, bytes UTF-8/s e tempo sulla stessa quantità di testo; TTFT/prefill separati; p50/p95 latenza, RAM picco, working set, contesto, thread, frequenze, occupancy, compilatore e flag. Proporre lower-CI95%≥50, poi≥100: un valore centrale49.37 con CI che attraversa50 non passa. Nessun bonus per ridurre il vocabolario producendo più token per lo stesso testo.

### 2.2 Equazione da usare

Per ogni organo i e formato f, registrare peso letto B_i, riusi, FLOP/operazioni effettive e rate **dello stesso kernel/formato/layout**. Un modello iniziale conservativo è:

\[
t_{AR}\approx\sum_i\max\{B_i/\beta_i,\ O_i/\pi_i\}+t_{routing}+t_{sync}+t_{state}+t_{sampling}.
\]

I termini state/glue vanno esclusi dalla somma se già incorporati in un kernel. L'overlap fra organi può rendere la somma conservativa; non addizionare una misura integrata e il suo costo DRAM una seconda volta. `β_i` non è il picco della memoria: include granularità, cache, decode dei codici e contention. Usare il roofline per scartare proposte implausibili, non per promuoverle.

**Caso denso 10B:** se ogni peso viene letto una volta per token, solo i pesi richiedono 10 GB a int8, 5 GB a 4 bit, 2.5 GB a 2 bit, 2 GB a 1.6 bit. Anche a 40 GB/s ideali, il caso 1.6 bit si ferma a 20 tok/s prima di ogni altro costo; a 28 GB/s a 14. Nessun packing da solo produce 50–100 con 10B densi attivi. Le uscite sono: meno peso effettivamente toccato, riuso esatto su più token, o architettura/capacità diversa. Questa è aritmetica condizionale sul traffico, non un benchmark.

**Budget di progetto, non misura:** riservare il 30% del tempo a compute/glue/margine lascia 14 ms a 50 tok/s e 7 ms a 100 per streaming. Il budget è condiviso da esperti, shared projections e head.

| Rate ipotizzato | Byte/token a50 | Byte/token a100 |
|---|---:|---:|
| 28 GB/s |392 MB|196 MB|
| 40 GB/s |560 MB|280 MB|

A 28 GB/s questo corrisponde, se tutti i byte fossero pesi, a 392M/196M pesi int8 oppure 1.96B/0.98B a 0.2 B/peso. Metadata, padding, scale, read amplification e stato consumano parte del budget. Con il kernel esperti storico a 4.2 GB/s, gli stessi 14 ms comprano solo 58.8 MB; con 17 GB/s kernel-pure, 238 MB. Non scegliere simultaneamente tutte le ipotesi migliori. [L03]

### 2.3 Due pool, tre footprint e un costo nascosto

Separare **file/RAM totale**, **set caldo riutilizzato fra token**, **traffico/token**. Non occorre che tutti gli esperti visitati lungo l'intero forward risiedano contemporaneamente inL3. Occorre che il working set locale e il pool dichiarato residente rispettino il budget, oppure che gli spill siano pagati. Il modello deve includere stato SSM, buffer, lookup tables, scale e cache pollution.

Per un MoE gated omogeneo:

\[
P_{experts}=3LDhE,\quad P_{selected}=3LDhk,\quad B_{selected}=b_eP_{selected}+B_{scales}+B_{padding}.
\]

La gate e parte del lavoro up vanno calcolate prima di sapere cosa saltare: 92% hidden sparsity **non significa** 92% di tutte le letture evitate. Il 2.12× del sandbox è già il risultato di tale asimmetria; non moltiplicarlo automaticamente per routing e altre sparsità. [L06]

**Router:** un router flat fp32 costa `4LDE = 4P_experts/(3h)` byte. Per 10B di esperti e h=128: 104.2 MB di router; h=1024: 13.0 MB, prima del backbone/head. Quindi aumentare E a costo attivo costante non è gratuito all'infinito. Un router gerarchico/product-key/fattorizzato è una nuova architettura da allenare e verificare; le lookup devono restare in cache e gli esperti scelti essere streammati in blocchi.

**Head:** `4DV` byte; D=4096, V=32768 ⇒ 536.9 MB fp32. Weight tying toglie la seconda matrice **memorizzata**, non il prodotto head letto ad ogni token. Una head low-rank con D=4096, V=32768, r=256 ha `4r(D+V)=37.75MB`, ancora oltre 16 MB. Il rank necessario a stare a 16 MiB sarebbe circa ≤113 senza includere nient'altro: è una restrizione qualitativa severa, non un consiglio automatico.

**Esempio SSM puramente progettuale:** D=512, L=16, E=400, h=1024 ⇒ 10.066B parametri solo esperti. Con top-2, 50.33M pesi expert/token: 25.17 MB a 0.5 B/peso o 10.07 MB a 0.2 B/peso. Ma le proiezioni, usando l'estrapolazione storica, valgono circa 117 MB, head V=32768 circa 67 MB, router 13 MB: il pool caldo già non entra. Il basso costo attivo è credibile come contabilità, **la qualità top-2/400 è totalmente non misurata** e il tempo non si ricava dalle sole MLP. [L03; derivazione]

**Copertura training:** routing uniforme dà circa `T·k/E` assegnazioni per esperto/layer su T token. Ridurre k/E riduce il segnale per esperto; con skew alcuni non imparano. La capacità da10B deve essere utile e appresa, non solo allocata.

### 2.4 Obiettivo multi-vincolo

Ottimizzare la tupla `(ΔBPB, task quality, t_token, RAM, GPU-hours, rischio)` con vincoli sul formato eseguibile. Ogni punto Pareto deve avere un checkpoint identificabile. Non sommare miglioramenti da checkpoint incompatibili; H5 dimostra concretamente perché. Per50 cercare margine (progettare16ms, verificare≤20); per100 progettare8ms e verificare≤10. Sono margini proposti, non cambi dei gate storici. [L15]

## A. Tre roadmap distinte

### R1 — Conservare un pretrained già sparse/hybrid

**Filo conduttore:** evitare di dover insegnare ex novo a un FFN denso a funzionare con1–5% di attività. Conservare tokenizer, esperti, routing e mixer del donor inizialmente; comprimere in maniera selettiva, poi convertire verso le primitive target solo dove il gate lo consente.

Due candidati nuovi rispetto allo screen locale meritano una valutazione di metadati. **Granite-4.0-H-Tiny-Base**, Apache 2.0, ha 7B totali/~1B attivi e mixer Mamba2+attention con shared experts: è un pilot sotto target, non un 10B. **LFM2.5-8B-A1B-Base** dichiara 8.3B totali/**1.5B attivi**, conv gated+GQA e licenza LFM Open License 1.0: vicino alla scala richiesta, ma non SSM Mamba1. Sono metadati dei produttori, non evidenza locale di qualità/rate. Il suffisso A1B non sostituisce la contabilità. [IBM model card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base), [Liquid model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base).

Non scaricare tutti i donor per curiosità. Lo screen già esistente di OLMoE, StdMoE14B e Qwen3-30B-A3B resta utile; corregge proprio il trasferimento illegittimo del rate mixed E63 a full-int8. I nuovi candidati devono attraversare lo stesso conteggio, includendo shared experts, head, routing e mixer. [L11]

| Fase e dipendenza | Output | Gate / pivot | Risorsa proposta |
|---|---|---|---|
| R1.0 screen metadati | Config esatta, licenza/revisione, inventario operatori, byte per organo | Scartare incompatibilità o traffico non riducibile plausibilmente entro20ms; nessun verdetto da active params nominali |1–2 giorni di analisi,0T4 |
| R1.1 dopo0: baseline e calibrazione | Teacher nativo, attivazioni e sensitivity; precision map int8/4bit/fp32 | Fedeltà baseline; se il formato perde>0.02 prima di surgery, fermare quella combinazione |12–24 pair-hours |
| R1.2 dopo1: compressione minima | Checkpoint con gruppi/outlier o adapter, stessa struttura del donor | ΔBPB e task pass; affinità al kernel e bytes pass; preferire4bit espressivi a ternario se qualità lo richiede |24–96 pair-hours |
| R1.3 dopo2: bridge verso SSM target | Sostituzioni progressive mixer/attention; dReLU/QAT separati | Gate per blocco e cumulativo; mantenere mixer originale se sostituzione fallisce, marcando bridge incompleto |12–72 pair-hours di pilot; conversione completa da riprezzare |
| R1.4 dopo3: export e misura futura | Stesso checkpoint inC, formati corretti, rate ai contesti dichiarati | Parity + qualità +50; solo dopo tentare100 |3–10 giorni engineering CPU;2–8 pair-hours calibrazione finale |

**Confine di scope:** preservare Mamba2/conv/GQA richiede estendere l'engine C; non è una conversione già supportata da `phase60/engine.c`. Se il vincolo è l'esatta architettura SSM v1, R1.3 è obbligatoria e un bridge ibrido non conta come completamento. Se nessuna sostituzione supera i gate, R1 consegna solo una baseline utile per R3. Aumentare artificialmente7B a10B clonando esperti non soddisfa la capacità richiesta.

**Risorse/rischi:** circa 48–192 pair-hours per fattibilità, più export/calibrazione; con 12 h/giorno, 4–16 giorni attivi; con quota 90 pair-hours/settimana, 0.5–2.2 settimane di sola GPU. Non è la stima del completamento 10B. Rischi principali: conversione mixer, qualità del dominio, vocab/head, licenza e codice custom. Mitigazione: una coordinata alla volta, checkpoint base, router pesato esatto prima della sua compressione. **R1 è la prima scelta per ridurre la quantità di conoscenza da riapprendere; non è oggi una promessa del target.**

### R2 — Chirurgia congiunta del donor denso: shared path + esperti residui

**Filo conduttore:** il dense FFN è distribuito; selezionarne pochissimi gruppi elimina funzioni non recuperabili con un semplice selector. Costruire un percorso condiviso che copra la funzione comune e pochi esperti che modellino il residuo, allenando insieme anche le proiezioni compresse. L'operatore proposto è:

\[
f_l(x)\approx g_{shared,l}(x)+\sum_{e\in S_l(x)}a_{l,e}(x)f_{l,e}(x).
\]

`g_shared` può iniziare con low-rank, ma va confrontato con un piccolo MLP nonlineare. La somma e il peso del router devono coincidere con il deployment; il selector non può richiedere il FFN denso nascosto per scegliere i gruppi. E68 motiva questo cambio di operatore, ma il suo segnale è locale, concentrato in un layer e ancora dipendente da dense-z; non è prova che questa soluzione funzionerà. [L17]

| Fase e dipendenza | Output | Gate / pivot | Risorsa proposta |
|---|---|---|---|
| R2.0 baseline nuova geometria | Donorfp32/int8, forma esatta, costo shared+router+experts+rank | Riutilizzare prove valide, nessun rerunH2I/H4/H5; step-zero della nuova geometria |8–16 pair-hours |
| R2.1 dopo0: shared/residual distillation | Fit layer/block, confronto linear/nonlinear e router realmente economico | Local error per layer/p95 migliora; nessun layer nascosto dalla media; costo entro envelope |16–48 pair-hours |
| R2.2 dopo1: curriculum congiunto | Rank e sparsità decrescono gradualmente; precisione inizialmente fedele; CE+KD+feature loss | Almeno pilot1.5B passa ΔBPB≤0.02 e suite generativa; se curva si appiattisce fuori budget, stop |48–176 pair-hours |
| R2.3 dopo2: conferma target-scale | Identica ricetta sul donor~10B, tuning per sensitivity; esperti distinti realmente appresi | Nessun trasferimento di esponente1.5→10B; quality gate sul target, contabilità aggiornata |240–960 pair-hours riservabili, non garanzia di convergenza |
| R2.4 dopo3: export/rate | ArtefattoC effettivo | Gate congiunto;50 prima di100 |5–15 giorni engineering,4–12 pair-hours calibrazione |

Per isolare cause rispettando “una variabile per stage”, confrontare il **fattore protocollo di training**: geometria finale uguale, dati/token uguali, train separato versus train congiunto. Non lanciare una griglia combinatoria. Dentro il curriculum fissare una sola transizione per volta: precisione, rank, sparsità, mixer. Il modello finale va comunque riaddestrato con tutti gli operatori presenti, perché gli errori interagiscono.

L'ordine proposto è: baseline fedele → shared/residual → rank moderato → routing/sparsità progressiva → QAT selettiva → recupero congiunto. L'inversione “QAT prima di sparsità” è un A/B successivo soltanto se il primo pilot dimostra apprendibilità. Reverse-KL può affinare traiettorie, ma è mode-seeking e non sostituisce copertura CE/forward-KL. Nessuna inversione matematica restituisce l'informazione eliminata da rank/quantizzazione.

**Rischio decisivo:** per raggiungere il budget potrebbe servire una sparsità che il donor non supporta a quella qualità. H4 mostra recupero score senza generazione; H5 chiude il semplice assemblaggio. R2 è il seguito più informativo della linea locale, ma ha rischio maggiore diR1. La fase mixer SSM, se richiesta strettamente, resta un'ulteriore conversione: il primo successo diR2 nel runtime donor non soddisfa automaticamente l'architettura nativa. [L14,L15,L02]

### R3 — Distillazione nativa SSM con capacità condizionale

**Filo conduttore:** ottimizzare direttamente l'architettura CPU: SSM selettivo, piccoli shared blocks, esperti ternari/QAT, dReLU e head controllata. Acquisire la conoscenza del pretrained tramite distillazione progressive/sequence-level, invece di pretendere una conversione quasi algebrica.

La progressione mixer→blocchi→logits segue una linea documentata da [MOHAWK](https://arxiv.org/abs/2408.10189); [Mamba in the Llama](https://arxiv.org/abs/2408.15237) dimostra distillazione di ibridi e riuso di pesi, ma non l'equivalenza specifica richiesta su Zen2. La letteratura comporta miliardi di token di adattamento, non una semplice calibrazione. Il caso BitNet nativamente ternario è stato preaddestrato da zero: non è evidenza di conversione economica di un 10B. [BitNet2B4T](https://arxiv.org/html/2504.12285v2).

| Fase e dipendenza | Output | Gate / pivot | Risorsa proposta |
|---|---|---|---|
| R3.0 scelta teacher/tokenizer | Baseline on-domain; byte likelihood e dati deduplicati | Conservare tokenizer se possibile; cambioV solo con tradeoff misurabile |8–24 pair-hours |
| R3.1 dopo0: pilot nativo | Student100–500M o blocchi distillati, CE-primary con KD challenger | Segnale rispetto CE a uguali token; transitare senza shock non recuperato; non chiamare pilot10B |120–360 pair-hours |
| R3.2 dopo1: crescita capacità | Esperti da pesi riusati/upcycling, curriculum routing, coverage | Unique capacity utile, nessun dead expert sistematico, curva qualità/active-budget |Costo da misurare: E/k e optimizer dominano fattibilità |
| R3.3 dopo2: target~10B | Distillazione+QAT con geometria finale e poche parti dense | ΔBPB≤0.02 **contro teacher**, non solo contro student iniziale; generazione/task |Ordine di grandezza1000–10000+ pair-hours possibile; stop/riprezzo dopo pilot |
| R3.4 dopo3: export nativo | Nuovo formato deployment, golden, decode end-to-end | Tutti i gate sul target |5–15 giorni engineering,4–12 pair-hours finali |

**Rischio:** un core piccolo potrebbe non rappresentare le competenze del teacher, anche con molti esperti. Distillare 10B→200M produce un altro obiettivo; può essere un prodotto utile e un pilot, ma non soddisfa 10B totali appresi. Weight tying e depth-reuse riducono parametri distinti: non conteggiare la stessa matrice N volte come capacità 10B. Upcycling copia una funzione iniziale; la diversità va appresa, e i costi riportati in [Sparse Upcycling](https://arxiv.org/abs/2212.05055) non dimostrano completamento con poche settimane T4.

**Decisione:** R3 è la strada più fedele all'SSM v1 e la meno compatibile con una promessa di conversione in sessioni brevi. Avviarla a piena scala soltanto con gate del pilot e un preventivo misurato. Un risultato piccolo molto veloce non deve mascherare il fallimento del vincolo10B/qualità.

## B. Matrice delle opzioni

“Impatto” indica il meccanismo e un limite teorico quando disponibile, **non speedup previsto**. B/M/A = basso/medio/alto. Le priorità non riaprono i gate storici.

| Leva | Impatto tok/s | Impatto qualità | Complessità | Rischio | Nota decisionale |
|---|---|---|---|---|---|
| Donor già MoE/ricorrente | Alto potenziale: evita dense-active | Preserva più pretraining |A|M/A|R1, nuova semantica da supportare |
| W8 con attivazioni fp32 | Più byte del ternario, ma recupero costoso evitato | Miglior baseline locale donor |M|B/M|E66 non è full-int8 né 50 tok/s |
| INT4 multi-livello, mixed 3/4/8 bit | Fino 2× riduzione byte su W8, prima di overhead | Da calibrare organo per organo |A|M|4 bit di storage ternario ≠ quantizer a 16 livelli |
| Ternario QAT | Byte e LUT favorevoli | Rischio alto su pretrained |A|A|Selettivo; vietato ereditare costo sandbox |
| Packing 4→2 bit |2× codici; P1 già misurato | Lossless |M|B|Riutilizzare, non nuovo esperimento |
| Packing 2→1.6 bit |≤1.25× parte bandwidth | Lossless sul ternario |A|M|Decoder può annullare il 20% di byte risparmiati |
| Scale attivazioni per gruppo/outlier | Velocità ambigua | Può migliorare fedeltà |M|M|Scale/LUT aggiuntive e overflow da contare |
| Low-rank whitened per organo | Riduce bytes/FLOP se r<mn/(m+n) | Rischio crescente al rank basso |M/A|A|E65/H4 impediscono promesse a r/D=1/32 |
| Adapter low-rank residuali | Costo aggiuntivo | Recupero selettivo possibile |M|M|Fusione densa può perdere il vantaggio |
| Pruning di blocchi/layer | Riduce costo eseguibile | Lesioni possibili alle funzioni |M|A|Hessian/output-aware; niente sparsità non supportata |
| Shared nonlinear + sparse residual | Può ridurre attività richiesta | Ipotesi nuova di R2 |A|A|E68 solo locale lineare; serve router senza dense oracle |
| Router gerarchico/fattorizzato | Riduce il termine flat E·D e dispatch | Nuova partizione/routing |A|A|Contare misrouting e expert coverage |
| Hadamard/Fourier | Può aiutare il quantizer, non riduce il rank | Possibile beneficio/danno |M/A|A|Non diagonali gratis; T3 VOID; non ripetere generic sweep |
| Weight tying/cross-block reuse | Risparmia bytes residenti; compute persiste | Va allenato |M/A|A|Niente conteggio parametri duplicati |
| Early exit/adaptive halting | Riduce posizioni/layer eseguiti | Perdita e stato incoerente possibili |A|A|Più sicuro come drafter verificato |
| SKIP esatto dReLU | Evita up/down inattivi | Nessun cambio se zeros esatti |M|B|Sparsità da misurare sul target |
| SKIP predetto | Potenzialmente alto | Falsi negativi alterano output |A|A|Non chiamarlo lossless |
| Prefetch predittivo | Nasconde latenza, non crea bandwidth | Neutro se nessuno skip |M|M|Miss/cache pollution; temporal locality non assunta |
| Co-training predictability | Può dare blocchi stabili | Può sacrificare specializzazione |A|A|Regolarizzare con budget qualità e router load |
| Head low-rank/PQ/RVQ | Abbassa DV | Head sensibile, ricerca approssimata rischiosa |A|A|Codebook cache-residente; E17 già limita low-rank head |
| Vocab ridotto/adaptive softmax | Riduce costo head | Tokenizer/normalizzazione cambiano |A|A|Report bytes/s; softmax gerarchica richiede training |
| Entropy coding per blocco | Riduce traffico se decoder economico | Lossless |A|M/A|Metadata e accesso indipendente; niente decode intero file/token |
| SSM state compression | Stato più piccolo | Errori nel lungo periodo |A|A|Lo stato non cresce col contesto; il costo dipende da D·N·L |
| Scan poly/LUT | Riduce costo exp residuo | Errore ricorrente |M|M|E3.5 già implementato; nuova prova solo su range target |
| SWA adattiva | Riduce KV/local attention | Può peggiorare retrieval/dipendenze |M/A|A|Misurare target-length, non solo corto |
| Threading/dispatch/layout | Recupera overhead | Lossless se ordine preservato |M|B/M|Già molti assi chiusi; no moltiplicatori storici riusati |
| Block-verify esatto | Amortizza pesi shared | Distribuzione preservabile |A|M|Unioni MoE, draft, replay, KV inclusi; C/T condizionale |

Riferimenti della matrice: evidenze locali L03–L23; metodi esterni W01–W08. AQLM/QuIP# sono candidati per bitrate e qualità, non percorsi già ρ-safe nell'engine: codebook residenti e traversal contiguo devono essere dimostrati. [W02,W03,W05]

## C. Sottoproblemi prioritari

Scala 1–5: 5 significa massimo leverage o maggiore trattabilità. Sono giudizi progettuali, non misure.

| Priorità | Sottoproblema | Urgenza | Leverage | Trattabilità | Decisione concreta |
|---|---|---|---:|---:|---|
|0|SP0 — identità, semantica, metriche|Bloccante|5|5|Quale donor, quale runtime, cosa conta come10B e tok/s; reference esatto |
|1|SP4 — preservazione qualità congiunta|Bloccante|5|2|Pilot della geometria eseguibile; BPB+task+rollout |
|2|SP6 — training memory e costo|Bloccante|5|3|Master/optimizer/attivazioni/offload; throughput reale e token budget |
|3|SP2 — streamed budget|Bloccante|5|4|Bytes per organo, precisione, k/h/E; shared+expert insieme |
|4|SP3 — compute floor|Bloccante per100|5|3|Profilo per organo; rank/mixer/head prima di un'altra exp-LUT |
|5|SP1 — resident budget|Bloccante per percorso hot|4|3|Head/router/stato oltre alle proiezioni; spill esplicito se ammesso |
|6|SP7 — export/ABI/parity|Bloccante al deployment|4|4|Formato per-matrice, scale/layout e dati del checkpoint identici |
|7|SP8 — contesto/tokenizzazione|Bloccante alla claim finale|4|3|Qualità e throughput su medesimo testo e contesto |
|8|SP5 — throughput engine|Ottimizzare dopo plausibilità qualità|3|4|Nuovi colli misurati; testare solo layout/operatori nuovi |

SP1 non si risolve imponendo una L3 “virtuale” da 32 MB condivisa perfettamente: t6 e matrici row-partition sono una condizione specifica, con stato e streaming concorrenti. SP2 include la granularità down e non solo il numero di esperti; SP3 include proiezioni lineari O(D²), stato O(DN), attention locale e output head O(DV), che scalano in modi diversi. [L03,L04]

## Analisi matematica operativa delle leve

### Algebra e geometria: comprimere funzioni importanti, non soltanto matrici

Per `W∈R^(m×n)`, SVD rank-r riduce i parametri da `mn` a `r(m+n)` solo se `r<mn/(m+n)`. Il minimo errore Frobenius **al quadrato** è `Σ_{j>r}σ_j²`, non il delta di loss. La metrica operativa è:

\[
E\| (W-\widehat W)x\|^2=\mathrm{tr}[(W-\widehat W)C_x(W-\widehat W)^T].
\]

Whitening/regularizzazione di `C_x` e sensitivity del downstream danno la priorità delle direzioni; un'approssimazione locale Hessian/Fisher, trascurando il termine di primo ordine solo quando giustificato, porta a `ΔL≈½δwᵀHδw`. Per matrici rettangolari o non normali gli autovalori di W non identificano “direzioni morte”: usare valori singolari di `W C_x^(1/2)` e Jacobiani osservati. Validare su held-out e su rollout, perché covarianza calibrata e generata possono differire. [W01; derivazione]

La manifold hypothesis è un'ipotesi, non un certificato di compressione. Misurare rank efficace, energy tail pesata, stabilità dei sottospazi fra domini e principal angles `cosθ_i=σ_i(U_aᵀU_b)`. Gradienti allineati fra layer suggeriscono esperimenti di tying solo dopo normalizzazione e allineamento della base; non provano che i layer siano sostituibili. Clustering di neuroni/pesi va confrontato con partizioni casuali a costo/layout identico e con errore di output, non con silhouette soltanto. [L19]

Un cambio di base ortogonale `Q` conserva i valori singolari: non crea low-rank. `Wx=(WQᵀ)(Qx)` può migliorare quantizzazione degli outlier, ma `φ(Qx)≠Qφ(x)` per nonlinearità generiche, e una rotazione densa può distruggere sparsità o diagonalità della ricorrenza. Fourier diagonalizza operatori con struttura appropriata, come convoluzioni circolari, non matrici apprese arbitrarie. Per MLP con dReLU/SwiGLU, permutation e riscalamenti positivi compatibili possono essere riassorbiti in modo esatto con algebra verificata; rotazioni dense vanno limitate ai confini lineari. [W02; derivazione]

**Inversioni utili:** cambiare ordine delle trasformazioni, non invertire una mappa che ha perso informazione. Quantizzazione Q e pruning P in generale non commutano: `Q(P(W))≠P(Q(W))`. Un curriculum continuativo verso l'operatore finale è più controllabile di due salti simultanei. Pseudoinversa e residui recuperano componenti osservabili dai dati, non conoscenza arbitrariamente eliminata.

### Analisi e informazione: il bitrate ottimo è per organo

Per piccole perturbazioni, `δh_(l+1)≈J_lδh_l+ε_l`; quindi l'errore finale contiene `Σ_l(Π_{j>l}J_j)ε_l`. Errori correlati non si sommano come rumore indipendente. Misurare amplification su sequenze, margini logit e routing, oltre al MSE locale. È la ragione per cui una somma di ΔBPB di singoli interventi non predice la loro composizione. [L15; derivazione]

Per logits con margine top1-top2 `m`, una perturbazione con norma∞ inferiore a`m/2` preserva l'argmax in quel punto; ciò non garantisce intere traiettorie. Per un router top-k il margine fra k-esimo e(k+1)-esimo dà un analogo criterio. Una piccola perturbazione continua può cambiare discretamente il percorso; tie-break stabile e formato esplicito sono parte del contratto.

Il rate-distortion da ottimizzare è pratico: minimizzare `Σ_i t_i(b_i,r_i,k_i)` sotto `ΔBPB≤ε` e memoria, stimando distorsioni empiriche e interazioni. Il Lagrangiano `L_quality+λ_t t+λ_R RAM` guida la selezione, ma non sostituisce i gate finali. Assegnare inizialmente, a scopo di gestione,0.005 BPB a precisione,0.005 a struttura,0.005 a mixer e0.005 a margine; **sono riserve, non additività garantita**. Se l'interazione consuma tutto il margine, tornare a un punto Pareto precedente.

Un ternario ha al massimo `log2(3)=1.585` bit di entropia per peso indipendente uniforme. Se `p0` è elevato, `H=-Σp_i log2p_i` può essere minore, ma scale, correlazioni, padding e decode restano. Cinque trit in un byte danno1.6bit/peso: non implicano un kernel diretto conveniente. Compressione entropica utile soltanto se:

\[
B_c/\beta+t_{decode}+t_{metadata}<B_{raw}/\beta.
\]

Blocchi indipendenti, offset piccoli residenti, codebook in L1/L2/L3 e letture contigue mantengono la ρ-law. Espandere tutto in DRAM prima dell'inferenza risparmia disco, non traffico/token. PQ/RVQ possono comprimere head/embedding, ma lookup per ogni peso da DRAM sarebbe la direzione sbagliata; preferire decodifica per tile o codebook residenti. [W03]

Il bitrate minimo necessario non è deducibile dal numero di parametri o da un istogramma di entropia. La quantità importante è informazione utile alla distribuzione target, stimata tramite curve rate-quality held-out e ablation. La conversione da ΔBPB a perplexity dipende da byte/token: a 4 byte/token, +0.02 BPB significa moltiplicare PPL per `2^(0.08)≈1.057`; il tetto richiesto è molto severo. Non tradurre un miglioramento PPL di un paper con tokenizer diverso direttamente in BPB.

### Sistemi e controllo: stabilità della ricorrenza e del routing

Una ricorrenza locale `h_t=A_t h_(t-1)+B_t x_t` con `||A_t||≤a<1` ha perturbazione uniformemente limitata da `ε/(1-a)` sotto ipotesi di errore per passo limitato. Nell'SSM selettivo reale A/B e le proiezioni dipendono dagli input: la stabilità della diagonale `exp(ΔA)` da sola non certifica il sistema chiuso. Misurare Jacobiani empirici, norme, saturazioni e drift lungo rollout, soprattutto quando si approssimaexp o si quantizza lo stato. La parameterizzazione`A<0,Δ>0` evita una classe di instabilità, non garantisce qualità. [L01,W08; derivazione]

State compression: stimare osservabilità delle componenti su loss/recall futuro, non soltanto varianza dello stato. Balanced truncation offre intuizioni per sistemi lineari stabili; l'SSM selettivo richiede linearizzazioni locali e verifica del prodotto di transizioni. O(1) indica indipendenza dalla lunghezza della storia, non costo zero né memoria illimitata. Con b_s byte, il solo stato costa circa `b_s L_SSM·2D·N`.

Depth reuse conserva i pesi ma ripete compute. Adaptive halting ed early-exit devono mantenere aggiornato lo stato che servirà al token seguente; saltare layer ricorrenti può lasciare stati stantii. Allenare uno state-update economico o usare l'uscita anticipata come draft poi verificato. Non basta una confidenza softmax elevata, che può essere malcalibrata.

Prefetch come controllo predittivo: lo stato osservato genera una distribuzione sugli esperti futuri; scegliere prefetch con utilità `p_use·latency_saved-(1-p_use)·pollution_cost`, sotto un budget di bytes in flight. Non risparmiare le letture corrette in caso di miss. Per skip predetto, i falsi negativi entrano nella loss e il fallback deve essere esplicito. Regolarizzare temporalità e load balance può ridurre diversity e creare expert starvation: monitorare occupazione, entropia routing, worst-expert load e isteresi. L'i.i.d.-like routing locale impedisce di assumere un hot pool. [L04]

Granularità: primo modello `t_expert≈B/β_chunk+Lk·τ_dispatch`, **solo se β è kernel-pure**. Con 64.1b, `τ≈8.4 µs` è un'ancora storica, non una costante universale. A L=32, k=8 sono 2.15 ms solo di overhead ipotizzato. A capacità e frazione attiva costanti, diminuire h aumenta E e k, quindi dispatch; aumentare h riduce la flessibilità della selezione. La granularità ottima minimizza tempo a parità di qualità, non il numero di byte astratti. [L03]

### Verifica a blocchi e allocazione risorse

Per K posizioni verificate e a token medi emessi per ciclo, con l'eventuale bonus token contabilizzato nel lavoro:

\[
t_{emit}=\frac{t_{draft}+t_{shared,once}+K C_{position}+t_{expert,union}+t_{commit}}{a}.
\]

Con routing indipendente, l'unione attesa per layer è `U=E[1-(1-k/E)^K]`. Se E è grande e k/E piccolo, `U≈Kk`; gli esperti quindi non si ammortizzano e le proposte rifiutate costano. I pesi shared possono ammortizzarsi. La formula semplificata storica e C/T≈0.15–0.20 sono filtri di fattibilità, non gate universali. Draft gratuito non significa verify gratuito. [L04,W04]

Per sampling occorre accettazione/rejection corretta rispetto a teacher e draft; matching greedy basta solo al caso greedy fissato. Lo stato SSM/KV deve essere committato soltanto per la sequenza accettata, con semantica corretta delle penalità. La parità distributional non implica output bit-identico usando stream RNG diversi.

**Applicazione dei sei framework a ogni famiglia di leve:**

| Famiglia | Algebra/geometria da misurare | Analisi/controllo/sistemi | Informazione / criterio finale |
|---|---|---|---|
| Rank, tying, pruning | Spectra whitened, principal angles, null matched | Amplification layer-wise, ricostruzione e rollout | Distorsione per byte/compute risparmiato |
| Quantizzazione, basis, head/PQ | Outlier, covariance, simmetrie lecite, codebook geometry | Margini logit/router; decoder nel loop | Rate-distortion con metadata e costo decode |
| Shared+MoE, predictability | Errori dei contributi in output, residual geometry | Collapse, starvation, controller budget e routing stability | Entropia/coverage e qualità per active-byte |
| SSM/mixer/SWA | Osservabilità, rank temporale, struttura transizioni | Stabilità sequenziale e long-context recovery | Informazione utile ricordata, non varianza sola |
| Reuse, exit, distillazione | Ridondanza funzione, feature alignment | Stato coerente, calibrazione confidence, recovery curve | Informazione del teacher preservata a costo fissato |
| Packing, threading, verify | Equivalenza dell'operatore, ordine riduzioni | Pipeline, contention, exact commit | Bitrate effettivo e token accettati/s |

Non è nota a priori la convergenza di QAT o FT. Registrare ΔBPB e task rispetto a token/ore, learning rate, aggiornamenti applicati, clipping e nonfinite. Un fit esponenziale di una curva breve può aiutare il time-cap, ma non prova il floor raggiungibile: niente extrapolazione da poche ore a recupero totale.

## D. Dieci esperimenti ad alto valore informativo — tutti proposti, nessuno eseguito

Gli ID `STRAT-01…10` sono nomi di proposta, non nuovi E/H del programma. Prima dell'implementazione il successore deve confrontarli con la mappa assi aggiornata e assegnare un brief soltanto alla cella effettivamente nuova. “No-regret” significa informazione utile entro un budget limitato; non esito positivo garantito.

| ID | Cosa e come misurare | Significato degli outcome | Tempo/cap proposto |
|---|---|---|---|
|01|Screen esatto dei nuovi donor sparse/hybrid: byte e operatori per organo, head, shared path, stato, router, RAM loader|Se il budget è plausibile selezionare un donor; altrimenti evitare port e training. Nessun rifiuto sulla base di rate di un altro formato|1–2 giorni analisi, zero GPU|
|02|Quantizzazione espressiva 4/8 bit sul donor scelto, stessa attivazione e struttura; ΔBPB paired, margini, task e costo delle scale|Se 4 bit passa, apre una via meno distruttiva del ternario; se passa solo 8 bit, la riduzione deve venire dalla struttura. Non ripete il ternary-rule sweep|12–24 pair-hours|
|03|Successore di E68: shared path nonlineare più residui e router basato su x realmente eseguibile; errore held-out per layer/p95, confronto diagnostico con oracle|Oracle buono/router cattivo: problema di selezione; entrambi cattivi: capacità/operatore; entrambi buoni: autorizza valutazione end-to-end, non promozione diretta|16–48 pair-hours|
|04|Training congiunto versus separato, **stessa nuova geometria finale**, stessi token/seed; partire dal donor, non dalla composizione H5 congelata|Se passa BPB+task si supera la nuova cella; solo score indica un altro SCORE-ONLY, senza export; nessuno passa: pivot R1/R3|48–120 pair-hours|
|05|Allocazione del rank per sensitivity a budget byte fisso; shared basis/adapters; confronto con rank uniforme, senza ripetere E65|Se batte uniforme ma resta fuori +0.02, è progresso locale; se entra, recupera budget per attività FFN|12–36 pair-hours|
|06|Sostituire un blocco mixer verso SSM, poi un piccolo gruppo, preservando FFN e tokenizer; distillazione di feature/logits e contesti crescenti|Se fallisce già un blocco, niente conversione totale; se passa corto ma fallisce lungo, mantenere attention o memory tier. Non dedurre composizione da un blocco|24–72 pair-hours|
|07|Curva qualità contro **byte eseguibili** e sparsità sul primo pilot allenato valido, con tre punti preregistrati; margini router e coverage|Se il punto di qualità richiede >20 ms, cambiare forma; se esiste intersezione, confermare sul target, senza dichiarare 10B dal pilot|16–48 pair-hours|
|08|Solo dopo quality pass: profilo ed end-to-end della **nuova** geometria/formato; RAM senza copie debug, traffic amplification, thread/dispatch e 2→1.6 bit se pertinente|Se il decode costa più del traffico evitato, tenere 2 bit; se overhead domina, ottimizzare quello. Non rifare P1/E40/E63|2–5 giorni CPU, zero GPU|
|09|Solo se C/T e quota shared sono favorevoli: verify K2/K4 su target valido; token accettati, unione esperti, draft/replay, parity e bytes/s|Acceptance alta senza speedup: limite compute/unions; speedup netto e parity pass: adottare. Nessun moltiplicatore assunto|1–3 giorni CPU; zero GPU con n-gram|
|10|Conferma dello stesso checkpoint su nuovo holdout, più seed se sostenibili, contesti 2K/8K/32K e seconda CPU AVX2|Degrado solo OOD/long-context: limitare esplicitamente il prodotto; qualità+rate sul target: unica prova valida dell'obiettivo|2–5 giorni CPU, 8–24 pair-hours valutazione|

Gli esperimenti 02–07 sono un menu condizionale, non una coda da eseguire tutta. La prima tranche può fermarsi a 02 e un ramo fra 03/06. La verifica a blocchi non è prioritaria se la sparsità utile non esiste ancora.

## E. Sessioni T4×2, pipeline A–F e costi

### Limiti fisici e unità del preventivo

**Una pair-hour = entrambe le T4 occupate per un'ora = 2 GPU-hours.** Non assumo quote cloud oggi disponibili: per convertire ore in calendario uso due scenari ipotetici, 12 pair-hours/giorno oppure 90 pair-hours/settimana. Una sessione su una T4 vale 0.5 pair-hours economicamente, ma non dimostra scaling DDP. I 3860 tok/s misurati sul pilot piccolo non sono una stima del 10B. [L21]

La T4 ha 16 GB e picco 65 TFLOPS FP16; due T4 non costituiscono una memoria unificata da 32 GB. [Specifiche NVIDIA](https://www.nvidia.com/en-us/data-center/tesla-t4/). Usare FP16 loss-scaled con parti sensibili FP32; nessun assunto di BF16 hardware su Turing. Master weights e Adam non diventano ternari perché il forward usa 1.58 bit.

Contabilità indicativa per full-finetune 10B: 20 GB pesi FP16, 20 GB gradienti FP16, 40 GB master FP32, 80 GB per i due momenti Adam = 160 GB **prima** delle attivazioni, dipendente dall'implementazione. Gradienti FP32 aumentano ancora. Due T4 richiedono base congelata/quantizzata con adapters, blockwise training, optimizer offload o sharding aggressivo; traffico PCIe/CPU e memoria host devono entrare nel preventivo. DDP replica il modello e non risolve l'OOM. Una base quantizzata del teacher introduce a sua volta errore: il riferimento finale resta il teacher originale o un riferimento validato entro un budget separato.

Ordine di grandezza analitico per training denso: `F≈6PT`. Per 10B parametri e 1B token sono 6×10^19 FLOP; con 130 TFLOPS di picco aggregato, **circa 128 pair-hours al 100% teorico**, circa 427–1282 al 30–10% di utilizzo ipotizzato, prima del teacher/offload. Per MoE usare operazioni attive effettive, ma i parametri/optimizer totali restano da ospitare. Questa stima non è un benchmark né un limite stretto per LoRA, dove il costo dipende da cosa è congelato.

Il preventivo principale viene da `H=T/(3600·q_train) + H_teacher + H_eval + H_IO`. Con q_train di 100/300/1000 tok/s, 1B token richiede 2778/926/278 pair-hours. A 90 pair-hours/settimana: 30.9/10.3/3.1 settimane, prima degli altri costi. Sono scenari, non rate attesi. Distillazioni da 3–20B token della letteratura possono quindi significare mesi o anni di quota, pur essendo piccole rispetto al pretraining originario. [MOHAWK](https://arxiv.org/abs/2408.10189), [Mamba in the Llama](https://arxiv.org/abs/2408.15237).

### Fasi di preparazione offline

Questa tabella è un contratto di input/output. B–D si ordinano diversamente nelle tre roadmap; non vanno concatenati automaticamente. I cap GPU sono per pilot, non promesse di convergenza sul 10B.

| Fase | Input → output | Metrica/gate | T4×2 stimata | Rischio e fallback |
|---|---|---|---|---|
|A — analisi sorgente|Config, revisione/licenza, tokenizer, dati → baseline, sensitivity, forme e contabilità byte|Parità del reference; corpus/scoring riproducibili; compatibilità degli operatori|8–24 pair-hours oltre 1–2 giorni desk|Donor fuori dominio/incompatibile → prossimo candidato, nessuna surgery|
|B — pruning/factorization progressivo|Baseline + sensitivity → blocchi/rank/shared path con maschera eseguibile|Continuità della loss a ogni transizione; qualità e prestazioni proiettate con margine|16–72 pair-hours|Magnitude pruning fallisce → Hessian/output-aware o shared path; se ancora fuori, stop|
|C — distillazione/adattamento|Teacher pinned + nuova geometria → checkpoint recuperato|Controllo CE a uguali token; BPB/rollout/task; feature matching non basta|48–240 pair-hours R1/R2 pilot; 120–360 R3 pilot|Logits costosi o KD non utile → CE-primary; KD come challenger on-domain|
|D — QAT e sparsità|Checkpoint C, precision map, optimizer resume → checkpoint con forward finale|Nonfinite = fallimento; qualità cumulativa ≤0.02; margini/logit/router/task|24–168 pair-hours su pilot, se necessario|Ternario fallisce → 4/8 bit per organo critico; accettare più byte o abbandonare geometria|
|E — calibrazione|Checkpoint D + calibrazione disgiunta → scale/bias/outlier map finali|Nessun fit su val/test; miglioramento held-out; rollout lungo stabile|4–12 pair-hours|Recupero locale peggiora globalmente → tornare a D, non ritoccare gate|
|F — export|Checkpoint finale, tokenizer, operatore esatto → artefatto C + manifest + golden|Reference fp32, medesimo operatore quantizzato, output/token/BPB e RAM attesi; poi rate|1–3 giorni CPU se conversione supportata; 5–15 con nuovi operatori; 0–4 pair-hours controllo|ABI manca → prerequisito Sol; nessuna claim prima della conversione reale|

Per C, conservare il tokenizer quando possibile. Se cambia, likelihood sulle stesse bytes e contesto allineato; PPL/token non comparabile. Cross-tokenizer KD resta challenger: il MVE off-domain ha perso contro CE, non dimostrato impossibilità generale. Teacher logits devono includere normalizzazione e residuo tail se troncati; storage e tempo di scoring del teacher sono espliciti. Randomizzare membership dei chunk prima dello streaming per evitare l'ordine bloccato già misurato come dannoso. [L04,L06,L21]

Per F, specificare header/versione, endianness, dimensioni, identità tokenizer, operatore, bias/norm, tied weights, rank effettivo, orientamento dei fattori, precisione per matrice, packing, scale/zero-points, rounding/clipping, ordine degli esperti, pesi routing/top-k e checksum. E1M1/E4M1 non sono contenitori generici di queste scelte. Tagged-v2 del ramo donor non rende l'exporter automaticamente compatibile. Tenere solo rappresentazioni necessarie nel deployment, con reference offline separato. [L01,L02,L18]

### Sessioni proposte, in ordine decisionale

1. **Sessione S1: 24–72 pair-hours, 2–6 giorni a 12 h/giorno.** Input: un donor selezionato, dati/licenze, baseline e precision map proposte. Output: precision ladder e un pilot strutturale; decisione R1/R2/stop. Non un export 10B.
2. **Sessione S2: 120–360 pair-hours, 10–30 giorni attivi oppure 1.3–4 settimane di quota 90 h.** Soltanto se S1 dà un segnale nuovo. Input: geometria esatta, optimizer resume, costo per step misurato. Output: checkpoint pilot congiunto, confronto con CE/training separato, learning curves e proposta di budget target. Se migliora solo BPB senza task, interrompere la promozione.
3. **Sessione S3: 240–960 pair-hours, 20–80 giorni attivi o 2.7–10.7 settimane di quota 90 h.** Solo per adattamento target compatibile con memoria e costo. Output: target ~10B qualificato oppure fallimento documentato. La distillazione SSM completa può eccedere enormemente questa tranche: serve il riprezzo di R3, non una somma ottimistica delle ore precedenti.

Aggiungere 10–25% di riserva organizzativa per scoring/checkpoint/riavvii **se non già incluso nei rate effettivi**. Non contabilizzarla due volte. Ogni cap sostituisce una promessa di durata finita: allo scadere si adjudica il checkpoint predefinito, non quello scelto ex post.

**Costo monetario:** senza un'offerta attuale del provider, usare `c_pair` in €/pair-hour. Per rendere confrontabili le opzioni, una fascia **puramente ipotetica di pianificazione** di 0.50–1.50 €/pair-hour dà: S1 €12–108; S2 €60–540; S3 €120–1440. Non sono tariffe di mercato o preventivi verificati; storage/egress esclusi. Quota gratuita autorizzata riduce l'esborso, non ore/calendario. R3 a 1000–10000 pair-hours varrebbe €500–15000 a quelle ipotesi. **Le sessioni brevi servono a ridurre incertezza; non abbiamo evidenza che bastino a completare 10B.**

## Handoff per Sol e disciplina documentale

Per ogni nuovo esperimento: brief prima della misura con domanda, baseline, controipotesi, ambito che cambia, gate, budget, stopping rule, dati/revisioni/hash e protocollo; poi log grezzi immutabili, metriche per documento, configurazione effettiva stampata dal runtime, checkpoint finale e optimizer/RNG resume, manifest dei file e decisione. Collegare brief→run→checkpoint→export→risultato→ledger, indicando anche fallimenti, VOID e controlli piantati. Aggiornare l'indice canonico per assi; non riscrivere il passato.

**Tre livelli di parità:** (1) layout/packing/threading lossless: bit-exact contro il medesimo operatore; (2) kernel quantizzato: parità contro reference scalar con gli stessi pesi/scale/rounding; (3) surgery/QAT: confronto statistico con teacher fp32, ma non bit-exact per definizione. Un reference fp32 dei pesi già quantizzati verifica l'implementazione, non la fedeltà al teacher. La tolleranza di un kernel non autorizza a modificare ΔBPB o task gate. [L05]

Non richiedere che PyTorch GPU e C fp32 siano sempre bit-identici attraverso implementazioni diverse di exp/riduzioni: preregistrare tolleranze appropriate e conservare riduzioni deterministiche locali. Niente `fast-math` indiscriminato, niente VNNI nascosto, fallback AVX2 generico. L'eventuale seconda CPU avrà risultati specifici, non erediterà i 185 GB/s di Zen2. [L05]

Commit futuri: messaggi come `docs(strat-03): preregister deployable shared-residual pilot` o `research(strat-03): record failed quality gate and artifacts`, con ID e link ai risultati. **Nessuna firma dell'assistente, nessun Co-authored-by dell'assistente.** Questo lavoro non crea commit. Un esperimento non eseguito resta `PROPOSED`, non `PASS` né `OWED`.

Prima attività di Sol: leggere questo dossier e l'ultima mappa degli assi; proporre un singolo brief per STRAT-01/02 o il pilot scelto, partendo dalle evidenze valide. Non implementare l'export H2I fallito e non rieseguire E63 A10B, H4 terminale o H5. La priorità è colmare il gap del checkpoint congiunto, non generare altri speed test sintetici.

## Limiti e metodologia

Ho consultato il grafo con query circoscritte, letto i cinque riferimenti richiesti, distinto il runtime nativo da quello donor e controllato il programma attuale fino a E68/H5. I riepiloghi superati sono stati risolti dando precedenza al probe aggiornato; il caso T3 VOID resta esplicito. Ricerca esterna su fonti primarie per compressione, distillazione, donor e T4. Le due analisi parallele della skill deep-research hanno coperto evidenze storiche e letteratura, senza interventi sul codice.

I risultati locali sono evidenze di un singolo programma, spesso un host/un seed: non diventano tre prove indipendenti perché citati in README, INDEX e probe. Nessun esperimento nuovo, verificatore automatico o test run è stato eseguito, nel rispetto del vincolo. Il report è un handoff Markdown con registro delle evidenze; non un paper con risultati riprodotti in questa sessione.

Le nuove ipotesi — shared path nonlineare, conversione di nuovi donor, precisione alternativa, allocazione del rank e curriculum congiunto — hanno misure proposte, non probabilità numeriche di successo. Non sono verificati campi modello di ogni revisione futura, tariffe cloud, quote account, prestazioni sul 10B o recupero entro +0.02 BPB. La conclusione operativa resta: **minimizzare la distanza dal pretrained, poi dimostrare qualità e budget nella stessa geometria; impegnare settimane soltanto dopo quel segnale.**

## Bibliografia esterna aggiuntiva

Consultazione: 16 settembre 2026. W01–W08 e tutte le fonti locali sono nel [registro](EVIDENCE.md). Queste fonti aggiungono evidenze di metodo/metadati; nessuna certifica l'obiettivo composto.

- W09 — IBM Granite Team (2025). [Granite-4.0-H-Tiny-Base, model card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base). Sparse/hybrid pretrained e architettura/licenza dichiarate.
- W10 — LiquidAI (2026). [LFM2.5-8B-A1B-Base, model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base). 8.3B total/1.5B active, conv+attention; non sostituire il conteggio con il suffisso.
- W11 — Bick et al. (2024). [Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models](https://arxiv.org/abs/2408.10189). MOHAWK.
- W12 — Wang et al. (2024/2025). [The Mamba in the Llama: Distilling and Accelerating Hybrid Models](https://arxiv.org/abs/2408.15237). Distillazione progressiva, non tempo T4.
- W13 — Microsoft/BitNet authors (2025). [BitNet b1.58 2B4T Technical Report](https://arxiv.org/html/2504.12285v2). Modello nativamente ternario addestrato da zero.
- W14 — Komatsuzaki et al. (2022/2023). [Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/abs/2212.05055). Riutilizzo del pretraining, non taglio automatico del compute attivo.
- W15 — NVIDIA. [T4 Tensor Core GPU, specifiche](https://www.nvidia.com/en-us/data-center/tesla-t4/). 16 GB, 65 TFLOPS FP16, PCIe Gen3.
- W16 — NVIDIA. [CUDA GPU compute capability](https://developer.nvidia.com/cuda/gpus) e [CUDA math/type support](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html). Turing/T4 CC7.5 e vincoli dei tipi hardware.
