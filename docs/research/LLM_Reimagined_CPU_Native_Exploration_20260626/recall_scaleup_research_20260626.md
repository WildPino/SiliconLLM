# Ricerche scale-up sulla spina recall (2026-06-26)

> Tre ricerche lanciate in parallelo al training (pairs=32/sizing), per de-riscare le decisioni imminenti dell'Asse A (recall) + la compressione bancabile. Lette di testa mia (Architetto), non auto-read. Collegamento: [[project_phase55_plan]].

---

## R-A — Il muro recall-costo 1.65× / "IVF gerarchico" → **la fix NON è esotica: `nlist ∝ √N` + rebalancer streaming**

**La diagnosi vera (chiarisce il 1.65×):** il costo IVF ha due termini — *probe* (query vs `nlist` centroidi, cresce con nlist) e *refine* (scan `nprobe×N/nlist` codici, cresce con N). **Col nostro `nlist` FISSO, il refine è LINEARE in N → ecco da dove viene il 1.65× quando il contesto sale.** Risultato classico: il totale è minimo quando i due termini si eguagliano, a **`nlist ≈ √(N·nprobe)`**, che rende il costo **∝ √N (sublineare)** invece di ∝ N. (Euristica FAISS: `nlist = 4√N`.)

**Quindi "gerarchico" per noi = far crescere `nlist` con √N** (~360-512 liste a 128K, ~300 chiavi/lista) + un router cheap perché i più-numerosi centroidi non diventino il collo. **La parte DAVVERO difficile non è il costo-query (facile, √N) — è tenere `nlist ∝ √N` BILANCIATO sotto insert append-only SENZA re-clustering globale.** = il problema SPFresh/LIRE.

**Design raccomandato (CPU-pulito, niente grafi):**
- **IVF a livello singolo con liste adattive √N** + **PQ4 fast-scan** per lista (sequenziale, pshufb, no-VNNI = quello che abbiamo già).
- **Router = scan SIMD FLAT dei ~512 centroidi** (stanno in L2/L3) — **NIENTE grafo**: a 128K lo scan flat batte un grafo; il grafo-router paga solo a `nlist ~10⁵-10⁶`.
- **Streaming = LIRE/SPFresh** (SOSP'23, arXiv:2410.14452): nuova chiave → centroide più vicino → append del codice PQ4 al buffer contiguo della lista (O(1), scrittura sequenziale). Lista troppo lunga → **split LOCALE, riassegna solo le chiavi di confine**, niente re-cluster globale (recall resta piatta). Cresci `nlist` man mano che N sale.
- Opzionale **SOAR** (arXiv:2404.00774, 2ª assegnazione ortogonale) se serve margine di recall (~½ candidati a pari recall, +5-20% memoria).

**Stima costo @128K: ~2-5 µs/token/layer** (vs ~8µs attuali), split ~0.5-1.5µs probe + ~1.5-3.5µs refine. Il refine ora scala `nprobe·√N` (sublineare) → **dissolve il 1.65×** alla radice (veniva dal refine N-lineare).

**Verdetti netti:**
- **Grafi (DiskANN/HNSW/IP-DiskANN) = BOCCIATI per noi:** pointer-chasing, ~100ns random/hop, letale a batch-1. Un grafo va bene SOLO come router su centroidi cache-resident — ma a 128K non conviene comunque.
- **IMI (inverted multi-index) = celle più dense MA patologia di sbilanciamento/celle-vuote PEGGIORATA dallo streaming + assunzione di indipendenza-sottospazi sospetta per dimensioni-chiave SSM correlate.** In riserva.
- **L'asse decisivo è il bilanciamento dell'insert streaming, non il costo-query.** SPFresh/LIRE è il template.

**MIA LETTURA + correzione di testa mia:** la risposta è la migliore possibile = **niente architettura esotica, solo `nlist∝√N` + rebalancer locale, riusando il nostro fast-scan**. MA un caveat che il report non calca abbastanza: **i numeri SPFresh (<10% core, ~1% DRAM) sono a tasso-update 1%/giorno; noi appendiamo OGNI TOKEN = tasso enormemente più alto.** → **il costo del rebalancing LIRE sul critical-path di decode è la cosa da MISURARE davvero** (non lo split in sé, ma la sua frequenza/ammortamento al nostro insert-rate). È lì il rischio nascosto, non nel query-cost.

**Failure modes (dal report + mio):** (1) **drift/sbilanciamento liste** sotto append → LIRE; (2) **dimenticare di far crescere nlist** = il 1.65× ritorna (tutta la sublinearità dipende da `nlist∝√N`); (3) IMI cell-collapse; (4) router-grafo = random gather; (5) SOAR write-amplification; (6) residual-PQ over-engineering (2ª passata LUT = gonfia il refine che vuoi ridurre).

**Probe cheap che ne deriva (microbench, no-training):** estendi `phase56_ivfpq_profile.c` con `nlist∝√N` + simulazione insert-streaming con split LIRE; misura (a) costo-query @128K coi due termini (atteso 2-5µs), (b) **costo del rebalancing per-token al nostro insert-rate** (il rischio vero). Una variabile, ore.

**Connessione (mia) con R-B:** il failure-mode #1 (drift/sbilanciamento dell'INDICE sotto append) è cugino del query-drift di R-B (la DISTRIBUZIONE delle query che si sposta). Entrambi = "l'indice calibrato presto decade" → la cura è online-adaptive. Quando R-B rientra, le tratto insieme.

**Fonti:** SPFresh/LIRE arXiv:2410.14452 · IMI (Babenko-Lempitsky CVPR'12) · ScaNN/anisotropic arXiv:1908.10396 · SOAR arXiv:2404.00774 · FreshDiskANN arXiv:2105.09613 · IP-DiskANN arXiv:2502.13826 · Faiss arXiv:2401.08281.

---

## R-B — Query drift → **TENSIONE COL NOSTRO RISULTATO INFONCE. Da risolvere misurando, non assumere.**

**Il meccanismo (ParisKV arXiv:2602.07721 = la nostra arch, GPU):** i centroidi/codebook sono appresi solo dal *prefill*; le chiavi generate al decode (per noi: query = stati SSM tardi in un rollout lungo) seguono una distribuzione DIVERSA → centroidi stale → recall collassa a fine generazione. Misurato con **Recall@k-vs-step** + **centroid-mismatch** (re-cluster un campione di tutte le chiavi vs centroidi live; nessuna label).

**LA TENSIONE (la leggo io, è il punto):** ParisKV e — cruciale perché CPU/FAISS-native — **IVF-TQ (arXiv:2605.17415) dicono entrambi: ELIMINA il codebook appreso.** IVF-TQ mostra che **il drift non è nemmeno necessario**: anche i.i.d.-shuffled (stessa distribuzione) perde −3.94pp su IVF-PQ → **la fragilità è intrinseca al FITTARE K-means a un campione iniziale.** Numeri: codebook appreso (PQ/OPQ) perde **−3.2…−5.8pp** sullo stream; codebook-free **−0.8…+0.6pp**. → *"un codebook InfoNCE/anisotropico calibrato sull'early-context è l'opzione PIÙ drift-brittle."*

**MA NON RITRATTO il nostro InfoNCE>LSH (run-5/6). Regimi diversi:** il nostro risultato è **routing-quality IN-DISTRIBUZIONE a contesto modesto** (MQAR pairs16/32, ctx384) — e il ramp MQAR **NON testa il drift** (già notato 06-24). R-B parla di **fuori-dall'early-distribution su 128K di chiavi-decode accumulate.** Non si contraddicono: misurano cose diverse. Il nostro InfoNCE-batte-LSH **resta valido nel suo regime**; R-B avverte che **quel vantaggio potrebbe restringersi/invertirsi a 128K sotto drift.** Da risolvere, non ignorare.

**LA RISOLUZIONE ELEGANTE (la tengo): SEPARA RAPPRESENTAZIONE da PARTIZIONE.** InfoNCE aiuta il LATO-ENCODER (sfera, similarità angolare, basin più piatto = robusto allo shift) — quello tienilo. Ma la **PARTIZIONE/quantizzatore dell'indice rendila DATA-INDEPENDENT** (centroidi analitici sign-pattern di ParisKV / residual codebook-free di IVF-TQ + rotazione Hadamard inner-product-preserving). *"Keep InfoNCE for the representation, NOT the partition."* → il nostro "codebook InfoNCE" andrebbe **scomposto** in *rappresentazione-InfoNCE + partizione-data-independent*.

**IL REGALO SSM-SPECIFICO (cheap, da fare comunque):** lo **stato SSM cresce ~10× in norma su 1K→64K** (Mamba-2 1.3B: 131→1330+). → **il drift delle query SSM è DOMINANTEMENTE RADIALE (magnitudine).** → **ℓ2-normalizzare query/chiavi uccide la componente dominante quasi gratis.** Score retrieval su query NORMALIZZATE, MAI sullo stato SSM grezzo (sennò le query tardive over/under-scorano per la norma gonfiata). Ragione più forte per normalizzare di quanto abbiano i paper transformer-KV.

**Ricetta CPU anti-drift (da R-B, costo-ordinato):** ℓ2-normalize (O(d), banale, **il primo win**) → rotazione Hadamard fissa (O(d log d), AVX2, inner-product-preserving) → quantizzazione data-independent (sign analitici o scalare precomputato) → rerank esatto su pochi candidati → **multi-probe la coarse-stage** (obbligatorio: il drift colpisce l'assegnazione di cella PRIMA del rerank — cella sbagliata = vicino mai candidato). **Re-encode periodico del codebook = SCARTATO** (costoso su CPU, recupera ~0.15pp = rumore).

**MIA SINTESI R-A ∩ R-B (la convergenza):** R-A failure-mode #1 (sbilanciamento liste sotto append) e R-B (partizione calibrata-presto che decade) sono **lo STESSO problema visto da due lati: la PARTIZIONE COARSE è la cosa fragile.** Due cure componibili: **R-A = adattala online (LIRE rebalance); R-B = rendila data-independent (centroidi analitici).** Probabile la combinazione: coarse data-independent (non va stale) + fast-scan PQ4 sui codici. La rappresentazione resta InfoNCE.

**MAKE-OR-BREAK (l'esperimento che decide, e che il nostro ramp NON ha fatto):** **Recall@k-vs-posizione a 64K-128K**, InfoNCE-partition vs data-independent-partition, query ℓ2-normalizzate, su un rollout LUNGO. R-B avverte esplicito: *"a 4K il gap è nullo, emerge solo a 32-128K; un test su contesto corto green-lighta falsamente un indice fragile."* → **il nostro pairs-ramp valida il routing ma NON la drift-robustezza; serve un probe a lunghezza-target prima di freezare il design indice a scala.**

**Fonti:** ParisKV 2602.07721 · IVF-TQ 2605.17415 (CPU companion) · Online-PQ 1711.10775 · Mamba state-norm 2509.19633 · PQCache 2407.12820.

---

## R-C — Quantizzazione pesi output-aware → **il nostro principio è già mainstream PTQ; il pezzo nuovo = anisotropico-ScaNN sui pesi + QAT da-zero**

**Il nostro principio "minimizza errore-OUTPUT non errore-PESO" è GIÀ lo stato dell'arte PTQ:** `‖(W−Ŵ)X‖² = (W−Ŵ)ᵀH(W−Ŵ)` è errore-output, **anisotropico nello spazio-pesi** (penalizza di più lungo le direzioni-input ad alta energia). GPTQ (2210.17323), OBC (2208.11580), SqueezeLLM (Fisher/loss-Hessian, 2306.07629), AWQ (heuristica activation-aware, 2306.00978), QuIP#/AQLM. Floor PTQ ~2.4 bit; solo QuIP#/AQLM (macchinari pesanti) entrano nel 2-bit usabile.

**WHITE SPACE confermata:** la forma ESATTA di ScaNN — decomposizione del residuo in **parallelo** (lungo la direzione-score) vs **ortogonale**, e sovra-peso del parallelo per lo score — **non è mai stata trasferita ai PESI.** I metodi-pesi usano la norma-H piena, non un penalty score-condizionato sulla componente-parallela. **Quello è il pezzo genuinamente nostro** (= il nostro risultato recall trapiantato sui pesi).

**Verdetto CPU/LUT (un bivio duro):**
- **Output-aware + UNIFORME/ternario** (GPTQ→ternario, scale AWQ, BitNet b1.58) **compone PULITO** con bit-serial LUT (T-MAC/pshufb, no-VNNI): le scale FP si piegano nella LUT come moltiplicatori esterni; il ternario è il caso nativo. **Sequenziale, cache-resident. = la via sicura e veloce su Zen2.**
- **Output-aware + CODEBOOK VETTORIALE APPRESO** (AQLM, QuIP# lattice) = miglior 2-bit di *qualità* MA decode = **gather del codebook (index→vettore) = accesso random = anti-banda = combatte la nostra tesi.** Non gira su LUT bit-serial. **EVITARE** (stessa lezione di R-A sui grafi e del recall: mai gather-random non necessario).
- **Sparse-outlier hybrids** (SqueezeLLM/QuIP) rompono la sequenzialità LUT (passata densa + passata sparsa CSR). **EVITARE.**

**Il vantaggio train-from-scratch (la leva forte):** QAT da-zero **schiva il floor PTQ ~2.4-bit** (BitNet b1.58 2411.05882 prova che il floor è solo-PTQ, da outlier invisibili a training-time). Precedente score-preserving QAT **ESISTE** = QAT+distillation (QAD): allinea i **logit** dello student quantizzato al teacher FP (= obiettivo output/score-preserving, batte il QAT supervisato a basso-bit). HESTIA (2601.20745) = QAT curvature/anisotropy-aware.

**LA SCOMMESSA DEFENDIBILE (sintesi R-C):** train da-zero **ternario/uniforme**, STE che targetizza il NOSTRO formato LUT AVX2 (così train-grid = infer-grid), + obiettivo score-preserving = **distillation dei logit (QAD) + penalty anisotropico-ScaNN sul residuo di quantizzazione** (la white space). Nessuna fonte mostra questa tripla. **Evita codebook vettoriali e sparse-outlier (valuta random-access che non possiamo permetterci).**

**Failure modes (ranked):** (1) codebook vettoriali vs banda = rischio #1 (non farti sedurre dalle tabelle "miglior 2-bit perplexity"); (2) sparse-outlier rompe la sequenzialità LUT; (3) floor 2-bit PTQ reale ma irrilevante per noi (QAT lo schiva, *ma solo se ci si impegna presto*); (4) incoherence-processing di QuIP (rotazione Hadamard online) *appiattisce* l'anisotropia che vogliamo sfruttare = camp opposto, evitare; (5) ternario-QAT è STE-rumoroso/token-hungry → warm-start FP poi hardening (Continual-QAT 2502.11895); (6) metadata delle scale per-gruppo = banda extra, tieni i gruppi grandi/uniformi.

**Fonti:** GPTQ 2210.17323 · AWQ 2306.00978 · SqueezeLLM 2306.07629 · QuIP# 2402.04396 · AQLM 2401.06118 · ScaNN 1908.10396 · T-MAC 2407.00088 · BitNet-1.58-QAT 2411.05882 · HESTIA 2601.20745 · Continual-QAT 2502.11895.

---

# Core liberi come "piano di controllo" (idea utente, 2026-06-26)

> Idea utente: i core resi inutili dalla banda satura, invece di restare in tempi-morti, fanno altro a servizio del decode (predittore/branch-predictor potente, lavoro sull'output a latenza-0). Obiezione (sua, giusta): anche loro leggono RAM/cache → forse rubano la risorsa scarsa. **La LEGGE che ne ho ricavato: un aiutante è gratis solo se cache-resident e bandwidth-light; può pagare ALU-libera per RISPARMIARE/RIMODELLARE banda, mai consumarla.** 3 forme: (1) predittore-prefetcher=K3, (2) draft speculativo=K5, (3) manutenzione indice async. Lignaggio CS = helper-thread/runahead/decoupled-access-execute, twist nostro = predittore APPRESO su decode LLM.

## R-E — Manutenzione indice async off-critical-path → **SÌ qualificato; il catch è la BANDA, non i core (conferma esatta dell'obiezione utente)**

**Async maintenance = pratica STANDARD e low-risk** in ogni sistema streaming-ANN: SPFresh (Updater foreground ↔ Local-Rebuilder background, split/merge da task-queue, LIRE boundary-only), FreshDiskANN (TempIndex in-RAM + StreamingMerge background), Milvus (compaction su segmenti immutabili). Costo: a 1%/giorno LIRE eguaglia il rebuild globale con <10% core e ~1% DRAM; FreshDiskANN regge 1800 ins+1800 del/s con recall >95% per una settimana.

**Concorrenza = problema RISOLTO:** **single-writer (core-manutenzione) + reader lock-free/snapshot (lookup del decode)**, delete via tombstone, publish atomico per-epoca. Degrado recall durante maintenance misurato **<0.001%** (posting-absence transitoria), NON collasso — *purché* il reader usi snapshot e l'indice tenga slack strutturale (FreshDiskANN α>1; α=1 degrada). Per noi è ideale: inseriamo 1 chiave/token nello stesso processo che decoda → controlliamo il confine di versione; la staleness è limitata alle ultimissime chiavi (= i token più recenti, già nello stato SSM/finestra recente = irrilevante).

**IL CATCH (= la tua obiezione, confermata e resa precisa): "core libero ≠ banda libera".** Tutta la letteratura offloada su **I/O SSD** (risorsa DIVERSA da quella foreground). Da noi **il weight-streaming del decode E la manutenzione in-RAM colpiscono lo STESSO bus DDR4 45GB/s**, e il memory-controller Zen2 è **condiviso tra tutti i 6 core**. → un core con ALU idle ha compute libero ma **NESSUNA banda privata** → la manutenzione che streamma memoria-indice **ruba byte/s al weight-stream e rallenta i token anche girando su un altro core.** È il failure-mode che i paper ANN non affrontano e che il nostro [[project_cpu_bandwidth_research]] già segnalava.

**Mitigazioni (evidence-anchored dal mondo storage, stessa forma di contesa):**
1. **Rate-limit DURO del thread background** (TiKV/RocksDB: 75-80% banda, riserva ≥20% headroom al foreground). Su Zen2 in software (batch + pacing/yield, niente BW-QoS per-core).
2. **Schedula nelle finestre idle** (SILK ATC'19): il decode ha micro-idle tra token e nelle fasi non-bandwidth-bound → drena la coda manutenzione lì.
3. **Tieni la manutenzione CACHE-RESIDENT** (la leva #1): LIRE tocca solo i vettori di **confine** → se split/reassign stanno in **L2/L3**, costano quasi zero banda. Allineato alla nicchia L3.
4. **Batcha** (niente split a ogni token; burst schedulabili).

**Drift-recalibration async (R-B context): async FLIPPA "skip it" → "fallo incrementale in background".** Rimuove l'obiezione-LATENZA (il refresh non blocca più il token) → "compute-heavy" non è più squalificante. **MA non rimuove l'obiezione-BANDA** → design valido = **recalibrazione PARZIALE/incrementale** (pochi sub-quantizer codebook o i centroidi più-derivati per finestra, working-set bounded, rate-limited), **NON full-retrain** (resta O(N) = spike di banda). Il "recovers little" di R-B era per il full-sync; per noi la distribuzione query SHIFTA davvero su 128K → la correzione drift incrementale in background **guadagna il suo posto**.

**VERDETTO: sì qualificato.** Latency-win = inequivocabile e provato. Concorrenza = risolta. **Il vincolo è la banda, non i core** → reale SOLO se la manutenzione è (a) cache-resident boundary-only, (b) batchata in finestre idle, (c) rate-limited con ~20% headroom. Sbagliali e hai spostato il costo da latenza-per-token a banda-rubata-ai-pesi = perdita netta.

**Failure modes:** (1) **furto di banda** (il grande); (2) coarse-lock contention su SPFresh (UBIS arXiv:2602.00563 è il successore che lo risolve); (3) il core-manutenzione resta indietro al ritmo-produttore (1 chiave/token) → backlog split → skew → serve backpressure/queue-guard; (4) slack insufficiente ai version-swap (α=1); (5) drift-recalibration troppo aggressiva = spike O(N); (6) finestra stale-read troppo ampia.

**Fonti:** SPFresh 2410.14452 · UBIS 2602.00563 · FreshDiskANN 2105.09613 · IP-DiskANN 2502.13826 · Online-PQ 1711.10775 · CoDeQ 2512.18335 · SILK (ATC'19) · TiKV rate-limit.

## R-D — Helper-thread/runahead + contesa banda → **idea SANA, ma la risorsa libera è MLP non ALU (correzione al mio framing); e il valore durevole è SKIP, non prefetch**

**LA CORREZIONE (al mio "ALU libera"):** la risorsa recuperabile **NON è l'ALU idle — è il MEMORY-LEVEL-PARALLELISM (MLP) idle per-core.** Per Little's Law `BW = MLP × 64B / latenza`: un core Zen2 ha solo ~16-22 line-fill-buffer → satura ~20-27 GB/s dei ~45. **Un helper su un 2° core porta i SUOI LFB → alza la banda AGGREGATA sopra il tetto di un core solo** (due cori' MLP > uno). = "potentially increasing total memory bandwidth" (ASPLOS 2011). La tua intuizione è giusta; la moneta è la **concorrenza-memoria inutilizzata** (il gap single-core→all-core), non il calcolo.

**Lignaggio + condizione (ASPLOS 2011 = il match più vicino, inter-core prefetching):** "l'unico limite allo speedup è il **rapporto stall-memoria/compute**... quando è grande e gli stall si tolgono con pochi helper-core, speedup **sopra-lineare**" = il nostro regime. **Condizione DURA: il chunk/working-set dell'helper DEVE stare in cache** ("finché sta in L2, fa solo le miss prima, off-critical-path"); se sfora o se i dati già streammano bene → beneficio crolla a **3-4%**.

**Ghost Threading (MICRO 2025) = la lezione di design:** l'helper gira sul **fratello SMT (stesso core fisico, condivide L1/L2/L3), NON un core separato** → zero footprint-cache extra, zero linee duplicate. **1.33× idle, 1.31× sotto pressione-banda** (erosione solo ~2pt). → **su Zen2 6c/12t l'helper va sul SMT-sibling, non su un core fisico separato.**

**CAVEAT STRATEGICO ENORME (per "prodotto per tutti"): il premio si ASSOTTIGLIA sul silicio nuovo.** Un core **Zen4 satura ~57 GB/s da solo** (buffer profondi) → il gap-banda sparisce → **la modalità prefetch-overlap EVAPORA.** **Zen2 (target nostro, buffer shallow) è l'estremo FAVOREVOLE.** → non sovra-investire nel prefetch-overlap (trucco Zen2-specifico); investire nella modalità durevole (sotto).

**IL RIORDINO CHE CONTA (la leggo io): il valore CPU del predittore è SKIP, non prefetch.**
- **(A) Predittore-prefetcher = scommessa FORTE, due modalità in ordine di valore:**
  1. **SKIP (banda gratis):** predici i pesi che la sparsità SCARTERÀ → non li streammi mai. **Riduzione PURA di traffico, ZERO rischio-contesa, sopravvive anche a bus saturo E su silicio nuovo.** = la modalità top, durevole, silicon-independent. (= Deja Vu 2310.17157: MLP minuscolo predice la sparsità del blocco successivo, 80% sparsità, >2×.)
  2. **PREFETCH/overlap (latenza→streaming):** per i pesi che SERVIRANNO, gli LFB dell'helper aggiungono MLP. Vince solo finché `main+helper < 45 GB/s` E solo su silicio shallow-buffer (Zen2). Fragile.
- **(B) Draft speculativo = scommessa DEBOLE su CPU:** un draft streamato ri-legge i suoi pesi sullo STESSO bus → compete. Nets out solo se **cache-resident** (n-gram/lookup/testa Medusa-EAGLE, NON un 2° SSM).

**SOTTIGLIEZZA SSM-SPECIFICA (mia, da R-D): lo streaming dei pesi SSM è REGOLARE → l'HW-prefetcher lo copre già → l'helper-prefetch sul core-compute regolare compra poco.** Il valore dell'helper vive nella parte **IRREGOLARE/sparsa**: il gather della **product-key memory (K1)**, il routing MoE. → **il predittore-prefetcher si accoppia col TIER-MEMORIA (K1, accesso random), NON con lo scan SSM regolare (K2).** Placement preciso.

**Precedente LLM-CPU + nicchia:** Deja Vu/PowerInfer/SparseInfer (predittori di sparsità, ma GPU-side o hot/cold); speculative-decoding CPU in llama.cpp (flag affinity draft) ma issue #21453 = proposta aperta, **nessun risultato CPU misurato**. **GAP CONFERMATO: nessuno gira un predittore APPRESO su core CPU per overlappare il weight-streaming di un decode bandwidth-bound. Lignaggio (inter-core prefetch) + predittori-LLM esistono, MAI fusi su CPU = nicchia aperta** (= K3 + Finding-7, ora col lignaggio sistemistico).

**Budget cache-residency (il numero):** **L2-resident hard ≤ ~256-384 KB** (pesi helper + scratch) = zero traffico DRAM sostenuto = ~256-384K param int8 = abbondante per una testa-predittore Deja-Vu-class che legge lo stato già in L1. **L3-resident soft ≤ pochi MB** ma compete per L3 condivisa coi pesi caldi del main (contesa second-order).

**Ordine di validazione (da R-D):** (1) **profila la frazione di cicli DRAM-stalled + la frazione di traffico scartabile** dello scan SSM (= il soffitto del guadagno); (2) **prototipa prima il path SKIP** (predittore-sparsità, riduzione pura, zero rischio-contesa = il win robusto); (3) SOLO dopo aggiungi prefetch-overlap e misura `main+helper` GB/s vs 45; (4) helper ≤256-384KB L2-resident + throttle stile Ghost-Threading.

**Failure modes:** (1) helper non-cache-resident = killer #1; (2) main già saturo/ben-streamato = floor 3-4%; (3) **prefetch ridondante** (lo streaming SSM regolare è già coperto dall'HW-prefetcher → il valore è quasi tutto SKIP); (4) prefetch mistimed; (5) eviction L3 condivisa; (6) **silicio sbagliato** (Zen4 satura single-core → muore il prefetch-overlap; sopravvive lo SKIP).

**Fonti:** Inter-core Prefetching ASPLOS'11 · Ghost Threading MICRO'25 · Deja Vu 2310.17157 · PowerInfer 2312.12456 · SparseInfer 2411.12692 · runahead HPCA'03 · DAE (Smith'82) · llama.cpp spec-decode #21453 · DuoDecoding 2503.00784 · Chips&Cheese Zen4 BW.
