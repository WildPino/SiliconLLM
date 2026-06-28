# Ripensare l'LLM come funzione matematica — esplorazione CPU-native

> **Mandato (utente, 2026-06-26):** progetto ORIGINALE (novità gradita, non obbligatoria); focus = **LLM velocissimo su CPU**. Ripensare cosa SIA un LLM (è una grossa funzione matematica). Iterare su OGNI passaggio di un LLM tradizionale e provare a **sostituire / fondere / invertire**. Poi nuove architetture + applicazioni di strumenti non-inerenti. Meticoloso e maniacale, con un pizzico di follia. Documento vivente, si ri-itera.
>
> **Stato progetto al via:** backbone SSM trainato (TinyStories BPB 0.81-0.90, ~3200 tok/s su Zen2), recall tier IVF-PQ (costo 8µs@128K validato, qualità InfoNCE in convergenza su run-6). Era-compressore SEE archiviata; **SEE vive come metrica (BPB) + filosofia (compressione-come-design)**.

## La lente (fissa, non negoziabile)

Ogni "ripensamento" è giudicato da:
1. **Byte/token mossi da DRAM** = `byte_per_peso × pesi_attivi_per_token`. Il decode batch-1 su CPU è bandwidth/latency-bound. Meno byte = più tok/s.
2. **Pattern d'accesso**: sequenziale **streama** (banda piena), random **stalla** (latenza DRAM ~100ns). Un algoritmo che dimezza i byte ma li rende random **peggiora**. (Lezione che ha ucciso il gather naive del recall e HXI.)
3. **Cache-residency**: se il set-attivo/token sta in L2/L3 (Zen2: L2 512KB/core, L3 16MB/CCX), la banda effettiva sale 45→200+ GB/s = gradino 4-5×.
4. **Qualità (BPB)** non deve crollare. Anti-Goodhart: barre pre-registrate, una variabile alla volta.
5. **Vantaggio sleale = train-from-scratch**: schiviamo TUTTI i muri post-hoc della letteratura (floor PTQ, sparsità bolt-on). Possiamo *progettare* il modello per essere CPU-ottimale per costruzione.

**Metodo per stazione:** SOSTITUIRE (rimpiazza il meccanismo), FONDERE (collassa due stazioni in una), INVERTIRE (gira il meccanismo al contrario / mettilo sottosopra). Verdetto CPU + flag KEEPER/TRAPPOLA.

---

# FASE A — Le 10 stazioni dell'LLM tradizionale, smontate

Pipeline transformer canonica: testo → **(1)tokenize** → **(2)embed** → **(3)posizione** → L×[ **(4)token-mixing/attn** + **(5)channel-mixing/FFN** + **(6)norm** + **(7)residual** ] → **(8)head/unembed** → **(9)softmax+decode** → **(10)loop autoregressivo**; addestrato con **(obiettivo)** next-token CE.

## A1 — Tokenizzazione (testo → unità discrete)
- **Tradizionale:** BPE, vocab fisso V (32K-100K). Compromesso: V grande = meno token/testo ma head costosa (D×V); V piccolo = head cheap ma più passi autoregressivi.
- **SOSTITUIRE:** byte-level (V=256, head minuscola) — tradeoff noto. Oppure **patching a entropia** (Byte Latent Transformer / SpaceByte / MegaByte): segmenta i byte dove il modello è *sorpreso* (alta entropia next-byte), nessun vocab fisso, compute allocato per imprevedibilità.
- **FONDERE:** tokenizzazione DENTRO il modello — niente tokenizer esterno, il primo strato impara il pooling dinamico (fonde A1+A2). Il confine-token diventa un gate appreso.
- **INVERTIRE:** invece di "dizionario fisso di sotto-parole", un **codebook content-addressed che CRESCE** (è già il nostro lavoro recall/VQ). O più radicale: la tokenizzazione *è* un **coder aritmetico** — i "token" sono codeword di compressione, il modello predice in spazio-codice. = SEE puro.
- **VERDETTO CPU:** il patching-a-entropia (BLT) è **profondamente allineato a SEE** (raggruppa per predicibilità = compressione) E dà compute variabile (meno passi su tratti prevedibili = tok/s effettivi su). Head minuscola byte-level = bandwidth win. **KEEPER (alta leva + on-brand): patch-a-entropia + I/O in spazio-codice.** Madness: il tokenizer È il compressore.

## A2 — Embedding (id → vettore)
- **Tradizionale:** tabella lookup V×D. NB: al decode è **1 gather/token** = cheap in banda (il costo gemello è la head, A8).
- **SOSTITUIRE:** hash-embeddings (niente tabella, embedding = funzione dell'hash del token), embedding fattorizzato (V×E×D, E<D, ALBERT), embedding *calcolato* (funzione dei bit del token).
- **FONDERE:** tie con la head (weight tying, già standard); fondere la posizione nell'embedding (rotary).
- **INVERTIRE:** token → **codice SPARSO** (k-hot in spazio grande) invece di vettore denso → l'embedding diventa una selezione di atomi, non una riga densa.
- **VERDETTO CPU:** l'embedding è già cheap (1 gather, letto una volta). Lo storage V×D non è il collo del decode. **Bassa leva** salvo V enorme → allora hash/fattorizzato. Non è qui la battaglia.

## A3 — Informazione posizionale
- **Tradizionale:** RoPE / ALiBi / PE appreso (sommato all'embedding).
- **SOSTITUIRE:** **la ricorrenza SSM porta la posizione implicitamente** → niente PE esplicito (Mamba non ha PE). Nel nostro mondo SSM, A3 è **già eliminata/fusa**.
- **INVERTIRE:** invece di codificare posizione assoluta/relativa, codifica per **DECADIMENTO**: lo stato dimentica naturalmente → la posizione *è* quanto sei decaduto. La matrice A dell'SSM è un decadimento-posizionale appreso.
- **VERDETTO CPU:** già risolta dall'SSM (zero byte extra). Nota: nel recall tier la posizione rientra come bucket temporale dell'indice → **posizione-come-bucket** (lega al recall temporale/segmentato). **Chiuso (già fuso), con un gancio per il recall temporale.**

## A4 — Token-mixing (attention) — LA stazione grossa
- **Tradizionale:** self-attention, O(n²), KV-cache che CRESCE → bandwidth-dead a 128K. È ciò che stiamo rimpiazzando.
- **SOSTITUIRE:** SSM / linear-attention / state-space (stato O(1)) = backbone. + tier sottile di recall (IVF-PQ) per l'associative recall che l'SSM da solo non fa (Mamba-MQAR collassa). = la scommessa centrale del progetto.
- **FONDERE:** fondere token-mixing con channel-mixing (il meccanismo selettivo di Mamba già mescola tempo E canali in un gate). Un solo operatore "mix" invece di attn+FFN separati.
- **INVERTIRE (potente):** invertire la direzione dell'attention. Tradizionale = ogni token **TIRA** (query) da tutto il passato (pull, ricomputa). Invertito = ogni token **SPINGE** (write) la sua info in un set fisso di slot/registri → **fast-weights / memoria associativa scritta** (Schmidhuber fast-weights, Hopfield moderno). Lo stato SSM *è* la memoria scritta; il recall tier è il lato-lettura. Niente ricomputo: mantieni uno stato aggiornato.
- **MADNESS:** token-mixing come **processo fisico** — diffusione / propagazione d'onda / automa cellulare (regole locali → long-range emergente). Oppure mixing in **dominio trasformato** (FNet: attention → 2D FFT, *zero parametri*). Oppure mixing via **sort appreso** (ordina per chiave, mescola i vicini = attention sub-quadratica tipo merge-sort).
- **VERDETTO CPU:** SSM + recall tier è la risposta (già in corso). L'**inversione push/write = esattamente il nostro stato** (conferma teorica della rotta). **Gancio originale:** mixing **parameter-free** (FFT/trasformata fissa) muove **ZERO byte di peso** per il mix → bandwidth win se la qualità tiene; ma FFT per-token causale su CPU è scomoda (vedi Fase C). **KEEPER concettuale: l'inversione write-state**; **DA-VERIFICARE: mixing parameter-free.**

## A5 — Channel-mixing (FFN/MLP) — DOVE STANNO I BYTE
- **Tradizionale:** up-proj (D→4D) → nonlinearità (SwiGLU) → down-proj (4D→D). ~2/3 dei parametri = **il costo di banda dominante al decode.**
- **SOSTITUIRE:** MoE (FFN sparso), FFN ternario (BitNet), FFN low-rank, FFN strutturato (Monarch). Oppure — il colpo — rimpiazzare l'FFN con un **LOOKUP**: **product-key memory** (Lample et al.): l'FFN diventa una gigantesca memoria key-value con product-keys, **lookup sparso** O(√N), capacità enorme, pochissimi pesi attivi/token.
- **FONDERE:** fondere le due matmul (già fuse nei kernel); **FFN condiviso tra layer** (depth-recurrence → stream-once, vedi A7); fondere FFN con l'output dell'attn.
- **INVERTIRE (il cuore):** l'FFN **È GIÀ una memoria key-value** (Geva et al. "FFN are key-value memories": up-proj = chiavi, down-proj = valori, la nonlinearità seleziona). **Invertilo da implicito-denso a esplicito-sparso:** invece di moltiplicare per TUTTE le chiavi e lasciare che la nonlinearità ne uccida il 95%, **RECUPERA solo le poche chiavi attive** (product-key / sparse-dictionary). Stessa funzione, ma da `D×4D` matmul denso a un **gather sparso cache-resident**.
- **MADNESS:** FFN come **hash-table appresa** (l'attivazione hasha a un bucket di pesi). FFN come **mappa iterata** (riusa la stessa piccola matrice N volte = weight-sharing in larghezza). Nonlinearità come **LUT** (non-smooth, lookup).
- **VERDETTO CPU:** **QUI si vince o si perde** (i byte sono qui). Direzione vincente = sparso + strutturato + ternario, idealmente come **lookup esplicito cache-resident** (product-key memory: O(√N) attivo, la capacità sta in DRAM ma la *parte attiva* in cache). **KEEPER #1 (massima leva): FFN-come-memoria-sparsa-esplicita (product-key).** È la fusione tra "channel-mixing" e "recall" — la stessa macchina.

## A6 — Normalizzazione
- **Tradizionale:** RMSNorm/LayerNorm per-token (compute cheap, ma è un punto di sincronizzazione seriale).
- **SOSTITUIRE:** no-norm con init accurato (ReZero, Fixup); norm fusa nella matmul successiva.
- **FONDERE:** al decode, **piega la gain di RMSNorm nel peso successivo** (folding) → un'op in meno.
- **INVERTIRE:** invece di normalizzare le ATTIVAZIONI a runtime, **vincola i PESI** a train-time (weight-norm / spectral-norm) così le attivazioni restano limitate senza norm runtime. Sposta il costo da inferenza (per-token) a training (gratis a inferenza).
- **VERDETTO CPU:** la norm è cheap in banda (zero byte di peso) ma è una **riduzione seriale per-token** (latenza/sync). Eliminarla/piegarla toglie una dipendenza seriale. **KEEPER-minore (gratis): vincoli-peso a train-time → niente norm a runtime.**

## A7 — Residual / profondità
- **Tradizionale:** x + sublayer(x), L strati impilati, **ognuno coi suoi pesi** (stream L volte).
- **SOSTITUIRE:** **depth-recurrence** (pesi condivisi tra strati — Universal Transformer / ALBERT) → stream-once, cache-resident. **Profondità dinamica / early-exit** (ferma presto sui token facili = banda media giù).
- **FONDERE:** fondere gli strati in un singolo operatore "profondo".
- **INVERTIRE (madness-ma-reale):** invertire la profondità in TEMPO — invece di L strati spaziali, **UN strato eseguito L volte** (profondità = iterazioni di una mappa fissa). Limite = **Deep Equilibrium Model (DEQ)**: trova il punto-fisso di un solo strato (profondità infinita, memoria O(1)), iterazioni adattive/token. Su CPU: **streama UN set di pesi, riusalo** → cache-resident per costruzione, e spendi iterazioni solo dove serve.
- **VERDETTO CPU:** weight-sharing in profondità = **win di banda diretto** (stream-once). DEQ/early-exit = compute adattivo. Entrambi abilitatori di cache-residency. **KEEPER #2: profondità-come-ricorrenza (pesi condivisi) + iterazioni adattive (DEQ/early-exit).**

## A8 — Output head / unembedding
- **Tradizionale:** proiezione D×V → V logit → softmax. A V grande = **costo di banda maggiore** (leggi D×V ogni token).
- **SOSTITUIRE:** tied embedding; adaptive/hierarchical softmax (log V); vocab piccolo (byte/patch) → head minuscola; head low-rank/VQ.
- **FONDERE:** fondere con embedding (tying); fondere softmax+sampling (non materializzare tutti i logit — campiona via albero o via indice).
- **INVERTIRE (fusione originale per noi):** invertire la head da "scora TUTTI i V token poi scegli" a "**predici un vettore-query, RECUPERA i pochi token probabili via indice** (ANN sul vocab / MIPS)". **La head e il recall tier diventano la STESSA macchina** (IVF-PQ sopra gli embedding del vocab). Softmax-come-retrieval, sublineare in V.
- **MADNESS:** non predire una distribuzione sui token affatto — predici i **prossimi BYTE via il coder aritmetico** (la head è un decoder aritmetico). SEE puro, chiude il cerchio con A1.
- **VERDETTO CPU:** la head è un costo di banda a V grande. La **retrieval-head** (predici-vettore → ANN) la rende **sublineare in V con la STESSA tech IVF-PQ** che abbiamo costruito. **KEEPER #3 (fusione molto on-brand): head = recall tier.** Fonde A8 con A4-recall.

## A9 — Decoding / sampling / autoregressione
- **Tradizionale:** un token alla volta, feedback, O(n) seriale.
- **SOSTITUIRE:** **multi-token / speculative** (MTP, Medusa, EAGLE) — emetti K token per lettura-pesi (ammortizza la banda). Self-drafting (niente 2° modello da streammare).
- **FONDERE:** fondere draft+verify in un modello (teste MTP).
- **INVERTIRE (madness-ma-attivo):** invertire l'autoregressione — **non-autoregressivo / diffusion-LM** (predici tutti i token in parallelo, raffina iterativamente — discrete diffusion, SEDD). Su CPU: **leggi i pesi UNA volta, aggiorna TUTTE le posizioni** = profilo di banda radicalmente diverso (potenziale win enorme, ma il refinement iterativo = più passi; netto da misurare).
- **MADNESS:** generazione come **minimizzazione di energia** (il testo è il minimo di una funzione-energia; decode = ottimizza, non campiona). Generazione per **analogia/retrieval** (recupera continuazioni passate simili, splice — semi-parametrico/RETRO).
- **VERDETTO CPU:** MTP = ammortizza letture-pesi (win di banda sicuro). Diffusion/NAR = leggi-pesi-una-volta-per-molte-posizioni (win potenziale grosso, costo iterativo incerto). **KEEPER sicuro: MTP. MADNESS-da-verificare: diffusion-LM su CPU** (il profilo di banda è intrigante).

## A10 — Obiettivo di training (la META-leva)
- **Tradizionale:** next-token cross-entropy (teacher-forced).
- **SOSTITUIRE:** **obiettivo di compressione** (MDL/bits-back esplicito — CE lo approssima ma l'MDL esplicito potrebbe modellare meglio il long-range); multi-token / fill-in-middle; il **regolarizzatore di predicibilità** (Finding 7 — addestra il modello a essere cache-predicibile).
- **FONDERE:** **fondere l'obiettivo LM con l'obiettivo-indice (InfoNCE recall co-trained) E la predicibilità-cache** = un'unica loss multi-termine che plasma il modello per essere veloce-su-CPU **per costruzione**.
- **INVERTIRE:** invertire il teacher-forcing — addestra sui PROPRI rollout (l'iron-law TF≠generativo; closed-loop / scheduled-sampling / DAgger) per chiudere il gap train-test che ci ha morso nell'era mantra. O addestra ALL'INDIETRO (predici il token precedente) come ausiliario.
- **MADNESS:** addestra a minimizzare **wall-clock per unità di qualità** (loss latency-aware) — cuoci il costo-CPU nell'obiettivo, l'ottimizzatore baratta qualità per velocità dove costa poco.
- **VERDETTO CPU:** **questa è la leva META** — l'obiettivo può RENDERE tutte le stazioni sopra CPU-friendly per costruzione (routing predicibile, attivazione sparsa, working-set cache-resident). Train-from-scratch + obiettivo CPU-aware = il vantaggio sleale reso operativo. **KEEPER-meta: obiettivo CPU-aware (predicibilità + sparsità + compressione co-trained).**

---

# Sintesi Fase A — i keeper trasversali (NON 10 idee, ne convergono ~5 che si rinforzano)

| # | Keeper | Stazioni | Leva CPU | Maturità |
|---|--------|----------|----------|----------|
| **K1** | **FFN/head/recall = UNA memoria sparsa cache-resident** (product-key + retrieval-head + IVF-PQ) | A4,A5,A8 | **Massima** (i byte sono nell'FFN+head; lookup sparso = pochi attivi) | Media (product-key noto, fusione no) |
| **K2** | **Profondità-come-ricorrenza** (pesi condivisi + iterazioni adattive/DEQ) | A7 | Alta (stream-once, cache-resident) | Media (ALBERT/DEQ noti) |
| **K3** | **Obiettivo CPU-aware** (predicibilità+sparsità+compressione co-trained) | A10 | Meta (rende K1,K2 veri per costruzione) | Bassa (originale, da-zero) |
| **K4** | **I/O a entropia** (patch-a-entropia in ingresso + coder aritmetico in uscita) | A1,A8 | Media (compute variabile, head minuscola) = SEE-native | Media (BLT noto, fusione SEE no) |
| **K5** | **MTP self-drafting** | A9 | Media (ammortizza banda ~1.5-2.5×) | Alta (DeepSeek/Medusa) |

**LA TESI UNIFICANTE (il "ripensare cosa sia un LLM"):**
> In un LLM tradizionale, **COMPUTE (matmul densi) e MEMORIA (parametri) sono ENTANGLED** in ogni strato denso. Su CPU il pattern d'accesso-memoria è tutto. **Quindi: SPLITTALI.** Un **piccolo nucleo di compute ricorrente, predicibile, ternario, strutturato** (depth-shared SSM) avvolto attorno a una **grande memoria content-addressed cache-resident** (FFN-come-product-key + recall tier + retrieval-head), con **I/O a tasso variabile per entropia**, addestrato con un **obiettivo che premia la velocità-CPU**.
>
> Slogan: **"un LLM è una funzione; rifattorizzala così che la parte pesante sia un LOOKUP, non un MOLTIPLICA."** Il moltiplica streama tutti i pesi; il lookup tocca solo gli attivi e, se predicibile, sono già in cache.

Questo è coerente, modestamente originale (la *fusione* lo è; i pezzi esistono), e dritto sul focus velocità-CPU. K3 (obiettivo CPU-aware) è il pezzo più originale e quello che lega tutto.

---

# FASE D — Risultati ricerca

## Agente 1 → K1 (FFN+head+recall = memoria sparsa) — VALIDATO ma con CONDIZIONE-DI-SCALA tagliente

**Cosa esiste (la base è solida e MATURA):** la linea product-key memory (PKM, Lample 2019, arXiv:1907.05242) è stata rianimata e validata a scala 2024-25:
- **UltraMem (arXiv:2411.12364, ByteDance, ICLR2025)** = il paper chiave per noi. Eguaglia la qualità di un denso 6.5B al compute di un 1.6B, **2-6× più veloce di MoE**, fino a 20M slot. Costo d'accesso per-token framato come **memory-access** (la metrica giusta).
- **PEER (arXiv:2407.04153, DeepMind)** = PKM su >1M esperti-singolo-neurone, batte il denso sul frontier.
- **Memory Layers at Scale (arXiv:2412.09764, Meta)** = scaling laws, regge a 128B param-memoria.
- **Geva (arXiv:2012.14913)** = la licenza teorica: l'FFN denso *è già* una key-value memory, e i suoi valori **vivono nello spazio-vocabolario** (ogni valore induce una distribuzione su token) → **è il ponte non sfruttato** verso la fusione FFN↔head.
- Head-come-retrieval esiste e matura: **FlashHead (arXiv:2603.14591)** drop-in retrieval/clustering head no-retrain; adaptive-softmax (Grave 2016); LSH-softmax; kNN-LM (arXiv:1911.00172).

**ORIGINALITÀ CONFERMATA (due pezzi davvero nostri):**
1. **Nessuno ha fuso la memoria-FFN e la head in UN UNICO indice condiviso.** Le due letterature sono disgiunte. Il ponte di Geva ("i valori FFN sono già in spazio-vocabolario") non è mai stato operazionalizzato come un solo product-key index che serve sia il recall-FFN sia l'unembedding finale. = **il cuore originale.**
2. **Il framing batch-1 / cache-residency-CPU / gather-prefetchato è vuoto.** Tutta l'analisi memory-access esistente (UltraMem incluso) mira a GPU/server-batch. Il co-design a batch=1 su Zen2 è terreno aperto (uguale forma di nicchia del recall-128K-CPU).

**LA CORREZIONE DECISIVA (la leggo io, tempera il keeper):** PKM riduce i byte/token a batch=1 **solo SPOSTANDO il muro da banda-sequenziale a latenza-gather-random.** Vince **solo nel regime large-capacity/DRAM-bound**: quando l'FFN denso **eccede L3 (16MB)**, lo streaming denso domina e il gather di pochi KB lo schianta. **Ma se il modello è PICCOLO e l'FFN sta già in L3, lo streaming denso è GIÀ a velocità-cache → PKM è un pareggio o una PERDITA** (paghi index + gather-random + rischio-collasso per niente). → **K1 NON è una leva sandbox-scale (1.6M-6.7M stanno tutti in L3): è una leva SCALE-UP.** A scala grande è esattamente giusto (capacità disaccoppiata dalla banda: memoria enorme in DRAM, set-attivo piccolo). Coerente col framing del progetto (TinyStories=sandbox, target vero=modello agentico grosso).

**IL BRIDGE ORIGINALE (la scoperta che fonde K1+K3+Finding-7):** l'unica debolezza di PKM su CPU è la **latenza del gather-random** dei top-m valori. Ma index-search e gather sono **separabili** → appena hai gli indici top-m puoi **software-prefetchare** le righe-valore. La ricerca dice che questo è **il singolo più grande ottimizzatore CPU ed è impubblicato.** E noi abbiamo l'oracolo di prefetch: **lo stato SSM predice quali slot serviranno** (K3 = obiettivo di predicibilità) → prefetch accurato → **la latenza-gather, unica debolezza di PKM, sparisce.** = product-key memory (K1) + regolarizzatore di predicibilità (K3) + predittore-prefetcher (Finding 7) **collassano in UNA tesi CPU-native coerente e originale.**

**Failure modes da sorvegliare (evidence-anchored):** (1) **collapse/dead-keys = #1 documentato** (flat-key usa ~10% slot; product-key ripristina ~100%) → query-norm + LR-warmup + load-balance obbligatori; **BatchNorm-su-query va piegata in affine statica a batch=1**. (2) bias topologico del top-k product-key (top-k a metà ≠ top-k vero) → UltraMem TDQKR (Tucker+rank1). (3) **degrado rare-token se fondi la head** (kNN-LM long-tail crisis, arXiv:2503.22426) — i token rari sono dove l'unembedding esatto conta. (4) latenza-gather non nascosta → align cache-line (64B) + software-prefetch.

**PROBE CHEAP RIVISTA per K1 (il punto chiave: NON è testabile a sandbox-scale):** microbench C su Zen2 = costruisci un memory-layer con value-table **deliberatamente > L3** e misura (a) gather batch-1 m=16-32 righe allineate **con vs senza software-prefetch** (la latenza si nasconde?), (b) confronta tok/s vs un FFN denso di pari qualità che eccede L3. Se il prefetch nasconde la latenza E batte lo streaming denso → K1 è la leva scale-up. È un microbench, non un training T4.

## Agente 2 → K2 PROMOSSO (leva near-term) + B3 diffusion UCCISO

**K2 (profondità-come-ricorrenza) = la scommessa CPU più forte, DECISIVO — e ancora di più su backbone SSM.** Cosa esiste: ALBERT (sharing, arXiv:1909.11942), Universal Transformer (sharing+halting adattivo, arXiv:1807.03819), DEQ (punto-fisso di UN blocco, arXiv:1909.01377), CALM/PonderNet (early-exit ~3× decode), e **MoR — Mixture-of-Recursions (arXiv:2507.10524, NeurIPS'25)** = lo SOTA che fonde blocco-condiviso-riusato + router per-token che sceglie la profondità di ricorsione, ~2× throughput a parità di accuratezza = **lo scheletro architetturale giusto da portare sul nostro substrato Mamba con pesi ternari/int8.**

**VERDETTO BANDA (la condizione È tutto il gioco):** il weight-tying NON riduce i byte/token da solo. Paga **solo se il blocco condiviso STA IN CACHE** — dopo la prima lettura-DRAM, i K riusi sono serviti da L2/L3, e allora **il fattore di sharing K diventa un moltiplicatore quasi-lineare della banda effettiva.** Vincolo duro e quantificabile: un blocco a d≈768 4×FFN ≈ 7M param = ~14MB fp16 (sfora L3) → **a int8/ternario 3.5-7MB = comodamente L3-resident** → i K riusi costano ~1 lettura-blocco ammortizzata su K. **Più naturale per SSM che per attention** (niente KV che cresce, ricorrenza sequenziale bandwidth-friendly, set-pesi per-step piccolo e fisso = esattamente ciò che vuoi pinnare in cache e iterare). Adaptive-K (DEQ/MoR/CALM) = **puro upside su CPU**: i token che convergono presto smettono di rileggere → ogni "strato di pensiero" extra legge **ZERO byte nuovi da DRAM.**

**IL RISULTATO APERTO E NOSTRO (la disuguaglianza che nessuno scrive):**
> Un blocco weight-tied/ricorsivo/DEQ riduce i byte/token-DRAM di fattore K **solo se** `sizeof(blocco, alla precisione scelta) ≤ L3` (idealmente L2-tileabile); allora `tok/s ≈ eff_BW / (byte × block_size) × K_medio`.

Riformula "quanti strati condividere" come **problema di co-design del budget-cache.** Nessuno benchmarka un blocco **SSM** weight-tied, low-precision, cache-resident, con halting adattivo, su CPU no-VNNI. = **risultato ownable**, e MoR è lo scheletro.

**Failure modes K2:** (1) **DEQ naive costa decine di NFE** (più lento per-token!) → serve K piccolo e bounded (C-DEQ arXiv:2602.03024 → 1-8 NFE); senza cap, DEQ perde su CPU. (2) il tying puro come generatore sotto-performa la profondità untied a pari compute → i guadagni vengono CON la profondità adattiva (UT/MoR), non dal tying da solo. (3) **cache-residency cliff**: se il blocco non sta in cache, ogni riuso ri-streama DRAM → la tesi collassa a "stessi byte, qualità peggiore". Sizing non opzionale.

**B3 (Diffusion-byte) UCCISO per il nostro regime — pulito.** La tesi "leggi-pesi-una-volta-raffina-K-posizioni" è fatalmente minata: (1) step-count × lettura-pesi-piena mangia il guadagno (serve K-step ≪ L-token; la diffusion standard vuole centinaia-migliaia di step, il few-step SOTA NFE≈8-32, e la qualità degrada tagliando step); (2) **l'attention bidirezionale uccide il KV-cache** → ricomputa O(L²) ogni step. Il paper di profiling (arXiv:2510.04146) lo conferma: gli AR scalano meglio, i DLM vincono solo small-batch SE gli step sono tagliati aggressivamente. **Per NOI è il caso peggiore**: la diffusion **butterebbe via** il vantaggio KV-free / no-quadratic dell'SSM sostituendolo con K passaggi bidirezionali pieni. Batch-1 single-core bandwidth-bound = caso peggiore diffusion, migliore AR-SSM. (Bonus: arXiv:2601.12979 = i DLM sotto-performano su workflow agentici/tool = proprio il regime che vogliamo.) → **B3 in trappola.**

**SEQUENCING INSIGHT (mio, dai due report insieme):** K2 è **testabile e benefico GIÀ a sandbox-scale** (il blocco condiviso sta in cache, sharing = moltiplicatore banda subito). **K1 NON è testabile a sandbox-scale** (modello intero in L3) → è leva SCALE-UP. Quindi: **K2 = leva near-term probeable; K1 = leva scale-up.** Si compongono: K2 = nucleo-compute cache-resident; K1 = tier-memoria grande prefetchato; K3 = obiettivo che predice gli slot per il prefetch di K1 *e* può informare l'halting adattivo di K2.

## Agente 3 → K3 (obiettivo CPU-aware): **è la WHITE SPACE — confermato il più originale — ma condizionale (2 contro-argomenti duri)**

**La decomposizione (decisiva — K3 non è un'idea, sono 5 con novità diversissima):**
| Sotto-idea | Stato |
|---|---|
| Sparsità d'attivazione da-zero | **RISOLTO** (Q-Sparse arXiv:2407.10969, dReLU/TurboSparse 2406.05955) → adottare diretto |
| Predire il set-attivo successivo per prefetch | **PUBBLICATO** (Deja Vu 2310.17157, Pre-gated MoE ISCA'24) — ma predittore su modello **FROZEN** |
| Allenare il routing a essere temporalmente coerente | **PUBBLICATO per MoE/DRAM-offload** (ReMoE 2605.27081, Oracle-MoE ICML'25) — fine-tune, GPU/flash |
| Co-trainare l'indice di retrieval | **PUBBLICATO** (ScaNN anisotropic 1908.10396, TRIME 2205.12674) |
| **Regolarizzatore di predicibilità che il MODELLO paga col gradiente + termine costo-cache L2/L3, da-zero, su SSM denso CPU** | **GENUINAMENTE NUOVO = la white space** |

**La white space, precisa:** nessuno allena **il modello stesso** (vs un predittore bolt-on) perché il **set-attivo del prossimo token sia predicibile dallo stato corrente**, via un **predittore cheap co-trainato la cui accuratezza è NELLA loss**, + un **penalty di turnover/working-set calibrato su L2/L3**, su un **SSM denso per CPU single-core**. La sintesi — *"pesi branch-predictor-friendly come proprietà ALLENATA di un SSM CPU-bound"* — è impubblicata. **Conferma: K3 è il pezzo più nostro.**

**Formulazioni concrete da adattare (rifiniscono Probe-3):** (1) **predittore** `g_φ(h_t)→ŝ_{t+1}` (MLP low-rank Deja Vu), `L_pred = BCE` nella loss **TOTALE, gradiente nel backbone** (non solo φ) = converte "misura" in "ottimizza"; (2) **coerenza temporale** (ReMoE): `L_reuse` (massa di prob sul set del token precedente, stop-grad) + `L_smooth` + `L_WS` (penalty union dei set su finestra vs capacità L2/L3), ancorato con **Trust-KL** a un reference non-regolarizzato; (3) **sparsità** Q-Sparse top-K+STE = **CREA l'oggetto discreto "quali-pesi"** (un SSM denso non ne ha uno finché non lo sparsifichi); (4) **costo-cache** stile HALP (penalty sale quando il set supera L2 poi L3).

**I 2 CONTRO-ARGOMENTI DURI (li leggo io, riscrivono il probe — onestà cost-first):**
1. **La predicibilità potrebbe essere GIÀ alta.** Deja Vu riporta **93-99% di accuratezza del predittore su modelli NON-regolarizzati.** Se il collo è *nascondere la latenza col prefetch async* (non l'accuratezza), allora **allenare per la predicibilità aggiunge poco.** = il contro-argomento più forte → **MISURARE la baseline PRIMA di aggiungere il termine.**
2. **Stesso muro-di-scala di K1: a sandbox (~5M = 5-20MB) il modello sta in L3 → prefetch/cache-residency MOOT.** Paga solo quando modello/tier-memoria eccede L3.

**Ordine-pipeline obbligato (load-bearing):** **sparsifica (Q-Sparse) → rendi coerente (ReMoE) → rendi predicibile (regolarizzatore novel).** Senza sparsità non c'è oggetto discreto da prefetchare. E STE+predicibilità+costo = tiro-alla-fune a 3 → stadiare e annealare. **Vantaggio del nostro setting (B=1 single-core): il costo load-balance/CV della pressione di località è IRRILEVANTE** (ReMoE: CV +293% ma per noi non conta) = la coerenza ci costa meno che agli altri.

---

# FASE B — Architetture-intere (anteprima, da approfondire)

**Ancore HW per le stime:** Zen2 R5 3600X — DRAM ~45 GB/s, L3 ~200 GB/s eff, latenza DRAM-random ~100ns, ~16-22 miss L2 in volo (MLP). Ternario ≈ 0.25 B/param (1.58bit + packing). *Numeri = stime da modello, da confermare a microbench.*

### B2 — "Recursive-SSM" (= K2/MoR sul substrato Mamba) — **la near-term, concreta**
Architettura: UN blocco SSM ternario (SSM-proj + gated-MLP), riusato in profondità K volte con **halting adattivo per-token** (router MoR / early-exit: facile→K=2, difficile→K=8); head + embedding tied.
- **Sizing (load-bearing):** blocco D=512 ≈ 3M param (SSM ~1M + MLP ~2M) → ternario ~0.6-0.75MB = **L3-resident, borderline L2**. A D=384 ≈ 1.7M → ~0.4MB = **L2-resident**.
- **Il colpo di banda:** weight-tied → il blocco resta in cache **per TUTTA la generazione** → traffico-pesi-DRAM ≈ **0 dopo il warmup** → il decode è **compute-bound, non bandwidth-bound.** Le iterazioni adattive K leggono **zero byte nuovi da DRAM** — "pensare di più" è gratis in banda.
- **Byte/token DRAM ≈ ~0** (pesi cache-resident). tok/s limitato dall'ALU: K_medio × MAC-blocco (~3M MAC × K≈3 ≈ 9M MAC/token) + costo-scan SSM (l'exp vettorizzato già risolto). Resta nel regime **compute-bound** che già godiamo a 1.46M (~3200 tok/s), **ma scala la profondità senza cadere bandwidth-bound** — cosa che un modello untied profondo NON può (a scala i suoi L blocchi distinti sforano L3 → muore in banda).
- **Slogan B2:** *"compra profondità senza pagare banda."* Originalità = blocco **SSM** ternario cache-resident + halting adattivo su CPU no-VNNI, con `tok/s ≈ banda/(byte×block) × K_medio` — impubblicato. Scheletro = MoR (arXiv:2507.10524).
- **Rischio:** DEQ-puro costa decine di NFE → serve K **bounded e piccolo** (cap stile C-DEQ); il tying-puro come generatore sotto-performa → il guadagno viene dall'halting adattivo, non dal tying da solo.

### B1 — "Memory-LLM" (= B2-core + tier product-key) — **la scale-up**
Architettura: il core B2 (pesi-DRAM ≈ 0) + un tier di **product-key memory** grande in DRAM al posto dell'FFN denso, + retrieval-head sullo stesso indice, + I/O a entropia (K4) + MTP (K5).
- **Sizing memoria:** N=4M slot, Dv=256 @ int8 → **tabella 1 GB in DRAM**; attivi m=32 righe × 256B = **8 KB gatherati/token** (random, **prefetchati**); sub-key 2√N=4000 × 256B = **1 MB cache-resident** (scanditi/token).
- **Byte/token DRAM ≈ 8 KB** (solo il gather) vs un denso di pari **capacità 1GB** = 1 GB/token = morto. → **~5 ordini di grandezza** (esattamente il claim UltraMem batch-1: accesso = m×Dv, **indipendente da N**).
- **Costo memoria-tier:** sub-key scan 1MB@L3 ≈ **5µs** + gather 8KB → **~0.6-1µs SE prefetchato** (lo stato SSM predice gli slot, K3), **~13µs se serializzato senza prefetch**. → **~6µs/token/memory-layer prefetchato.** Il prefetch è il make-or-break (= failure-mode #4 di K1).
- **Risultato:** un modello a **capacità 1GB** che decoda a **migliaia di tok/s su un core Zen2** = il titolo. Disaccoppia capacità (DRAM, enorme) da banda (set-attivo piccolo, cache-resident). = materializzazione del north-star recall + weight-streaming.
- **Quando ha senso:** SOLO a scala (la tabella DEVE eccedere L3, sennò K1 non morde). È l'architettura del modello grosso agentico, non della sandbox.

### B3 — "Diffusion-byte" — **SCARTATA** (vedi Fase D / traps): su CPU batch-1 perde, butta via il vantaggio SSM.

---

# FASE C — Trapianti cross-domain (sketch, da approfondire con deep research)

Strumenti totalmente non-LLM applicati al "rendere f(context) cheap su CPU", con verdetto:
- **Matrici strutturate (Monarch / butterfly / Benes):** mixing/proiezioni via permutazioni strutturate SIMD-friendly = meno byte E accesso sequenziale. **KEEPER — rinforza K2** (il blocco condiviso può essere Monarch invece di denso → ancora più cache-tileabile). Già flaggato nel bandwidth-report.
- **Branch-predictor / prefetcher (architettura CPU):** → **load-bearing per K1** (il prefetch nasconde l'unica debolezza di product-key memory). Già Finding 7.
- **FFT / FNet (mixing parameter-free):** mix a **zero-peso** (allettante in banda). MA FFT causale + streaming per-token su AVX2 è scomoda (butterfly non-SIMD-banale, non-causale per natura). **Verdetto: subsumed** — l'SSM **è già** un mixer strutturato parameter-light e causale; cattura il beneficio senza la FFT. Marginale.
- **Coding aritmetico / range-coder (compressione):** → **K4** (head/tokenizer come coder a entropia). DNA SEE. Tenuto.
- **Automi cellulari / reservoir / sistemi dinamici:** mixing per regole locali, accesso ultra-locale (cache-perfetto). **Verdetto: un CA-trainabile È sostanzialmente un conv-SSM locale** — già lo abbiamo. Basso marginale (e l'era-mantra ha mostrato il muro addestrabilità del reservoir frozen; ora con backprop ≈ SSM).
- **Sketch / Count-Min / Bloom (streaming algorithms):** stato compresso probabilistico **frequency-based** (≠ VSA/HXI, morto per capacità di *superposizione* — questa è un'altra famiglia). **Nicchia minore possibile:** hot-cache di n-grammi/token frequenti residente in L2 (recall cheap dei pattern caldi). Da tenere a mente, non top.
- **Quantizzazione vettoriale / codici (Walsh-Hadamard, Gray):** ternario-LUT, codici-pesi, PQ. Asse già aperto (bandwidth-report + recall tier).
- **Energy-based / Hopfield moderno / ottimizzazione:** decode-come-minimizzazione-energia = la "follia" di A9. Speculativo, parcheggiato.

**META-VERDETTO FASE C (onesto, anti-Goodhart): lo sweep cross-domain NON produce un nuovo top-candidate — INDURISCE i keeper esistenti** (Monarch→K2, prefetcher→K1) e svela una sola nicchia minore (frequency-sketch hot-cache). Questo è informativo, non deludente: significa che la tesi unificante (K1-K5) è il vero baricentro, non un caso locale. Non fabbrico novità dove non c'è.

---

# Keeper traps (scartati con motivo — non riproporre)

- **VSA/HXI superposition** per memoria O(1): MORTO (capacità ~D/16, ortogonale non batte random — finding Phase 51 fondamentale).
- **Sparsità non-strutturata / pointer-chasing**: accesso random = il muro latenza, non la vittoria.
- **Tensor-Train / core condiviso a rango bassissimo**: capacità ammazzata (Beta archiviato).
- **Pesi entropy-coded decodificati-on-stream**: il decode seriale sul critical-path di solito perde contro "meno bit" (ternario). Tenuto come nota SEE, non inseguito.
- **Diffusion / non-autoregressivo (B3)**: su CPU batch-1 bandwidth-bound PERDE (K passaggi pieni × no-KV-cache ≥ L letture cheap AR), e butta via il vantaggio KV-free dell'SSM. Verificato (arXiv:2510.04146). Chiuso per il nostro regime.

---

# FASE E — Shortlist rankizzata + probe cheap (cost-first)

**Ranking (leva × maturità × CPU-fit × originalità):**
1. **B2/K2 — Recursive-SSM** (blocco ternario cache-resident condiviso + halting adattivo). **Leva near-term**, testabile GIÀ a sandbox-scale, lever di banda decisivo, scheletro MoR. = il prossimo colpo naturale dopo run-6.
2. **K1 — product-key memory + retrieval-head fuse** (scale-up). Leva massima a scala + **più originale** (la fusione FFN+head+recall in un indice), ma morde solo se memoria > L3.
3. **K3 — obiettivo CPU-aware** (regolarizzatore di predicibilità). Il collante-meta, il più originale, meno coperto (ricerca in corso). Abilita il prefetch di K1 e l'halting di K2.
4. **K4 — I/O a entropia** (patch BLT + head aritmetica). SEE-native, leva media. Dopo.
5. **K5 — MTP self-drafting.** Moltiplicatore noto a livello-stage. Dopo.

**Probe cheap (una variabile, no-T4 dove possibile, NON forka run-6):**
- **Probe-1 (K2, testabile ORA a sandbox-scale):** allena un blocco SSM **weight-tied riusato K volte** vs baseline untied K-layer a param-matchati → misura **BPB (costo-qualità del tying)** + tok/s. Poi aggiungi **halting adattivo** → misura K_medio + BPB. Domanda secca: *tying + halting tengono la qualità?* PyTorch piccolo + motore C esistente. Cheap (≤1 run T4 o anche CPU).
- **Probe-2 (K1, microbench C, NO training):** su Zen2, gather di m=32 righe-valore allineate da tabella **> L3**, **con vs senza software-prefetch** → µs/token; confronta con streaming FFN-denso di pari qualità che eccede L3. Domanda secca: *il prefetch nasconde la latenza E batte il denso?* Ore.
- **Probe-3 (K3, diagnostico — riscritto dall'Agente 3):** prerequisito = un SSM con **sparsità allenata** (Q-Sparse top-K → crea l'oggetto "quali-pesi"). Poi, **PRIMA di ogni regolarizzatore, MISURA la baseline:** un predittore cheap `g_φ(h_t)` predice il set-attivo del token successivo — *quanto è già accurato?* (Deja Vu dice 93-99% su modelli non-regolarizzati → **se è già alto, K3 NON serve, il collo è il latency-hiding**, non l'accuratezza). SOLO se la baseline lascia gap → aggiungi `L_pred` (gradiente nel backbone) + coerenza ReMoE e vedi se sale **senza** alzare la BPB. = l'esperimento originale di Finding 7, ma **make-or-break = il numero baseline.** Viene dopo Probe-1 (che produce il modello sparsificabile). NB: morde solo a scala (modello > L3).

**Ordine:** Probe-1 (K2) e Probe-2 (K1) sono **cheap, indipendenti, paralleli**. Probe-3 (K3) viene dopo Probe-1. **Disciplina: niente di tutto questo forka la rampa run-6**; si apre quando si apre il design scale-up. L'unica cosa testabile cheap SUBITO senza aspettare = Probe-1 (K2, sandbox) + Probe-2 (K1, microbench).

---

# Prossime iterazioni (todo)

- **Fase D (deep research, targeted — NON spray):** verificare originalità + evidenza CPU di K1 (product-key memory su CPU; esiste un'impl cache-resident? fast-scan?), K2 (DEQ inferenza CPU; early-exit bandwidth), B3 (diffusion-LM throughput CPU vs AR), A4-parameter-free-mixing (FNet causale streaming). Failure modes.
- **Fase B/C espanse** con stime byte/µs per architettura.
- **Fase E:** shortlist rankizzata + probe cheap (una variabile, no-T4, NON forka la rampa recall run-6).

**Disciplina:** tutto questo è asse SCALE-UP / esplorazione, NON tocca la rampa run-6 in corso. Probe locali cheap quando si apre il design del modello grosso.
