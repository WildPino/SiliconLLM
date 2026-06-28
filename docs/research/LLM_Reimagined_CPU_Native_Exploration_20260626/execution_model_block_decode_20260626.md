# L'execution model: block-decode layer-major (idea utente, 2026-06-26)

> **Idea utente:** invece di streammare tutti i pesi dalla RAM per OGNI token, costruisci uno strato fatto apposta per stare in L1/L2/L3 + il contesto; caricalo, la CPU calcola una "matrice" (non un token), salva il risultato in cache, intanto fai swap dello strato successivo; ripeti strato per strato fino in fondo. Questo NON genera 1 token ma **TUTTA la risposta insieme** (o un blocco di ~500, poi fonde col contesto e continua finché non ha concluso).
>
> **Verdetto Architetto: è la LEVA PIÙ IMPORTANTE di tutta l'inferenza — l'argomento roofline / arithmetic-intensity — ri-derivato da zero. È il KEYSTONE che sblocca tutte le altre leve. Il pezzo nuovo per noi: applicarlo al decode su CPU, ed è PIÙ naturale su SSM che su transformer.**

---

## 1. Perché è giusto: la matematica del roofline

Il decode batch-1 è lento per UNA ragione quantificabile: il **riuso dei pesi = 1**. Carichi un peso dalla RAM, lo usi una volta, lo butti.

Misura chiave = **operational intensity** `I = FLOP / byte-letti`.
- **Matvec (1 posizione):** `y = Wx`, W di `d×d`. FLOP `= 2d²`, byte `= d²·b` (b = byte/peso). → `I = 2/b`. A ternario (b≈0.25) → **I ≈ 8 FLOP/byte = sotto il ridge → bandwidth-bound.**
- **Matmul (N posizioni insieme):** `Y = WX`, X di `d×N`. FLOP `= 2d²N`, byte pesi `= d²·b` (caricati UNA volta per tutte le N). → `I ≈ 2N/b`. A N=500, b=0.25 → **I ≈ 4000 FLOP/byte = ben sopra il ridge → COMPUTE-bound.**

**L'operational intensity scala con N.** Processare N posizioni per ogni caricamento-pesi **ammortizza i byte-pesi N volte** → attraversi il "ridge point" del roofline → **da bandwidth-bound diventi compute-bound.** Questo è *esattamente* ciò che la tua idea fa, ed è il motivo per cui il prefill (che processa tutti i token del prompt in parallelo) è veloce mentre il decode (1 token/volta) è lento. **La tua domanda implicita — "possiamo far sembrare il DECODE un PREFILL?" — è LA domanda giusta.**

Il dataflow che descrivi è anche corretto in ogni dettaglio:
- **layer-major** (strato per strato, pesi caricati una volta ciascuno) invece di token-major (tutti gli strati per token, pesi ricaricati per token) = l'ordine bandwidth-ottimale;
- **attivazioni residenti in cache** (la "matrice" delle N posizioni = `N×d`, ~0.5MB a N=500/d=512/fp16 → sta in L2/L3);
- **double-buffering** ("intanto un altro swap"): mentre la CPU calcola lo strato L (pesi in cache), prefetch dello strato L+1 dalla RAM → nascondi la latenza-pesi dietro il calcolo. E a N grande **c'è N× più calcolo per nascondere il fetch** → il prefetch si nasconde facile.
- **memoria attivazioni = O(N·d)** non O(N·d·L): tieni solo lo strato corrente e sovrascrivi (la tua intuizione "salva il risultato, swap il prossimo" gestisce già lo spazio correttamente).

## 2. L'unico problema duro: la causalità

In autoregressione pura **non puoi avere N posizioni indipendenti**: il token t+1 dipende dall'OUTPUT del token t (che richiede di aver fatto tutti gli strati per t). La dipendenza è "tutti-gli-strati-per-token-t → input-token-t+1" = ordine **token-major**, l'OPPOSTO del layer-major che vuoi. **Tutto il gioco è: come ottieni l'ordine layer-major malgrado la dipendenza token-major?** Le risposte reali (e la tua idea è una di queste):

- **Speculative / MTP (= la tua "stampa 500, fondi, continua"):** un draft cheap propone N token (token-major ma economico), poi li **VERIFICHI tutti in UNA passata layer-major** (ogni strato caricato una volta, applicato a tutte le N posizioni). Accetti il prefisso corretto, committi, slitti la finestra. = block-parallel decode. È reale e deployato (Medusa/EAGLE/MTP). **Hai re-inventato la verifica speculativa batchata.**
- **Jacobi / Lookahead decoding:** indovina N token, falli passare TUTTI in parallelo (layer-major), ogni posizione raffina la stima, itera al punto fisso. Converge in <N passi → N token in <N passate. Nessun draft separato.
- **Diffusion / non-autoregressivo:** genera tutte le N posizioni in parallelo, raffina iterativamente (vedi §5 — la tua idea RIAPRE questo verdetto).

## 3. Il soffitto ONESTO (quanto guadagni davvero)

NON è N× (non puoi generare 500 token corretti in un colpo — il token 500 dipende genuinamente dal token 1). Il guadagno reale = **lunghezza di accettazione/convergenza `a`** (tipicamente **2-8×**, fino a ~10× con draft eccellenti/MTP). Contabilità banda: una passata layer-major sulle N candidate carica i pesi UNA volta e avanza di `a` token committati → **weight-stream per token committato = 1/a** (vs 1 in AR) → **speedup ≈ a**.

E il punto chiave: **il calcolo "sprecato" sulle N−a posizioni non-accettate è GRATIS** — riempie cicli-ALU che stavano in stallo sulla banda. Vale fino a quando l'extra-compute (N/a ×) tocca il tetto-compute.

> **IL KEYSTONE (la sintesi che conta): il block-decode è ciò che ti SPOSTA da bandwidth-bound a compute-bound — ed è esattamente lo spostamento che fa PAGARE tutte le altre leve.** Multi-core, cache-residency, core-liberi/helper: nel regime bandwidth-bound aiutano poco; nel regime compute-bound scalano. **La tua idea è la chiave che apre la porta alle altre.** Senza, sei bandwidth-bound e i core extra annaspano; con, sei compute-bound e i core scalano.

## 4. Il vantaggio SSM-specifico (perché è PIÙ naturale per NOI)

Per un **transformer**, verificare un blocco di N posizioni costa **O(N²)** (attention su tutte le coppie) + KV-cache. Per un **SSM**, lo stato è O(1) e processare N posizioni in parallelo si fa con lo **scan associativo parallelo — LO STESSO che usiamo in TRAINING** — in **O(N)** (o O(N log N)). → **la "matrice" che calcoli per strato È lo scan parallelo sul blocco**, ed è nativo del nostro backbone. **Il block-decode è più naturale per un SSM che per un transformer.** Questo è un vantaggio reale e poco sfruttato: il nostro modello è già costruito per la modalità a-blocco (è la modalità prefill/training).

## 5. La riapertura onesta: e la diffusion che avevamo ucciso?

La tua riformulazione mi obbliga a riaprire un verdetto. Avevamo **ucciso la diffusion/NAR** su CPU per due motivi: (1) molti step di refinement × lettura-pesi-piena; (2) attention bidirezionale → niente KV-cache → ricomputo O(N²)/step. **MA il motivo (2) è TRANSFORMER-specifico.** Un SSM non ha attention da ricomputare: un blocco bidirezionale SSM (scan avanti+indietro) resta **O(N)/step**, non O(N²). Quindi K step few-step (8-32) su un blocco di 500 = **O(K) per token**, e su un SSM il costo-per-step è cheap (scan parallelo). → **la diffusion-su-SSM-a-blocco merita un ri-esame onesto sull'asse BANDA** (il rischio aperto resta la qualità few-step, ma l'argomento-killer di prima non si applica al nostro backbone). Non la resuscito come keeper, ma la sposto da "morta" a "da ri-valutare in questo frame".

## 6. Cosa significa per l'architettura: NON un nuovo keeper, il KEYSTONE che unisce i keeper

Il block-decode non è una 6ª idea — è l'**execution model** che fa coerere quelle che abbiamo:
- **K2** (strato ternario cache-resident) = *ciò che streammi* layer-major;
- **K5** (MTP/draft) = *come ottieni* le N candidate da verificare;
- **helper-prefetch / core-liberi (R-D)** = *double-buffer* dello strato L+1 mentre calcoli L;
- **K1** (product-key memory): batchando N posizioni, i pesi-routing si caricano una volta per blocco (ammortizzati) e i gather dei valori si dedupano tra posizioni → la memoria sparsa beneficia, anche se meno del denso;
- **K3** (predicibilità) e **R-E** (manutenzione async) restano ortogonali.

**Il block-decode è il telaio; K1-K5 sono i componenti montati sopra.** È la risposta operativa allo slogan "rifattorizza l'LLM": *non* generare un token alla volta, genera **a blocchi, layer-major, attivazioni-residenti, double-buffered** — e su un SSM è la modalità nativa.

## 7. Soffitto e probe

**Soffitto onesto:** speedup ≈ `a` (lunghezza accettazione, 2-8×) **moltiplicato** per il fatto che ti porta nel regime compute-bound dove multi-core e cache-residency aggiungono il loro. Combinato, è il pezzo che fa la differenza grossa a scala — ma vive nel regime bandwidth-bound (= modello grande; alla sandbox attuale, già compute-bound, è muto).

**Probe cheap (quando si apre):** (1) microbench roofline — matmul `d×d × d×N` ternario-LUT su Zen2 al variare di N: misura dove N attraversa il ridge (bandwidth→compute) e il tok/s effettivo; (2) misura la lunghezza di accettazione `a` di un draft cheap (n-gram/MTP) sul nostro modello → il soffitto reale dello speedup; (3) costo dello scan-parallelo-a-blocco SSM (già lo abbiamo in training) riusato in inferenza.

**Da ricercare (lancio):** block/Jacobi/Lookahead decoding e i loro `a` reali; block/chunked decode nativo su SSM (è il fit naturale che credo?); la ri-valutazione diffusion-su-SSM sull'asse banda; il framing roofline per decode CPU.

---

## 8. R-F — verifica (2026-06-26): TUTTO confermato + 1 crux SSM nuovo + 1 regalo

**Lo split (il trap principale):** metà-sistema (layer-major, cache-resident, ammortizza su N) = corretta e supportata. Metà-algoritmo (un blocco di N) = quelle posizioni NON ESISTONO ancora → vanno FABBRICATE (speculazione/Jacobi/diffusion). **Il win roofline è governato dalla lunghezza ACCETTATA `a`, NON da N. Pianifica su `a`, non su N.**

**Q1 roofline — CONFERMATO E MISURATO SU CPU:** intensità decode ≈1 op/byte, prefill ≈1000+, ridge ~160 (GPU). **llama.cpp FP16 batch 1→32 = 16.9→106.4 tok/s (~6.3×)** mentre il prefill non si muove. **Conferma il keystone misurato: "a batch-1 più core sono già memory-bound, il GEMV non dà nulla — SOLO il blocco alza l'intensità, non i thread."** E serve poco: **M≈8-32 basta** (devi salire ~2 ordini, l'intensità scala lineare con M).

**Q2 `a` ONESTO (correggo il mio "2-8" → 2-4):** Blockwise-Parallel (Stern'18) accettati 4.7-5.3, 2× lossless / 3.3-4× rilassato; Lookahead/Jacobi 1.5-1.8× (esatto); EAGLE-3 2.9-4×; MTP decade ripido. **`a` è bounded da decadimento GEOMETRICO con la profondità del draft** (accuratezza per-posizione si moltiplica) → **`a` cluster a 2-4 a prescindere da N.** Costruisci il loader per N≈8-16, committi `a`≈2-4. Ammortizza comunque 2-4× = moltiplica direttamente il tok/s.

**Q3 SSM-nativo — CONFERMATO il fit migliore, con UN crux nuovo:**
- **Lo scan parallelo È il block-verify:** Mamba-2 SSD chunked (chunk Q≈64-256) = matmul weight-reuse-heavy (il win roofline GRATIS); verificare N posizioni = un chunk dello scan. Block-pass O(N·d²) vs transformer O(N²·d); stato fisso → il blocco non gonfia la memoria → **ideale L2-resident su Zen2.** Mamba-drafter: costo/token costante, `a`=2.80@8K vs Mistral 2.21 con meno memoria.
- **IL CRUX SSM-SPECIFICO (non l'avevo anticipato): rollback dello STATO su accept-parziale.** Un transformer annulla un draft rifiutato **troncando il KV (gratis)**; un SSM comprime la storia in UNO stato fuso → su accept-parziale devi **recuperare lo stato all'ultima posizione accettata** (non puoi sottrarlo). **Fix (cheap): durante lo scan a-blocco gli stati per-posizione sono GIÀ prodotti → CHECKPOINTALI → il rollback diventa un movimento-di-puntatore, non un ricomputo.** (Mamba-in-the-Llama 2408.15237 = kernel multi-step che ritorna gli stati intermedi; SpecMamba 2509.19873.) **= la cosa #1 da azzeccare nel kernel C.**
- **IL REGALO: la verifica ad ALBERO è naturalmente incompatibile con lo stato ricorrente** (ogni ramo = traiettoria-stato diversa) → serve verifica **lineare/FIFO**. **La nostra ricetta-decode lockata (rep1.2/win128 greedy, NO top-p) FAVORISCE GIÀ la verifica lineare** → un asset gratis, non un vincolo.

**Q4 diffusion-su-SSM — risolto con precisione:** il motivo-killer (b) O(N²) **È transformer-specifico e si dissolve** (DiffuMamba 2511.15927: BiMamba-2 denoiser, 8.2× throughput@65K, crossover ~2K token). MA il motivo (a) **step-count NON si dissolve**: K=L/p step = migliaia a L lungo, ognuno una passata-pesi piena → su CPU bandwidth-bound **lo step-count moltiplica il traffico-pesi.** **Verdetto: l'SSM toglie l'obiezione quadratica ma NON quella sullo step-count, ed è quest'ultima a governare la banda.** Plausibile SOLO few-step (≤4-8) su blocco bounded cache-resident = **collassa in "block-parallel decode con ≤3 passate-correttore".** → **NON resuscitare la diffusion piena; RUBA la sua idea di block-refinement come correttore 1-3 passate sopra lo speculative block-decode.**

**Composizione (confermata):** × K2 (lo strato cache-resident è il consumatore naturale; complementari: blocco alza il RIUSO, residency abbassa la LATENZA-al-peso) · × K5 (MTP = il FORNITORE del blocco; allena la testa MTP perché le posizioni siano scan-verificabili in un chunk) · × R-D (double-buffer sul SMT-sibling, ma il win DUREVOLE è il blocco che alza l'intensità — silicon-independent — non il prefetch, che è Zen2-only).

**Failure modes (ranked):** (1) confondere N con `a` (speedup=`a`); (2) **rollback stato SSM su accept-parziale = il pericolo definitorio** → checkpointa gli stati dello scan; (3) verifica-albero su stato ricorrente → resta lineare/FIFO; (4) blocco troppo grande per L2 (le attivazioni N×d devono stare in cache); (5) blowup step-count diffusion; (6) decadimento geometrico accettazione → tara la profondità draft al ginocchio.

**Probe rivisto:** (1) microbench roofline matmul ternario-LUT al variare N (dove attraversa il ridge); (2) **misura `a`** di un draft cheap sul nostro modello (il soffitto reale); (3) **prototipa il checkpoint-stato-per-posizione nello scan a-blocco** (il crux); verifica lineare (già la nostra ricetta). Tutto microbench, no-training pesante.

**Fonti:** roofline 2402.16363 · Blockwise-Parallel 1811.03115 · Lookahead 2402.02057 · CLLMs 2403.00835 · Mamba-2 SSD 2405.21060 · Mamba-drafters 2506.01206 · SpecMamba 2509.19873 · Mamba-in-Llama 2408.15237 · DiffuMamba 2511.15927 · Set-Block-Decoding 2509.04185 · CPU-batching 2501.00032.
