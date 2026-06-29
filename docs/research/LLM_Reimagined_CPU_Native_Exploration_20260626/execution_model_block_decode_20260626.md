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

---

## 9. R-G — Il "concetto a-priori" (idea utente "scultore"): ridimensiona la qualità, MA risolve il gap-larghezza del single-user

**Idea utente:** la risposta nell'uomo nasce parola-per-parola, ma il CONCETTO è a-priori → come uno scultore che ha l'idea prima e la rivela. Lettura mia = piano compatto a-priori (= codice compresso, DNA SEE), poi realizzazione (= decompressione).

**IL VERDETTO CHE RIDIMENSIONA (onesto): l'AR PIANIFICA GIÀ a-priori, provato causalmente.** Anthropic "Biology of an LLM" (Claude sceglie la parola-rima a FINE riga PRIMA di scrivere la riga — provato sopprimendo/iniettando) + Future Lens (2311.04897: lo stato a posizione t codifica linearmente i token a t+2+). → **rendere il piano ESPLICITO NON migliora la prosa** (il modello pianifica comunque). Tutta la famiglia concept-first (LCM 2412.08821, Coconut 2412.06769, SoT 2307.15337, Semformer, SentenceVAE) **non batte un AR token-level di pari taglia su qualità open-ended** — i win sono in parallelismo/latenza, efficienza-reasoning, multilingue. **Quindi: il piano esplicito NON è una leva di qualità.**

**MA — IL REGALO (la sinergia che chiude un gap reale): il piano esplicito compra PARALLELISMO, ed è ESATTAMENTE ciò che manca al single-user nel block-decode.** Il gap noto: il 6.3× misurato era batch-32 multi-STREAM (server, 32 utenti); **un singolo utente che genera UNA risposta cattura solo `a`≈2-4×** (la speculazione lineare). **Un piano globale fisso rende le sezioni INDIPENDENTI → le realizzi in PARALLELO → quella è la larghezza-di-batch.** Il piano **converte la richiesta sequenziale di UN utente in un carico batchabile.** Implicito non si può parallelizzare (è impigliato nel residual stream); esplicito sì. = il piano è **infrastruttura di throughput, non leva di qualità.**

**Due regimi di costo (decisivo per CPU):**
- **(A) Piano-come-vettore-separato (LCM/Coconut/Semformer):** costa una passata-modello extra (LCM = diffusion-su-embedding = molte passate + decoder) → **su CPU bandwidth-bound PERDE.** Scartato.
- **(B) Piano-come-scaffold-cheap-poi-fill-parallelo (Skeleton-of-Thought, SentenceVAE):** piano corto + realizzazione **parallela** → **mappa DIRETTAMENTE sul telaio block-decode (R-F).** = il regime che sopravvive al roofline.

**IL FIT PULITO PER NOI (unifica tutto):** il piano = **skeleton corto, DISCRETO, cheap** (UNA passata SSM veloce, greedy, **rep1.2 = la nostra ricetta lockata**) → poi **block-decode delle sezioni in PARALLELO** sotto il piano fisso. Unisce **K2** (streama i pesi una volta) + **K5** (il piano FORNISCE il blocco, non un draft cieco) + **R-F** (verify layer-major) + dà al single-user la **larghezza** che il block-decode pretende. **Skeleton discreto evita il gather dei codici-vettoriali** (che R-B/R-C dicono drift-brittle/anti-banda) — coerente. SSM = nativamente latent-variable (lo stato È un riassunto compresso; Mamba già backbone di latent-diffusion).

**Failure modes:** (1) **extra-pass tax** (piano pesante = morto su CPU; solo cheap-plan/parallel-fill sopravvive); (2) **dipendenza inter-sezione** = il limite vero: il fill parallelo SI ROMPE quando la sezione N dipende dalla N-1 (matematica, codice, narrativa stretta); funziona su contenuto separabile/a-lista, non su prosa strettamente accoppiata; (3) il concetto non sta in un vettore (LCM serviva diffusion perché MSE-su-un-concetto sotto-determina); (4) drift dei codici-latenti appresi (= R-B, serve ℓ2-norm); (5) nessun win di qualità open-ended dimostrato.

**SINTESI (mia): la tua metafora-scultore è reale e pubblicata, MA la parte "pensa meglio" è già fatta dall'AR internamente (provato).** Ciò che resta da guadagnare NON è coerenza — è **parallelismo**, e il parallelismo è proprio il pezzo mancante che dà al block-decode la larghezza su singolo utente. → **reframe: il piano-a-priori è infrastruttura di throughput che converte una richiesta sequenziale in un carico parallelo batchabile.** Il pezzo discreto-cheap (SoT × SentenceVAE) è quello che sta nel roofline. **Limite duro: vale su contenuto separabile; la prosa strettamente causale non si parallelizza.** (R-H — lo scultore few-step su token — pending, completa il quadro.)

**Fonti:** LCM 2412.08821 · Coconut 2412.06769 · SoT 2307.15337 · Semformer 2409.11143 · SentenceVAE 2408.00655 · Anthropic Biology-of-LLM · Future Lens 2311.04897 · DiM 2405.14224.

---

## 10. R-H — Lo scultore in POCHI colpi (carving few-step): soffitto onesto `c≈3-5`, MA più SEMPLICE dell'AR su SSM

**Metrica decisiva:** `c` = token committati per passata-pesi-piena. AR ha c=1; un carve vince sse c>1 a qualità-AR.

**IL BOUND FONDAMENTALE (non una manopola): l'indipendenza condizionale.** L'unmasking parallelo approssima il giunto `p(x_S|ctx)` col prodotto delle marginali `∏p(x_i|ctx)`; se l'insieme S contiene posizioni mutuamente dipendenti → token ognuno localmente-probabile ma **congiuntamente incoerente.** I metodi recenti (DEMASK/DAPD/DOS) stimano un grafo di dipendenza e sbloccano solo un **insieme indipendente** per step. → **la larghezza parallela sicura/step è limitata dalla struttura di mutua-informazione locale del testo.**

**→ `c` È CONTENT-DEPENDENT, e qui sta il nostro edge:** prosa non strutturata → insieme sicuro ~2-4 token/step (= lunghezza-accettazione speculativa); **contenuto a BASSA ENTROPIA (codice, log, JSON, boilerplate, whitespace = la nostra Cat-A silicon-native) → larghezza MOLTO maggiore** → c grande. **Il carve paga di più ESATTAMENTE sui dati per cui il progetto è nativo** ([[project_silicon_native_data]] Cat-A).

**Il numero ONESTO:** **Set Block Decoding (SBD, 2509.04185, Meta) = `c≈3` affidabile** (fonde NTP+masked-prediction, no cambio-arch, KV-compatibile, fine-tune di un NTP esistente; **3.0-3.4× meno passate a qualità mantenuta, downstream 8B**). Diffusion non-distillata = c≈1 a qualità-AR (NFE≈lunghezza, no free lunch). Distillazione compra few-step: **FS-DFM 8-step = parità-perplexity con 1024-step (~128×) MA perplexity su modello PICCOLO, non reasoning 8B** → soffitto-demo, non `c` consegnato. **Bankable oggi = `c≈3-5`, NON 8-128.**

**Il contest carve-vs-speculative-AR:** il block-speculative-AR (R-F) dà già `c≈a≈2-4` a **qualità ESATTA** (il verify rigetta i draft cattivi). SBD pareggia (~3×) MA **senza draft separato e senza KV-cache** = decisivo su CPU. Few-step-diffusion 8-128× dominerebbe ma a qualità downstream non provata. → **il carve vince la gara-banda solo dove `c` è strutturalmente grande (Cat-A); sulla prosa entrambi collassano a ~2-4.**

**IL RIBALTAMENTO SSM (la cosa più bella): il carve DISSOLVE il crux dello state-rollback di R-F.** Lo speculative-AR su SSM richiede il checkpoint-stato-per-posizione per il rollback su accept-parziale (il rischio #1 del kernel C). **Un blocco diffusion/parallel-refine RICALCOLA l'intero blocco a ogni step → NON c'è NULLA da rollbackare → ri-scansiona e basta.** → **sul nostro SSM il carve è PIÙ SEMPLICE del percorso speculative-AR con cui compete.** La cosa che sembrava esotica è il fit più pulito. (E un SSM block-pass = scan bidirezionale O(N) no-KV = un weight-stream = il "colpo di scalpello" naturale.)

**NICCHIA APERTA:** SBD/FS-DFM/BD3-LM sono tutti **transformer**. SSM+discrete-diffusion è quasi solo vision. **Few-step set-block-decoding su backbone SSM testuale = combinazione essenzialmente non lavorata = territorio di ricerca**, e compone pulito con R-F + K2 (layer-major su blocco 16-64, pesi streammati una volta, cache-resident → il compute extra in-blocco è gratis).

**Failure modes:** (1) collasso indipendenza-condizionale (il bound duro, cap prosa ~2-4); (2) step-count creep (ripristinare qualità = più step = il win svanisce); (3) costo distillazione + validazione stretta; (4) commitment lunghezza-blocco (diffusion fissa la lunghezza; l'edit-based la flessibilizza ma oscilla); (5) uscire dal regime cache-resident (blocco grande → compute non più gratis → B≈16-64); (6) **decode-hygiene transfer: la ns greedy rep1.2/win128 va RI-VALIDATA per l'unmasking confidence-ordered** (non è più left-to-right).

**Fonti:** SBD 2509.04185 · FS-DFM 2509.20624 · BD3-LM 2503.09573 · Fast-dLLM-v2 2509.26328 · Esoteric-LM 2506.01928 · DEMASK/DAPD · MaskGIT 2202.04200 · LLaDA 2502.09992 · Levenshtein 1905.11006.

---

## 11. CAPSTONE — la convergenza R-F + R-G + R-H (sintesi Architetto)

Le due intuizioni utente (block-decode + scultore) e i tre filoni convergono su **UN solo execution model**, ora mappato onestamente:

1. **Il soffitto è `c≈3-5` token/weight-stream sulla PROSA** — vero sia via speculative-AR (R-F, esatto) sia via carve (R-H). È un **bound FONDAMENTALE** (indipendenza condizionale = quanta parte del testo è localmente indipendente), non un fallimento di tuning. Non 500×, non 128×.
2. **Ma il bound è CONTENT-DEPENDENT:** su contenuto strutturato/basso-entropia (Cat-A: codice/log/agentico) `c` è molto maggiore. → **il motore è più veloce ESATTAMENTE dove punta il prodotto (LLM agentico = carico ricco di struttura/tool/codice).** Il fit prodotto-architettura è allineato.
3. **Carve vs speculative-AR:** pareggiano sulla prosa (~3×), ma il carve **non ha draft né KV E dissolve il crux state-rollback** → **su SSM il carve è il fit più semplice e pulito.**
4. **Il piano a-priori (R-G):** non migliora la qualità (l'AR pianifica già, provato) — fornisce **larghezza-di-batch** al single-user (sezioni indipendenti → parallele). = infrastruttura di throughput, limitata a contenuto separabile.
5. **Tutto fonde:** chassis block-decode (R-F) + piano-che-fornisce-width (R-G/SoT) + carve-che-riempie-il-blocco-in-place (R-H/SBD) = **un execution model coerente, nicchia aperta (SSM × few-step-set-block), nativamente adatto al nostro backbone** (scan parallelo = il refiner, no-KV, no-rollback) + DNA compressione.
6. **Deliverable onesto:** ~3-5× sulla prosa (pareggia gli alternativi ma più pulito su SSM), di più su Cat-A, SOPRA lo shift bandwidth→compute-bound che sblocca multi-core/cache. **Originalità = la combinazione SSM-CPU-few-step-block, impubblicata.**

**Probe (quando si apre lo scale-up, C puro, no-training-pesante):** (1) microbench roofline matmul ternario-LUT vs N (il ridge); (2) **decidi il main-loop: speculative-AR (checkpoint-stato) VS carve/SBD (ri-scan, no-rollback)** — R-H suggerisce che il carve è più semplice su SSM, da verificare; (3) misura `c` su prosa vs Cat-A (il content-dependence); (4) ri-valida la decode-hygiene per l'unmasking. Niente tocca il sizing recall in corso.
