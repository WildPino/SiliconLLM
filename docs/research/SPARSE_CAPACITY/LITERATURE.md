# Sparse Capacity: la letteratura letta prima di misurare

2026-11-17. Fase 0 del [piano](PLAN.md), domande D-M e D-R.

**Domande.**
- **D-M:** esiste gia un metodo che tiene la banca di esperti nella RAM dell'host, la calcola e la aggiorna sulla
  CPU, e allena il nucleo sulla GPU? Dove resta spazio libero?
- **D-R:** una banca grande compensa un nucleo piccolo? Su quali compiti?

**Come leggere le fonti.** Ogni numero viene dal testo del paper. Accanto a ogni fonte c'e come e stata letta:
- **[letto]:** testo del paper;
- **[abstract]:** solo l'abstract;
- **[sintesi]:** riassunto automatico di parti non lette alla lettera, da ricontrollare prima di citarlo fuori;
- **[doc]:** documentazione del progetto, non un paper.

## 1. In breve

- **Lo spazio "nucleo su GPU, esperti su CPU" e gia occupato** per il fine-tuning di MoE grandi, su macchine da server:
  KTransformers-FineTune (2026), OOM (novembre 2026), ES-MoE (2024), ZeRO-Offload (2021).
- **Il modo di fallire e gia misurato:** calcolare gli esperti sulla CPU perde contro il portare i loro pesi sulla GPU.
  Lo dice OOM, ed e confermato da un conto a tavolino (§3), che smentisce la premessa "si spostano le attivazioni, non i
  pesi".
- **Resta libero un angolo stretto ma reale:**
  - **dove:** pre-addestramento di banche a grana estrema, su una sola GPU consumer e una CPU AVX2 senza AMX;
  - **come:** la scelta CPU o GPU si fa per esperto, in base a quanti token ciascuno riceve nel passo;
  - **inoltre:** aggiornamenti sparsi come nei recommender.

  E un contributo di sistema, non un paradigma nuovo.
- **D-R:** la letteratura dice che una banca grande da soprattutto **conoscenza**, mentre il **ragionamento** segue i
  parametri attivi.
  - Aiuta anche il ragionamento, ma in modo indiretto: Engram libera i primi strati e fa BBH +5,0.
  - Pero non sostituisce il nucleo: oltre una certa quota il ragionamento satura.
  - Per le banche di micro-esperti la domanda sul ragionamento non e misurata da nessuno (§5).

## 2. Chi occupa lo spazio

| Lavoro | Cosa fa | Cosa lascia libero |
|---|---|---|
| **KTransformers-FineTune** (Li et al. 2026, preprint degli autori) [doc + citato da OOM] | Attenzione sulla GPU, esperti instradati calcolati sulla CPU (forward e backward) con kernel AMX; LoRA e parametri pieni; DeepSeek-V3 671B in LoRA a 40,35 tok/s con ~1,2-1,3 TB di RAM | Richiede AMX (Xeon); solo fine-tuning di modelli esistenti; esperti grossi |
| **OOM** (arXiv 2611.11364, nov. 2026) [letto in parte + sintesi] | Tiene master FP32 e momenti di Adam sull'host, porta i pesi in BF16 sulla GPU, AdamW sulla CPU per le parti in streaming; 4090 e H100; Qwen3.6-35B-A3B (256 esperti, top-8) | Niente pre-addestramento da zero, niente esperti piccoli, niente bit bassi ("no quantization"), aggiornamenti sincroni |
| **ES-MoE** (ICML 2024) [letto] | Parametri degli esperti e stato dell'ottimizzatore sull'host, esperti portati sulla GPU in pipeline, ottimizzatore sulla CPU per esperto; nota che Adam su CPU e "31x slower" | Calcolo sempre su GPU; esperti grossi |
| **ZeRO-Offload** (2021) [letto] | Gradienti, stati e passo dell'ottimizzatore sulla CPU, forward e backward sulla GPU; ">13 billion parameters on a single GPU"; aggiornamento ritardato di un passo senza perdita finale | Modelli densi, nessuna sparsita |
| **Recommender** (HugeCTR ETC, Persia, ScratchPipe) [sintesi della ricerca] | Tabelle di embedding enormi nella RAM dell'host, aggiornate in modo sparso mentre la GPU allena il resto | Embedding, non esperti con non-linearita, e niente LM |
| **Memory Layers at Scale** (Berges 2024) [letto] | Memorie a chiavi prodotto fino a 128B parametri, valori "sharded across the embedding dimension" sulle GPU | Tutto in VRAM su piu GPU; nessun host |
| **Engram** (DeepSeek, arXiv 2601.07372) [letto] | Tabella di n-gram da consultare: in training "sharded across available GPUs"; in inferenza 100B parametri sull'host con perdita del 2,8% (Tab. 4) | L'addestramento con la tabella sull'host |
| **PEER** (He 2024) [letto] | 1024² esperti da un neurone, chiavi prodotto, 8 teste x top-16; a parita di FLOP batte denso, MoE e PKM (Tab. 1: 20,63 contro 23,84 del denso a 6e18) | Tutto su GPU |
| **SLIDE** (MLSys 2020) [letto] | Addestramento sparso su CPU (LSH sui neuroni, aggiornamenti in stile Hogwild); 44 core "3.5 times" piu veloce di TF su V100, su classificazione estrema | Non e un LM; confronto con TF del 2019, non con una GPU ottimizzata |
| **Hogwild!** (2011) [letto] | Aggiornamenti asincroni senza lock, con convergenza quasi ottima se gli aggiornamenti sono sparsi | Teoria, utile per la sparsita della banca |
| **Ternario con kernel LUT nel backward** | Nessuna fonte trovata: le LUT stanno solo nell'inferenza (bitnet.cpp, T-MAC) | Non trovato non vuol dire che non esista |

## 3. Il lavoro piu vicino e il modo di fallire

**I lavori piu vicini sono KT-FT e OOM.** OOM li confronta sulla stessa macchina (§7.2, [letto]):
- KT-FT fa 659-803 tok/s, OOM e "26-60x" piu veloce;
- "even on a 48-core AMX Xeon, KT-FT's CPU expert path takes 70% of its exposed step time at batch 32, against 4% for
  the FP32 optimizer update".

**Il motivo, in un conto a tavolino** (ipotesi non misurate su questa macchina):
- **Costo sulla CPU:** forward e backward di un esperto costano circa 6 FLOP per parametro per ogni token che riceve.
  A 200 GFLOP/s (una stima, non una misura) fanno 0,03 ns per parametro per token.
- **Costo dello streaming:** portare il peso in BF16 sulla GPU e riportarne il gradiente costa 4 byte per parametro.
  Con PCIe a 25 GB/s fanno 0,16 ns per parametro, una volta sola per passo.
- **Il pareggio:** si raggiunge quando ogni esperto riceve circa **5 token per passo** (circa 11 con PCIe a 12 GB/s).
  Sopra, conviene portare i pesi sulla GPU; sotto, conviene la CPU.
- **Le configurazioni candidate sono tutte sopra.** Con 128k token per passo:
  - h128 con E = 12.700 da circa 83 token per esperto;
  - h8 con E = 8.192 circa 2.000;
  - PEER con 1M esperti circa 16.
- **Un esempio:** una banca stile PEER da 4,3 G parametri costa circa 0,7 s per passo in streaming, contro circa 2 s di
  calcolo su CPU.
- **Quando vince la CPU:** solo con passi piccoli (PEER a 16k token per passo da circa 2 token per esperto) o con banche
  molto piu grandi.

**Quindi la premessa "si spostano le attivazioni" cade nel regime di addestramento.** Spostare decine di GB per passo costa 1-2 s su
PCIe, ed e la strada che OOM ha misurato come la migliore.

## 4. Lo spazio libero

**Nessuna fonte trovata fa insieme queste quattro cose.** Ogni pezzo singolo ha un precedente.
1. **Pre-addestramento da zero** di un LM con una banca a grana fine o estrema piu grande della VRAM. Tutti gli altri
   fanno fine-tuning, oppure tengono la banca su piu GPU.
2. **Una sola GPU consumer e una CPU senza AMX ne AVX-512.** KT-FT chiede AMX, OOM e ES-MoE girano su Xeon.
3. **Il posto di ogni esperto scelto in base ai token che riceve nel passo.** Gli esperti molto usati vanno in streaming
   sulla GPU, quelli poco usati si calcolano sulla CPU, con l'aggiornamento sparso dei recommender.
   - OOM separa gli esperti residenti da quelli in streaming, ma calcola tutto sulla GPU.
   - KT-FT calcola tutto sulla CPU.
   - Per l'inferenza esistono sistemi che scelgono CPU o GPU per esperto, non letti in questa ricerca.
4. **Pesi ternari con i kernel LUT nel percorso CPU,** esportabili nel motore senza riaddestrare.

**Il merito che ne verrebbe:** un metodo misurato e rifacibile per allenare su un PC banche che non entrano nella VRAM.
Ha valore, ma e ingegneria su idee note.

## 5. D-R: una banca grande compensa un nucleo piccolo?

**Sulla conoscenza si, sul ragionamento no.**
- **Mixture of Parrots** (arXiv 2410.19034) [letto in parte + sintesi]: "the memorization performance consistently
  increases while the reasoning capabilities saturate". Il sapere segue i parametri totali, il ragionamento quelli
  attivi (§4.2-4.3). C'e anche un limite teorico: oltre (log N)² esperti il modello sparso non risolve un compito su
  grafi che il denso risolve (§3.2).
- **Engram** [letto]: la curva e a U. Conviene dare alla memoria circa il 20-25% dei parametri sparsi; oltre, la memoria
  "cannot replace computation". Con 3,8B attivi da guadagni anche nel ragionamento (BBH 50,9 -> 55,9, Tab. 1), quindi
  una parte di memoria aiuta anche li.
- **Memory Layers** [letto]: a 1,3B di base NQ sale da 7,76 a 13,68, TQA da 32,64 a 42,89 (Tab. 1). A 8B il codice e
  misto: MBPP scende da 44,20 a 42,20 (Tab. 2). Per le banche grandi segnala instabilita, "especially for small base
  models".
- **Clark 2022** [sintesi]: il guadagno degli esperti cala al crescere del modello base, e il numero di esperti
  satura (E_max stimato fra circa 300 e 480).
- **Krajewski 2024** [letto in parte]: la grana fine aiuta, ma i ritorni "converging to a positive constant"; a G = 16
  "the routing cost dominates gains". Riguarda gli esperti h8.

**Cosa resta aperto.** Le prove sul ragionamento vengono da MoE classici (Parrots: top-2, al massimo 64 esperti) e da
memorie consultate per indice (Engram, Memory Layers). Nessuna fonte trovata misura il ragionamento con banche di
micro-esperti a parita di nucleo, ne con un nucleo piccolo che pensa piu a lungo spendendo il margine di velocita
(`docs/PHASE64_BUDGET.md` §4). E l'unico punto in cui il programma potrebbe dare piu di un metodo; la stima e che la
probabilita sia bassa.

**Cosa vuol dire per il programma.** La prova di D-0 giudicata solo sul BPB passerebbe quasi certamente, perche il BPB
premia anche la conoscenza. Ma non risponderebbe a "il modello diventa intelligente" (regola 3). Deve riportare anche
due misure separate, una di conoscenza e una di ragionamento.

## 6. Cosa ne segue per il piano

- **La premessa cambia:** "si spostano le attivazioni, non i pesi" non regge in addestramento. La misura 1b del piano
  trova la soglia r* sul 3600X con la 3060.
- **Le sonde:** la fase 2 misura conoscenza e ragionamento separati, oltre al BPB.
- **Cosa smentirebbe l'angolo libero di §4:**
  - la CPU perde anche con circa 1 token per esperto per passo;
  - oppure un lavoro pubblicato fa gia i punti 1-3 di §4.

## Fonti

- KTransformers-FineTune: note tecniche, https://kvcache-ai.github.io/ktransformers/en/SFT/KTransformers-Fine-Tuning_Developer-Technical-Notes.html ;
  preprint `paper-ktft-v1.1` nel repo kvcache-ai/ktransformers (non letto).
- OOM, arXiv 2611.11364. ES-MoE, ICML 2024 (PMLR 235:24342). ZeRO-Offload, arXiv 2101.06840. SlideFormer, arXiv
  2603.16428 (solo la sintesi della ricerca).
- SLIDE, MLSys 2020. Hogwild!, arXiv 1106.5730. HugeCTR ETC (documentazione NVIDIA Merlin), ScratchPipe (ISCA 2022) e
  Persia (citato in arXiv 2208.05321) solo dalla sintesi della ricerca.
- PEER, arXiv 2407.04153. Memory Layers at Scale, arXiv 2412.09764. Engram, arXiv 2601.07372. UltraMem, arXiv 2411.12364
  (solo l'abstract). Product Key Memory, arXiv 1907.05242 (solo l'abstract).
- Mixture of Parrots, arXiv 2410.19034. Krajewski, arXiv 2402.07871. Clark, arXiv 2202.01169.

**Non letto, da leggere prima di un protocollo:**
- il preprint di KT-FT;
- MoP (Nguyen 2026), LazyTrain e MegaTrain, citati da OOM;
- i sistemi d'inferenza che scelgono CPU o GPU per esperto;
- il corpo di UltraMem e la parte finale di Clark.

## Aggiunte del 2026-11-27

**Per la sonda R nuova (tracciamento di variabili).** Testo letto sull'HTML di arXiv. Le parti segnate [sintesi]
vanno ricontrollate prima di citarle fuori.
- **Wu, Geiger, Milliere,** "How Do Transformers Learn Variable Binding in Symbolic Programs?", arXiv 2505.20896 (ICML
  2025). Programmi di 17 righe `var = const | var = var`, catene fino a 4 salti con catene di distrazione.
  - In addestramento passa per un'euristica di posizione: il letterale della prima riga, "over 90%" dei casi dove e
    giusto.
  - L'euristica funziona perche il generatore mette le radici delle catene lunghe in alto.
  - Accuratezza: dal caso (~12%) a 56%, poi 99%.
  - Il formato della sonda viene da qui, e le catene sono tutte della stessa profondita proprio contro quell'euristica.
- **RULER** (Hsieh et al.), arXiv 2404.06654: compito di variable tracking (catene `VAR X2 = X1`). Modi di fallire:
  copia dal contesto, variabili dell'altra catena [sintesi].
- **Kim e Schuster,** "Entity Tracking in Language Models", arXiv 2305.02363: i casi banali (stato iniziale invariato,
  62,7% in un dataset precedente) gonfiano il punteggio. Kim, Schuster, Toshniwal, arXiv 2405.21068: l'addestramento
  sul codice migliora il tracciamento, con modelli da 7B in su.
- **Zaremba e Sutskever,** "Learning to Execute", arXiv 1410.4615. Due modi di fallire: la memorizzazione (sez. 8) e
  il teacher forcing, che gonfia l'accuratezza (sez. 6).
- **CRUXEval,** arXiv 2401.03065: predizione dell'output di funzioni di 3-13 righe. Phi-1 (1,3B) fa circa il 23%
  [sintesi, tabella 2].

**Per la fase 4 (pesi quantizzati nel trasferimento):**
- **ZeRO++,** arXiv 2306.10209, qwZ: pesi da FP16 a INT8 a blocchi nell'all-gather, con perdita di accuratezza se la
  quantizzazione non e a blocchi (sez. 3.1).
- **QSDP,** arXiv 2302.02390: pesi e gradienti a 8 bit, "biased estimators of the gradients, taken over quantized
  weights" (sez. 1).
- **OOM,** arXiv 2611.11364: niente quantizzazione nel trasferimento, i pesi viaggiano in BF16 [sintesi, sez. 7.6].
- In tutti e tre la quantizzazione nel trasferimento **cambia** la matematica. Qui no: la GPU usa gia solo il
  ternario e la scala per riga (`w_q`, gradiente straight-through g/s con s staccata). Mandare quei due e la stessa
  matematica.
- **Il precedente piu vicino:** float8 all-gather in FSDP2 (PyTorch dev-discuss, 7 agosto 2024, weifengpy e altri;
  torchao `float8/fsdp_utils.py`).
  - Stesso principio: il forward usa gia il peso castato, quindi castarlo prima della comunicazione e "almost a 'free
    lunch'".
  - La scala si calcola dopo il passo dell'ottimizzatore (`precompute_float8_dynamic_scale_for_fsdp`). Risparmia "50%
    bandwidth (vs bfloat16)".
  - Numerica "on-par": sono curve, non uguaglianza bit a bit.
  - La scala per riga e "highly exploratory", per il backward trasposto.
  - Qui la differenza e stretta: ternario a 2 bit con una scala per riga, dall'host alla GPU (offload, non
    all-gather), e la quantizzazione fusa nell'AdamW dell'host, uguale bit a bit a quello di torch (`sc4_adamw.py`).
  - Il backward trasposto qui non e un problema: il gradiente straight-through e g/s, con s per riga, e la GPU ha gia
    q e s.
- **Ternary Backpropagation** (preprint 2026 su alphaxiv, solo la pagina della ricerca, non letto): toglie i master
  fp32. E un'altra strada, che cambia la matematica.
- Ricerca chiusa per la fase 4: nessun lavoro manda pesi ternari impacchettati dall'host alla GPU tenendo master e
  AdamW sull'host.
