# 02 — Le cinque idee

*Dallo smontaggio delle 10 stazioni di un LLM sono sopravvissute 5 idee buone. Qui le spiego una a una, in parole semplici. (Il metodo dello smontaggio — sostituire/fondere/invertire ogni pezzo — è nel quaderno tecnico `../exploration.md`, sezione "Fase A".)*

Le chiamo **K1…K5** (K = "keeper", cioè "idea da tenere").

---

## K2 — Un solo blocco, riusato più volte *(la più pronta da provare)*

**Cosa fa un LLM normale.** Impila tanti "strati" diversi (layer), ognuno coi *suoi* pesi. Per ogni token li legge tutti, uno strato dopo l'altro. Se il modello è grande, questi strati non stanno tutti in cache → si rilegge dalla RAM ogni volta → lento.

**L'idea K2.** Usa **un solo blocco** di calcolo e **riusalo più volte** (es. 4 volte) per dare "profondità" al pensiero, invece di avere 4 blocchi diversi. È come **rileggere lo stesso passo di una ricetta** più volte, invece di voltare pagina ogni volta.

**Perché aiuta su CPU.** Il blocco unico è piccolo abbastanza da **stare in cache**. Una volta che è lì, riusarlo 4 volte **non costa nessuna nuova lettura dalla RAM** — i numeri sono già sulla scrivania. In più si può rendere il numero di ripetizioni **adattivo**: parole facili → 2 ripetizioni, parole difficili → 8. "Pensare di più" diventa **gratis in termini di memoria**.

**Slogan:** *"compra profondità senza pagare banda."*

**Quanto è originale.** I mattoni esistono (si chiamano "weight sharing", "Universal Transformer", "DEQ", "MoR"). La parte nostra: farlo su un motore di tipo **SSM** (il nostro tipo di modello, che non ha la memoria che cresce dei modelli ad attenzione), con numeri ternari, su una CPU senza istruzioni speciali, e misurarne il guadagno di *banda*. Nessuno l'ha fatto così.

**Cosa dice la ricerca.** Funziona **a una condizione precisa e quantificabile**: il blocco deve **entrare in cache** (per noi: a numeri ternari un blocco da ~3 milioni di pesi sta in L3). Se ci sta, il numero di riusi diventa un **moltiplicatore diretto** della velocità.

**I rischi.** (1) Il riuso "puro" da solo perde un po' di qualità → il guadagno vero viene **dalla profondità adattiva**, non dal riuso secco. (2) Alcune varianti (DEQ) richiedono troppe ripetizioni per stabilizzarsi → vanno tenute "corte".

**Stato:** **è la leva testabile SUBITO**, anche alla scala piccola di oggi.

---

## K1 — La parte pesante diventa una tabella da consultare *(la più potente sui modelli grandi)*

**Il fatto sorprendente.** La parte più grossa di un LLM (si chiama "FFN", ~2/3 di tutti i pesi) **è già, di fatto, una tabella di consultazione mascherata da moltiplicazione**: moltiplica per un mucchio enorme di "chiavi", ma poi ne usa davvero solo pochissime (la matematica scarta il resto). È un fatto dimostrato in letteratura (Geva et al.).

**L'idea K1.** Smettila di fingere: **rendila una tabella esplicita**. Invece di moltiplicare per tutto e buttare via il 95%, **vai a prendere direttamente solo le poche voci che servono** (tecnica: "product-key memory"). E fai lo stesso per l'**uscita finale** del modello (la parte che sceglie la prossima parola): invece di dare un punteggio a tutte le 30.000 parole possibili, **recupera solo le poche probabili** con lo stesso meccanismo a indice.

**Perché aiuta su CPU.** Tocchi **pochissimi pesi per token** → leggi pochissimi byte. Una memoria enorme (che vive nella RAM, anche 1 GB) ma di cui per ogni parola prendi solo ~8 KB.

**Quanto è originale.** I mattoni esistono (PKM, UltraMem, retrieval-head). **La parte davvero nostra: fondere la "parte pesante" (FFN) e l'"uscita" del modello in UN UNICO indice condiviso.** Nessuno l'ha mai fatto — eppure è naturale, perché (di nuovo Geva) le voci della FFN "vivono già nello spazio delle parole". Più: il nostro taglio è pensato per la CPU a una parola alla volta, che la letteratura ignora (guarda solo le GPU).

**Cosa dice la ricerca — e la mia correzione importante.** Funziona **solo sui modelli grandi**. Il motivo: K1 sostituisce una lettura *in ordine* (sequenziale, veloce) con un recupero *a salti* (casuale, lento). Conviene **solo quando la parte pesante è così grande da non stare in cache** (allora la lettura ordinata è comunque lentissima dalla RAM, e il piccolo recupero a salti la batte). Su un modello **piccolo** che sta già tutto in cache, K1 **non serve o peggiora**. → **K1 è una leva per il modello grande futuro, non per il giocattolo di oggi.**

**Il ponte con K3.** Il punto debole di K1 è proprio quel "recupero a salti" lento. Ma si può **andare a prendere in anticipo** le voci giuste (prefetch), *se* sai quali serviranno — ed è esattamente ciò che fa K3.

**I rischi.** Voci "morte" mai usate (si cura con accorgimenti di addestramento), e le parole rare che soffrono se fondi anche l'uscita.

---

## K3 — Allena il modello a essere prevedibile *(la più originale — è "terra vergine")*

**L'analogia.** Le CPU moderne hanno un "predittore di salti": indovinano in anticipo che strada prenderà il programma e preparano i dati prima ancora di saperlo per certo. Vincono perché **i programmi sono prevedibili**.

**L'idea K3.** Allena **il modello stesso** a essere prevedibile su *quali pesi gli serviranno per la prossima parola*, partendo dallo stato attuale. Se la prossima "manciata di voci" è prevedibile, possiamo **andarla a prendere in anticipo** mentre il processore è ancora occupato → la lentezza del recupero a salti (il tallone di K1) **sparisce**.

La mossa fine: non costruire solo un "indovino" migliore, ma **rendere il modello stesso più facile da indovinare** — un termine in più nell'addestramento che premia la prevedibilità. (Come la sua stessa analogia: i predittori di salto vincono perché i salti *sono* prevedibili.)

**Quanto è originale.** Questa è la **"terra vergine" confermata dalla ricerca**. Esistono pezzi vicini: chi *misura* la prevedibilità su un modello già fatto (Deja Vu), chi rende "vicini" gli esperti nei modelli MoE per le GPU (ReMoE). Ma **nessuno allena il modello a pagare lui stesso il prezzo di rendersi prevedibile**, su un modello del nostro tipo, da zero, per la cache di una CPU. La sintesi — *"pesi facili-da-prevedere come proprietà allenata"* — **non è pubblicata.** È il pezzo più nostro.

**Cosa dice la ricerca — e i due avvertimenti onesti.** (1) **Forse il modello è già abbastanza prevedibile da solo** (Deja Vu misura 93-99% di accuratezza senza nessun allenamento speciale). Se è così, allenarci sopra non aggiunge nulla → **prima si MISURA quanto è già prevedibile**, poi si decide. (2) Come K1, **conta solo sui modelli grandi** (su un modello piccolo che sta in cache, il prefetch è inutile).

**L'ordine obbligato.** Per "prevedere quali pesi servono" servono prima *pochi* pesi attivi e distinti → bisogna prima **rendere il modello sparso** (tecnica nota: Q-Sparse), poi renderlo coerente, poi prevedibile. Sparsifica → rendi coerente → rendi prevedibile.

**Un vantaggio nostro.** Su GPU questa "coerenza" costa, perché sbilancia il carico tra le unità. Da noi, **una parola alla volta su un solo core, quel costo non esiste**: la coerenza ci costa meno che a tutti gli altri.

---

## K4 — Entrare e uscire a "unità di compressione" *(l'anima storica del progetto)*

**L'idea.** Invece di spezzare il testo in pezzetti fissi decisi in anticipo, **spezzalo dove il modello è "sorpreso"** (alta imprevedibilità): tratti facili e prevedibili → pezzi grandi, pochi passi; tratti difficili → pezzi piccoli. E in uscita, produci direttamente **codici di compressione**. In pratica: il modello respira al ritmo dell'informazione.

**Perché c'entra con noi.** È letteralmente il DNA del progetto (**SEE = Silicon Entropy Engine**, la cui metrica è la compressione, i "bit per byte"). E dà un bonus di velocità: meno passi sui tratti prevedibili, e una "testa di uscita" piccola.

**Stato:** leva media, da innestare più avanti. I mattoni esistono (si chiama "Byte Latent Transformer"); la fusione con l'anima SEE è la parte nostra.

---

## K5 — Più parole per ogni lettura dei pesi

**L'idea.** Visto che il costo è "leggere tutti i pesi per ogni parola", **leggi i pesi una volta e produci più parole insieme** (tecnica: "multi-token prediction" / "self-drafting"). Ammortizzi la lettura.

**Stato:** moltiplicatore noto e a basso rischio (~1.5-2.5× reale su CPU). Si aggiunge sopra a tutto il resto, alla fine.

---

## Riepilogo

| Idea | In una riga | Leva | Originalità | Quando |
|------|-------------|------|-------------|--------|
| **K2** | Un blocco riusato, in cache | Banda (stream-once) | I pezzi esistono, il taglio SSM-CPU no | **Subito** (sandbox) |
| **K1** | Parte pesante = tabella da consultare | Pochi byte/token | **La fusione FFN+uscita è nuova** | Modello grande |
| **K3** | Allena il modello a essere prevedibile | Nasconde la latenza | **Terra vergine** | Modello grande |
| **K4** | I/O a unità di compressione | Meno passi + uscita piccola | Fusione con SEE | Più avanti |
| **K5** | Più parole per lettura | Ammortizza | Noto, basso rischio | Alla fine |

➡️ Queste 5 idee non vanno prese alla spicciolata: **si incastrano in una sola architettura**. La spiega il documento [03 — L'architettura unificata](03_Architettura_Unificata.md).
