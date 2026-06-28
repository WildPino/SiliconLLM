# 04 — Le prove e i prossimi passi

*Le idee non sono campate per aria: per le tre più importanti abbiamo fatto altrettante ricerche di approfondimento sulla letteratura. Qui spiego cosa abbiamo chiesto, cosa è venuto fuori (incluso ciò che abbiamo **scartato**) e quali esperimenti economici servono per verificare.*

---

## Come abbiamo verificato

Per ogni idea ad alto potenziale ho mandato un "ricercatore" (un sub-agente con accesso al web) con un compito preciso: **dimmi cosa esiste già, cosa è davvero nuovo, e qual è il verdetto onesto sulla CPU** (inclusi i motivi per cui potrebbe NON funzionare). Tre ricerche, tre verdetti. Le ho lette **di testa mia**, senza fidarmi delle conclusioni automatiche.

---

## Ricerca 1 — la "memoria da consultare" (K1)

**Cosa abbiamo chiesto:** la parte pesante del modello e l'uscita possono diventare una grande tabella da consultare? Esiste? Conviene su CPU?

**Cosa è emerso:**
- **La base è solida e matura.** Il metodo si chiama *product-key memory* (2019), ed è stato ripreso e validato di recente: **UltraMem** (2025) eguaglia la qualità di un modello denso 4× più grande, andando 2-6× più veloce. La teoria che la FFN "è già una tabella" è dimostrata (Geva).
- **Due cose sono davvero nostre:** (a) **fondere la parte pesante e l'uscita in un unico indice condiviso** — nessuno l'ha fatto; (b) il **taglio CPU a una parola alla volta** — tutti guardano le GPU.

**La correzione importante (la mia lettura, non l'automatismo):** K1 **conviene solo sui modelli grandi.** Sostituisce una lettura ordinata (veloce) con un recupero a salti (lento): vince **solo** quando la parte pesante è troppo grande per la cache. Su un modello piccolo che ci sta già, **non serve o peggiora.** → K1 è una leva per il futuro, non per oggi.

**Il regalo della ricerca:** il punto debole di K1 (il recupero a salti) si cura **andando a prendere le voci in anticipo** — e questa è la cosa più importante e *non pubblicata*. È esattamente ciò che abilita K3.

---

## Ricerca 2 — il "blocco riusato" (K2) e la diffusione (scartata)

**Cosa abbiamo chiesto:** riusare un solo blocco fa davvero risparmiare memoria su CPU? E la generazione "a diffusione" (tutte le parole insieme, raffinate a step) può battere quella una-parola-alla-volta?

**Cosa è emerso su K2 (promosso):**
- È **la scommessa CPU più forte**, e **più naturale sul nostro tipo di modello** (SSM) che sugli altri.
- Il risultato chiaro: il riuso fa risparmiare memoria **solo se il blocco entra in cache**. Se ci entra (a numeri ternari ci entra), **il numero di riusi diventa un moltiplicatore diretto della velocità**, e "pensare di più" non costa nessuna nuova lettura. Lo scheletro pronto da copiare si chiama **MoR**.

**Cosa è emerso sulla diffusione — SCARTATA, verificato:**
- Sulla nostra CPU, una parola alla volta, **perde**. Il motivo: per migliorare il testo fa *molti* passaggi completi (ognuno rilegge tutti i pesi) e perde i vantaggi del nostro tipo di modello. La promessa "leggi i pesi una volta sola" si rivela un miraggio nel nostro scenario. **Chiuso, con prova alla mano.**

> Perché conta scartare cose: ogni vicolo cieco eliminato con una prova è tempo e GPU risparmiati. La diffusione era seducente; l'abbiamo uccisa pulita.

---

## Ricerca 3 — "allenare alla prevedibilità" (K3)

**Cosa abbiamo chiesto:** esiste già l'idea di allenare il modello a essere prevedibile su quali pesi userà, così da prenderli in anticipo?

**Cosa è emerso:**
- **È terra vergine, confermato.** Esistono pezzi vicini (chi *misura* la prevedibilità su un modello già fatto; chi rende "vicini" gli esperti nei modelli MoE per le GPU), ma **nessuno fa pagare al modello stesso il prezzo di rendersi prevedibile**, da zero, per la cache di una CPU. È il pezzo più originale di tutti.
- **I mattoni per costruirlo esistono tutti** (per la sparsità: *Q-Sparse*; per la coerenza: *ReMoE*; per l'indovino: *Deja Vu*) → andrebbero assemblati + aggiunto il pezzo nuovo.

**I due avvertimenti duri (la parte onesta):**
1. **La prevedibilità potrebbe già essere alta da sola** (le misure dicono 93-99% senza nessun trucco). Se è così, allenarci sopra **non aggiunge niente** → **prima si MISURA il valore di partenza**, poi si decide. Questo è il "make-or-break".
2. **Conta solo sui modelli grandi** (come K1).

**L'ordine obbligato:** prima rendi il modello sparso, poi coerente, poi prevedibile. E un vantaggio nostro: la "coerenza" che su GPU costa, da noi (una parola alla volta, un core) **è gratis**.

---

## La cosa scartata che vale ricordare

Oltre alla **diffusione**, abbiamo fatto anche un giro su strumenti presi da **altri campi** (trasformate di Fourier, automi cellulari, sketch probabilistici, reti di ordinamento, ecc.) per vedere se ne nascesse un'idea nuova vincente.

**Verdetto onesto: no.** Lo sweep **non ha prodotto un nuovo candidato vincente** — ha solo **rafforzato** le idee che già avevamo (le matrici strutturate rinforzano K2; i predittori di salto rinforzano K1) e svelato una sola nicchia minore. **Non abbiamo fabbricato novità dove non c'era.** È un risultato informativo: vuol dire che le 5 idee *sono* il baricentro giusto, non un caso fortunato.

---

## I prossimi passi: esperimenti economici, in ordine

La regola d'oro del progetto: **prima il costo, una variabile alla volta, niente esperimenti costosi finché uno economico non ha dato il via libera.** E soprattutto: **niente di tutto questo tocca il lavoro in corso** (la rampa recall run-6) — sono prove per il modello grande futuro.

**Probe-1 — verifica K2 *(l'unica testabile SUBITO)*.** Allena un blocco riusato N volte e confrontalo con un modello "normale" a strati distinti, a parità di pesi. Domanda secca: **il riuso + la profondità adattiva tengono la qualità?** Economico, fattibile alla scala di oggi.

**Probe-2 — verifica K1 (microbenchmark, senza addestramento).** Sulla nostra CPU, misura quanto costa "andare a prendere a salti" poche voci da una tabella più grande della cache, **con e senza il prendere-in-anticipo**, e confrontalo con la lettura ordinata. Domanda secca: **il prendere-in-anticipo nasconde la lentezza e batte il metodo denso?** Ore di lavoro, niente GPU.

**Probe-3 — verifica K3 (dopo Probe-1).** Su un modello reso sparso, **misura prima quanto è già prevedibile** (è il make-or-break: se è già 93-99%, K3 non serve). Solo se c'è margine, aggiungi l'allenamento alla prevedibilità e vedi se sale **senza** peggiorare la qualità.

**Ordine pratico:** Probe-1 e Probe-2 sono economici e indipendenti → si possono fare in parallelo. Probe-3 viene dopo (ha bisogno del modello prodotto da Probe-1).

---

## In una frase

> Le tre idee centrali reggono alla prova della letteratura: **K2 è forte e pronta**, **K1 e K3 sono potenti ma solo a grande scala** (e K3 è la più originale, "terra vergine"). Abbiamo scartato con prova ciò che non regge (la diffusione) e siamo stati onesti su dove non c'era novità. Il prossimo colpo concreto, quando il lavoro in corso libera le risorse, è **Probe-1 su K2** — l'unico già verificabile alla scala di oggi.

---

*Documento tecnico di riferimento (denso, per chi vuole i numeri e le citazioni): `../exploration.md`. Memoria di progetto collegata: `project_llm_reimagined_exploration`, `project_cpu_bandwidth_research`, `project_phase55_plan`.*
