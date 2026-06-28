# 01 — Fondamenti: cos'è un LLM e perché su CPU il collo è la memoria

*Questo documento costruisce da zero l'intuizione che serve per capire tutto il resto. Nessun prerequisito.*

---

## 1. Un LLM è una grossa funzione matematica

Togliamo la magia. Un modello linguistico (LLM) è una **funzione**: gli dai in pasto "il testo scritto finora" e ti restituisce **una probabilità per ogni possibile parola successiva**.

```
   testo finora  ──►  [ LLM ]  ──►  "next = 'gatto' 12%, 'cane' 9%, 'sole' 3%, ..."
```

Per scrivere un testo, si ripete questo in cerchio:
1. dai il testo al modello,
2. il modello dice le probabilità della prossima parola,
3. ne scegli una,
4. la attacchi al testo,
5. torni al punto 1.

Questa ripetizione in cerchio si chiama **generazione autoregressiva** ("autoregressiva" = ogni nuova parola si basa su quelle già prodotte). Il punto chiave: **una parola alla volta**, in sequenza.

> **Parola tecnica — "token".** In pratica il modello non lavora su parole intere ma su pezzetti (sillabe, frammenti, a volte byte). Si chiamano *token*. Per noi "token" ≈ "una parola/pezzetto prodotto per giro". La velocità di un LLM si misura in **token al secondo (tok/s)**.

---

## 2. Di cosa è fatta questa funzione: numeri (i "pesi")

Dentro, la funzione è fatta di **tantissimi numeri** fissi, decisi durante l'addestramento. Si chiamano **pesi** (o parametri). Un modello piccolo ne ha milioni; uno grande, miliardi.

Generare un token significa, in sostanza: **prendere quei numeri, moltiplicarli per i numeri che rappresentano il testo, e sommare**. Tante moltiplicazioni-e-somme, organizzate in "strati" (layer) impilati uno sull'altro.

Due cose costano, in questo processo:
- **il calcolo** (fare le moltiplicazioni),
- **lo spostamento dei dati** (portare i pesi dalla memoria fino al processore, dove si fa il calcolo).

Quale dei due è il vero collo di bottiglia? Dipende dalla macchina e da come la usi. **E qui sta tutto il nostro progetto.**

---

## 3. La scoperta centrale: su CPU, a una parola alla volta, il collo è la MEMORIA

Quando generi **un token alla volta** (in gergo: "batch = 1", cioè una sola sequenza in lavorazione), succede una cosa cruciale:

> Per produrre **ogni singolo token** devi **leggere quasi tutti i pesi del modello dalla RAM**, ma con ognuno fai **poco calcolo**.

È uno spreco di capacità di calcolo: il processore passa il tempo ad **aspettare** che i numeri arrivino dalla memoria. Si dice che il sistema è **"memory-bandwidth-bound"** = *limitato dalla larghezza di banda della memoria* (quanti byte al secondo riesci a far arrivare dalla RAM).

L'analogia giusta: immagina un cuoco velocissimo (il processore) che però deve, per ogni piatto, andare a piedi fino a un magazzino lontano (la RAM) a prendere gli ingredienti, uno scaffale alla volta. Il cuoco non è il collo: **le camminate al magazzino lo sono.**

Questo ci dà l'equazione più importante del progetto:

```
                  larghezza di banda della memoria
   token/sec  ≈  ──────────────────────────────────
                     byte letti per ogni token
```

Per andare più veloci, c'è **una sola leva vera**: **leggere meno byte dalla RAM per ogni token.** E ci sono due modi di farlo:

- **(a) meno byte per peso** → comprimere i numeri (es. usare numeri "ternari" da ~0.2 byte invece di numeri normali da 2-4 byte). Questo è l'asse "quantizzazione/compressione".
- **(b) toccare meno pesi per token** → far sì che, per ogni parola, serva solo una *piccola parte* dei pesi (sparsità, e tabelle da consultare invece di moltiplicare). Questo è l'asse "sparsità".

**Tutte le idee dello sprint attaccano una o entrambe queste due leve.**

---

## 4. La gerarchia della memoria: la "cache" cambia tutto

La RAM non è l'unica memoria. Il processore ha vicino a sé delle **memorie piccole ma velocissime**, chiamate **cache** (livelli L1, L2, L3). Più sono vicine al processore, più sono veloci e più sono piccole.

| Memoria | Velocità (circa) | Dimensione (sul nostro PC) |
|---------|------------------|----------------------------|
| RAM (DDR4) | ~45 GB/s | tanti GB |
| Cache L3 | ~200+ GB/s (4-5× la RAM) | 16 MB |
| Cache L2 | ancora più veloce | 512 KB per core |
| Cache L1 | velocissima | pochi KB |

L'analogia: la **RAM è il magazzino lontano**, la **cache è la tua scrivania**. Se i pezzi che ti servono **stanno già sulla scrivania**, lavori molto più in fretta — niente camminate.

**Conseguenza enorme:** se i pesi (o almeno *quelli che servono ora*) **stanno dentro la cache**, la banda effettiva sale da 45 a 200+ GB/s → **gradino di 4-5× di velocità, gratis.** Questo principio si chiama **cache-residency** ("i dati risiedono in cache") ed è uno dei nostri grimaldelli principali.

---

## 5. Sequenziale contro casuale: non basta leggere "meno", conta "come"

C'è un'ultima sottigliezza, ma decisiva. Leggere la RAM in **ordine** (sequenziale) è veloce: arriva un "pallet" intero di dati in un colpo. Leggere la RAM a **salti** (casuale, random) è lento: ogni accesso fa aspettare ~100 nanosecondi, come un viaggio separato al magazzino per un singolo bullone.

> **Trappola contro-intuitiva:** un'idea che fa leggere *meno* byte ma in modo *casuale* può essere **più lenta** di una che ne legge di più ma in ordine. Lo abbiamo già visto morire più volte nel progetto.

Per questo ogni idea va giudicata su **tre** cose insieme, non solo sui byte:
1. quanti byte legge per token,
2. se li legge in ordine (bene) o a salti (male),
3. se la roba che serve sta in cache.

(C'è poi un quarto vincolo ovvio: **la qualità del testo non deve crollare.** Misuriamo la qualità con una metrica chiamata **BPB** — "bit per byte" — che è l'eredità storica del progetto, l'anima "compressione" del Silicon Entropy Engine.)

---

## 6. La nostra macchina e il nostro vantaggio sleale

**La CPU bersaglio** è un **AMD Ryzen 5 3600X** (architettura "Zen2"): una CPU da PC normale, volutamente non recentissima ("prodotto per tutti"). Caratteristiche che contano:
- niente istruzioni speciali per l'IA (niente "AVX-512", niente "VNNI") → certe scorciatoie usate dagli altri **noi non le abbiamo**;
- RAM DDR4 a ~45 GB/s; cache L2 512 KB/core, L3 16 MB.

**Il vantaggio sleale:** noi **addestriamo il modello da zero.** Tutti gli altri prendono un modello già fatto (nato per GPU) e cercano di rattopparlo per andare su CPU — e sbattono contro muri (qualità che cala, formati che non si comprimono oltre un certo punto). Noi invece possiamo **progettare il modello fin dall'inizio per essere amico della CPU**: piccolo dove serve, prevedibile, organizzato perché le cose giuste stiano in cache. Questo è ciò che rende le nostre idee possibili dove agli altri non riescono.

---

## In una frase

> Un LLM è una funzione fatta di tanti numeri. Su una CPU normale, scrivere testo una parola alla volta vuol dire **trascinare quei numeri dalla RAM per ogni parola**, e *quello* è il collo — non il calcolo. Andare veloci = **leggere meno byte, in ordine, tenendo in cache ciò che serve**. E siccome addestriamo da zero, possiamo costruire il modello apposta perché questo accada.

➡️ Con questo in testa, il documento [02 — Le cinque idee](02_Le_Cinque_Idee.md) spiega le soluzioni concrete.
