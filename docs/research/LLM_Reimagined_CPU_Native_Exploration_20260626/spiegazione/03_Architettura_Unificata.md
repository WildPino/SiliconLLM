# 03 — L'architettura unificata

*Le 5 idee del documento precedente non sono una lista della spesa: convergono in **una sola architettura**. Questo è il vero risultato dello sprint.*

---

## L'intuizione di fondo: separa il "pensare" dal "sapere"

In un LLM tradizionale, in ogni strato, due cose sono **mescolate insieme**:
- il **calcolo** (le moltiplicazioni — il "pensare"),
- la **conoscenza** (i pesi, cioè ciò che il modello "sa" — il "sapere").

Sono inscindibili: ogni moltiplicazione tira dentro un mucchio di pesi. E siccome su CPU il problema è *spostare i pesi*, questo mescolamento è esattamente ciò che ci rallenta.

**La mossa:** **separarli.**

```
   ┌─────────────────────────────┐        ┌───────────────────────────────────┐
   │   IL "PENSARE"              │        │   IL "SAPERE"                     │
   │   (piccolo, veloce)        │        │   (enorme, ma toccato a malapena) │
   │                            │        │                                   │
   │   un blocco SSM ternario   │◄──────►│   una grande memoria da           │
   │   riusato più volte,       │ stato  │   consultare (tabella in RAM),    │
   │   STA IN CACHE             │ SSM    │   ma prendi solo le poche voci    │
   │                            │        │   che servono — e le hai          │
   │   = K2                     │        │   PREVISTE, quindi sono già       │
   │                            │        │   in arrivo                       │
   │                            │        │   = K1 + K3                       │
   └─────────────────────────────┘        └───────────────────────────────────┘
              ▲                                          
              │ entra/esce a "unità di compressione" (K4)
              │ e produce più parole per giro (K5)        
```

- **Il "pensare"** è un **piccolo nucleo** di calcolo (un blocco SSM ternario) che **sta sulla scrivania** (in cache) e viene riusato più volte — **K2**. Non rilegge mai nulla dalla RAM: è già lì. Questo è il pezzo veloce.

- **Il "sapere"** è una **memoria gigantesca** che vive nella RAM (anche 1 GB), ma di cui per ogni parola si tocca solo una manciata di voci — **K1**. Sarebbe lento (recupero a salti), ma il nucleo che pensa **predice quali voci serviranno** e le si va a prendere in anticipo — **K3**. Così la lentezza sparisce.

- **K4** regola come si entra e si esce (a ritmo di informazione), **K5** sforna più parole per ogni lettura.

---

## La frase che riassume tutto

> **"Rifattorizza l'LLM così che la parte pesante sia un *consultare* (lookup), non un *moltiplicare*."**

- Il **moltiplicare** ti obbliga a leggere **tutti** i pesi (è denso): ogni parola = trascinare l'intero modello dalla RAM.
- Il **consultare** tocca solo le **poche** voci giuste (è sparso): pochi byte. E se hai **previsto** quali, sono **già in cache** quando ti servono.

Il collante che fa funzionare tutto è **K3**: è ciò che trasforma una scommessa fragile ("speriamo che le cose giuste siano in cache") in un **obiettivo di addestramento** ("alleniamo il modello perché le cose giuste *siano* prevedibili e quindi prefetchabili"). Senza K3, K1 paga il pedaggio del recupero casuale. Con K3, non lo paga.

---

## Perché questa architettura è (modestamente) originale

**Non per i singoli mattoni.** Ognuno esiste già in letteratura — e l'abbiamo verificato con tre ricerche (vedi documento 04). Product-key memory, weight sharing, predittori di sparsità, indici appresi: tutto pubblicato.

**L'originalità sta in due cose:**

1. **La fusione.** Mettere insieme: nucleo-in-cache (K2) + memoria-da-consultare (K1) + uscita-come-recupero (sempre K1) + addestramento-alla-prevedibilità (K3), **in un solo sistema dove l'uno abilita l'altro**. In particolare: *la FFN e l'uscita del modello come un unico indice condiviso* — nessuno l'ha fatto. E *allenare il modello a rendersi prefetchabile* — nessuno l'ha fatto.

2. **Il bersaglio.** Tutto questo **progettato da zero per una CPU comune, a una parola alla volta**. La letteratura ottimizza sempre per GPU, o per grandi lotti (batch), o rattoppa modelli già fatti. La nicchia "CPU normale, una parola alla volta, da zero" è **sistematicamente vuota** — è la stessa forma di nicchia che già occupiamo sul recupero a 128K.

> In breve: **i pezzi sono noti, l'incastro e il bersaglio no.** Ed è un'originalità onesta — non abbiamo inventato fisica nuova, abbiamo ricomposto pezzi veri in un punto che nessuno ha occupato.

---

## Due regimi: oggi e domani

È importante non confondere **quando** ciascun pezzo paga:

- **Alla scala di oggi (il "giocattolo" TinyStories, pochi milioni di pesi):** il modello **sta già tutto in cache**. Quindi **K1 e K3 non mordono ancora** (non c'è niente di troppo grande da prevedere/prefetchare). L'unico pezzo che **già conviene e si può provare subito è K2** (il blocco riusato).

- **Alla scala del modello grande futuro (agentico, contesto lungo):** la memoria è troppo grande per la cache → **K1 e K3 diventano decisivi** (è lì che "consultare invece di moltiplicare" vale ordini di grandezza). K2 continua a valere.

Questa è una mappa, non un cantiere aperto: serve a **non precludere il domani** mentre oggi si lavora ad altro.

---

## Cosa NON è questa architettura (per onestà)

- **Non** è già costruita: è un disegno con stime, non un modello allenato.
- **Non** tocca il lavoro in corso (la "rampa recall run-6"): è l'asse del **modello grande futuro**.
- **Non** include la diffusione / generazione non-autoregressiva: l'abbiamo **provata e scartata** per la CPU (vedi documento 04). Né la memoria "olografica" VSA (morta in una fase precedente per limiti di capacità).

➡️ Le prove dietro queste affermazioni — incluse quelle scartate — sono nel documento [04 — Le prove e i prossimi passi](04_Prove_e_Prossimi_Passi.md).
