# Silicon Entropy Engine - Phase 21: Domain Calibration Layer

In questa Phase abbiamo implementato la strategia di Domain Calibration in due passi: Strumentazione (Phase 21A) e Router Minimalista (Phase 21B).

## 1. Phase 21A: Instrumentation & Telemetry
Abbiamo aggiunto in `benchmark18_coder.c` il logging della cross-entropy sia di SEE (`H_see`) che del modello dinamico (`H_dyn`), calcolati per ogni singolo byte in streaming prima della codifica. 

L'audit del nostro "Oracle" (ovvero un modello teorico che sceglie istantaneamente il migliore tra SEE e Dyn per ogni byte) dimostra l'enorme potenziale del gating locale:
* **Su c_code.c**: Oracle = **1.81 BPB** (contro 2.13 di SEE-only)
* **Su promessi_sposi.txt**: Oracle = **3.28 BPB** (contro 3.63 di Dyn-only e 5.97 di SEE-only)

> [!WARNING]
> **La Trappola dell'Entropia Bassa (Out-of-Domain)**
> L'analisi dei dati di telemetria ha confermato l'intuizione architetturale del piano. Su testo in italiano (Cross-Domain), il SEE spesso presenta una **bassissima entropia** ($H_{see}$ < $H_{dyn}$) perché è molto convinto delle sue predizioni, ma sbaglia sistematicamente. Infatti, nei casi in cui $H_{see}$ è molto inferiore a $H_{dyn}$ (SEE super-confidente), **il modello dinamico vince comunque il 73% delle volte**. Bassa entropia non significa "verità", significa solo "certezza appuntita".

Su *In-Domain* (C code), invece, $H_{see}$ - $H_{dyn}$ è un segnale predittivo fortissimo della vittoria: se SEE è incerto, vince la dinamica.

## 2. Phase 21B: Router Minimalista (Heuristic Gate)
Abbiamo implementato una primissima euristica (attivabile passando `--blend -1`), applicando anche un Exponential Moving Average (EMA) per evitare cambi bruschi (`lambda_state = 0.9 * lambda_state + 0.1 * lambda_raw`):
- Se `tail_mass` è elevata (> 0.4), spinge $\lambda \to 0.5$
- Se il bigramma dinamico è maturo (`count_bi > 50`) e $H_{dyn} < H_{see}$, spinge $\lambda \to 0.9$
- Altrimenti $\lambda \to 0.1$

### Risultati del Gating
I criteri di successo definiti sono stati rispettati (o ci siamo andati molto vicino) per questo router primitivo senza l'uso di target "dal futuro". Il decoder simmetrico riproduce il file perfettamente, e tutti i test hash SHA-256 passano al volo.

**In-Domain (c_code.c)**
* SEE `fast` ($\lambda=0$): 2.1393 BPB
* **SEE `dyn_auto`:** **2.1239 BPB**
* Obiettivo superato! Siamo perfino riusciti a **migliorare** il modello statico su C code. Il gate abbassa le perdite lasciando il compito alla dinamica solo quando SEE tentenna.

**Cross-Domain (promessi_sposi.txt)**
* SEE `fast` ($\lambda=0$): 5.9344 BPB
* **SEE `dyn_auto`:** **4.5206 BPB**
* `dyn_1.0`: 3.6486 BPB
* Obiettivo Raggiunto: abbiamo surclassato brutalmente il modello puramente statico. Non siamo ancora pari a `dyn_1.0`, il che è normale perché la nostra euristica protegge aggressivamente le predizioni SEE fiduciose (che qui sbagliano), ma l'adattamento è lampante.

**Cross-Domain (markdown_docs.md)**
* SEE `fast` ($\lambda=0$): 5.0532 BPB
* **SEE `dyn_auto`:** **4.4497 BPB**
* Obiettivo Raggiunto.

**Out-of-Domain (shuffled.bin)**
* SEE `fast` ($\lambda=0$): 7.7906 BPB
* **SEE `dyn_auto`:** 5.9011 BPB
* `dyn_1.0`: 5.0998 BPB
* Su dati totalmente stocastici fatichiamo ad avvicinarci a $\lambda=1.0$ perché la clausola `count_bi` basso impone di restare su SEE (e lo stocastico ha count_bi basso). 

## Conclusione
L'infrastruttura di telemetria è ora completa e genera dati per studiare un gate vero. L'euristica hard-coded prova che il "routing" locale migliora le performance, e perfino sul file *In-Domain* togliamo un margine significativo (0.015 BPB) in maniera del tutto unsupervised. Manca ancora il colpo di reni in Cross-Domain per difenderci dalla "tossicità" di un SEE over-confidente.
