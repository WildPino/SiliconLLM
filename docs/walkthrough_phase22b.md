# Silicon Entropy Engine - Phase 22B: MoE Stability & Credit Audit

In questa fase abbiamo stressato l'architettura Online Mixture of Experts (MoE) per assicurarci che non fosse solo un "trucco" per fare una media su file omogenei, ma un vero meccanismo reattivo ai salti di dominio (domain shifts).

## 1. Exhaustive Fixed Lambda Sweep
Abbiamo testato esaustivamente le configurazioni statiche ($\lambda$ da 0.0 a 1.0 a passi di 0.1) per misurare la performance del MoE contro ogni singola miscela fissa testabile a posteriori.

Il risultato dimostra che **il MoE batte tutti i lambda fissi testati su tutti i dataset**:
* **c_code.c**: MoE = 2.08 BPB | Best Fixed ($\lambda=0.1$) = 2.12 BPB. *Il silicio migliora la grammatica C statica abbassando $\lambda$ solo localmente dove il bigramma eccelle.*
* **promessi_sposi.txt**: MoE = 3.31 BPB | Best Fixed ($\lambda=1.0$) = 3.64 BPB. *Vittoria netta (+0.33 BPB) grazie alla modulazione tra Unigram e Bigram.*
* **markdown_docs.md**: MoE = 4.09 BPB | Best Fixed ($\lambda=0.6$) = 4.12 BPB. 
* **shuffled.bin**: MoE = 5.01 BPB | Best Fixed ($\lambda=1.0$) = 5.09 BPB. *Il ritorno quasi perfetto allo zero-order unigramma.*

## 2. Multi-Domain File (Domain Shifts Audit)
Abbiamo concatenato i quattro dataset in due permutazioni (circa 1.9MB ciascuna) per studiare l'allocazione del credito nel tempo:
* `multi_domain_1.bin`: `c_code` $\to$ `prose` $\to$ `shuffled` $\to$ `markdown`
* `multi_domain_2.bin`: `shuffled` $\to$ `c_code` $\to$ `markdown` $\to$ `prose`

### Analisi dei Grafici (Credit Audit)
Guardando i plot estratti (salvati come PNG in `results/multi_domain_1_plot.png` e `2`):
1. **c_code**: Subito dopo l'entrata nel dominio C, la linea blu ($w_{see}$) schizza al 90-95% del peso e vi resta stabilmente. Il motore capisce che la sua fisica quantizzata è corretta.
2. **prose**: Entrando nel testo italiano, $w_{see}$ crolla brutalmente a zero. La linea verde ($w_{bi}$, il bigramma) conquista quasi l'80% del credito, supportata da una baseline di unigramma rosso ($w_{uni}$).
3. **shuffled**: Non appena inizia il blocco di byte stocastici, il bigramma viene sfiduciato, e la linea rossa ($w_{uni}$) schizza in pochi KB all'80-90% dominando la miscela. Il compressore smette di imparare contesti inutili e si salva rifugiandosi nelle probabilità globali di zero-ordine.
4. **markdown**: Qui si nota il vero "mixing". Il markdown alterna blocchi di prosa e snippet di codice. Il grafico mostra un intreccio continuo tra $w_{see}$ (che si attiva negli snippet) e $w_{bi}$ (che si attiva nella prosa), con scambi di leadership rapidi ma stabili localmente.

### Assenza di Deriva Numerica (Long Drift Test)
Nonostante il passaggio attraverso quasi 2 milioni di decodifiche con ricalcolo continuo `w_i = (1 - share) * w_i + share / 3`, la normalizzazione tiene perfettamente e il file originale multi-dominio è ricostruito al bit, superando il match **SHA-256**. L'esperto è perfettamente simmetrico.

## Conclusione
Il silicio ha ufficialmente abbandonato il "teatro cognitivo". Senza dover decidere che etichetta incollare al file, la fixed-share loss è sufficiente a fornire una marcia adattiva perfetta, stabile, e simmetrica. Questo Mixture of Experts in CPU è un mattone architetturale solido e pronto.
