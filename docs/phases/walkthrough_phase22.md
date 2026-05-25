# Silicon Entropy Engine - Phase 22: Online Expert Mixing & Dynamic Backoff

In questa Phase abbiamo abbandonato l'idea rigida di un classificatore "Out Of Domain" in favore di una soluzione molto più pura dal punto di vista della teoria della compressione dati: l'**Online Mixture of Experts (MoE)** con Fixed-Share Weight Update.

Invece di costringere il modello a diagnosticare il dominio, abbiamo diviso la predizione in tre esperti di base e lasciato che sia il flusso stesso a decidere a chi assegnare il credito, byte per byte, in base alla cross-entropy.

## 1. Architettura MoE
Nel decoder simmetrico abbiamo istanziato tre esperti:
1. `p_see`: il modello neurale/fisico statico preaddestrato.
2. `p_uni`: unigramma dinamico (ottimo per rumore/shuffled o contesti alieni primitivi).
3. `p_bi`: bigramma dinamico (potente per prosa, markdown e codice non-C).

La probabilità miscelata è $P_{mix} = w_{see} P_{see} + w_{uni} P_{uni} + w_{bi} P_{bi}$.

Dopo che il Range Coder decodifica il target, calcoliamo la "loss" di ogni esperto ($-\log_2(p)$) e applichiamo un update esponenziale penalizzante:
$$ w_i \gets w_i \cdot \exp(-\eta \cdot loss_i) $$

> [!TIP]
> **Fixed-Share Update**
> Per impedire che un esperto muoia matematicamente dopo una lunga sequenza avversa, abbiamo iniettato una quota $share$:
> $$ w_i \gets (1 - share) \cdot w_i + \frac{share}{3} $$
> Usando di default $\eta = 0.03$ e $share = 0.001$, il modello può adattarsi rapidamente ai cambi di dominio senza dimenticare a breve termine.

## 2. Risultati Finali e Success Criteria

I risultati dell'audit Phase 22 battono letteralmente **QUALSIASI** lambda fisso su **TUTTI** i dataset, polverizzando i criteri di successo originali e preservando sempre l'integrità del decoding in roundtrip SHA-256.

### In-Domain (c_code.c)
* `dyn_0.0` (SEE fast): 2.1393 BPB
* **`dyn_moe`:** **2.0861 BPB**
* Obiettivo superato abbondantemente! Piuttosto che stare "entro +0.03 BPB", il MoE ha **migliorato** la compressione in-domain togliendo lo ~0.05 BPB al modello statico (grazie all'apporto chirurgico del bigramma sui pattern ripetitivi).

### Cross-Domain (promessi_sposi.txt)
* Best Fixed Lambda (`dyn_1.0`): 3.6486 BPB
* **`dyn_moe`:** **3.3163 BPB**
* Obiettivo Polverizzato: L'approccio neurale statico fa 5.9 BPB. Il bigramma dinamico puro 3.6. Il MoE fa 3.3. Ha capito esattamente come miscelare la grammatica universale del C col bigramma italiano.

### Cross-Domain (markdown_docs.md)
* Best Fixed Lambda (`dyn_0.5`): 4.1369 BPB
* **`dyn_moe`:** **4.0932 BPB**
* Obiettivo raggiunto: Battuta l'euristica della Phase 21 e battuto il miglior lambda fisso.

### Out-of-Domain (shuffled.bin)
* Best Fixed Lambda (`dyn_1.0`): 5.0998 BPB
* **`dyn_moe`:** **5.0131 BPB**
* Obiettivo superato: L'esperto `w_uni` riceve quasi tutto il credito su file stocastici privi di memoria di bigramma, permettendoci di battere persino il fallback manuale e tornando alle prestazioni pure del compressore di zero-ordine.

## Conclusione
Il Fixed-Share MoE è una rivoluzione per il Silicon Entropy Engine. 
Senza imporre al motore di decidere "se è fuori dominio", la punizione moltiplicativa $\exp(-\eta \cdot L_i)$ zittisce elegantemente la predizione SEE quando questa è molto sicura ma errata.
Il motore ha ora un *organo naturale* per tacere, imparato direttamente dallo stream, rigoroso, deterministico, e simmetrico. E produce la migliore compressione mai raggiunta finora su tutti e quattro gli split di riferimento.
