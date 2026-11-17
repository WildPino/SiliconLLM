# Sparse Capacity: indice

Una riga per protocollo o risultato, dalla piu recente. Ogni riga dice la fase, la domanda (D-0, D-M, D-R) e cosa cambia,
in al massimo 300 caratteri. Il piano e in [`PLAN.md`](PLAN.md).

| Data | Fase | Documento | Esito, contro le domande |
|---|---|---|---|
| 2026-11-17 | 0 | [`F0_QUANT_AUDIT.md`](F0_QUANT_AUDIT.md) | Per la fase 2: pesi ternari e scale per riga gia uguali al motore; mancano le attivazioni a 63 livelli (ingresso FFN e ingresso down, per esperto) e la scala fp32 dopo il prodotto. Il motore non ha un percorso per esperti sotto h32. |
| 2026-11-17 | 0 | [`LITERATURE.md`](LITERATURE.md) | D-M: esperti su CPU gia fatti per il fine-tuning su server (KT-FT, OOM); lo streaming batte la CPU 26-60 volte; libero l'angolo "consumer, senza AMX, posto per esperto secondo r". D-R: nei MoE classici la banca da conoscenza; con micro-esperti non misurato. |
| 2026-11-17 | 0 | [`PLAN.md`](PLAN.md) | Piano iniziale: domande D-0, D-M, D-R; fasi 0-5; la fase 1 (gate del rung-1, misura r*) e la prossima. |
