# Sparse Capacity: indice

Una riga per protocollo o risultato, dalla piu recente. Ogni riga dice la fase, la domanda (D-0, D-M, D-R) e cosa cambia,
in al massimo 300 caratteri. Il piano e in [`PLAN.md`](PLAN.md).

| Data | Fase | Documento | Esito, contro le domande |
|---|---|---|---|
| 2026-11-17 | 0 | [`F0_QUANT_AUDIT.md`](F0_QUANT_AUDIT.md) | Per la fase 2: pesi ternari e scale per riga gia uguali al motore; mancano le attivazioni a 63 livelli (ingresso FFN e ingresso down, per esperto) e la scala fp32 dopo il prodotto. Il motore non ha un percorso per esperti sotto h32. |
| 2026-11-27 | 3 | [`F3_STREAM_RESULT.md`](F3_STREAM_RESULT.md) | D-M PASS: con la banca nella RAM ΔBPB −0,0005 (−0,1σ), top-1 +0,0005, ritmo 0,97. Costo 3,3 ns per parametro della banca. A 20-40 GB il ritmo scenderebbe a 0,47-0,31: la fase 4 deve togliere almeno meta del costo. |
| 2026-11-26 | 3 | [`F3_STREAM_PROTOCOL.md`](F3_STREAM_PROTOCOL.md) | D-M: protocollo registrato, run non ancora partito. m16 s0 con la banca nella RAM in streaming contro il run solo-GPU. Dry: un passo uguale al riferimento, 3,1 ns per parametro. Predizione: PASS, ma a 20-40 GB il ritmo scende a 0,48-0,32. |
| 2026-11-26 | 2 | [`F2_BANK_RESULT.md`](F2_BANK_RESULT.md) | D-0 PASS: m1 → m64 −0,041 BPB (8,1σ), monotono, top-1 +0,011; si appiattisce (m16 → m64 −0,004). D-R NON MISURABILE: R sotto la regola locale, seed instabili; K sale poco. Fase 3 possibile, sonda R nuova prima della fase 4. |
| 2026-11-18 | 1a | [`F1A_RUNG1_GATE_RESULT.md`](F1A_RUNG1_GATE_RESULT.md) | D-0: BANDIERA ROSSA, come predetto. Sul P62 la banca E e peggio del denso D di 0,016 BPB (+3,2σ), top-1 −0,003; la temperatura non prende nulla. Indizio, non risposta: raccomandata la fase 2 senza modifiche, che toglie upcycle e recall e ha la base m1. |
| 2026-11-18 | 2 | [`F2_BANK_PROTOCOL.md`](F2_BANK_PROTOCOL.md) | Protocollo D-0 e D-R registrato, run non lanciati (~122 h di 3060, aspettano E:). h8 top-128, base m1, m4-m64, 2 seed; sonde rubrica e cammino minimo. Predizione: D-0 PASS, D-R PIATTA. Revisore: 13 obiezioni, 2 bloccanti, corrette. |
| 2026-11-18 | 1b | [`F1B_CROSSOVER_RESULT.md`](F1B_CROSSOVER_RESULT.md) | D-M: GRIGIO. La CPU non batte lo streaming ideale in nessuna cella del gate (predetto ANGOLO a h8: sbagliato, la CPU e 1,2-2,6x sotto il modello). Metodo = streaming; con Adam il collo e l'aggiornamento sulla CPU (2 ns/param). D-R: si decide se continuare. |
| 2026-11-17 | 1b | [`F1B_CROSSOVER_PROTOCOL.md`](F1B_CROSSOVER_PROTOCOL.md) | Protocollo D-M registrato, run non lanciato. Predizione: ANGOLO CPU solo a h8 (r* 2-4), a h1 la CPU perde gia a r = 1; guadagno sul passo con instradamento uniforme <= 2%. Revisore: 4 obiezioni bloccanti, corrette. |
| 2026-11-17 | 0 | [`LITERATURE.md`](LITERATURE.md) | D-M: esperti su CPU gia fatti per il fine-tuning su server (KT-FT, OOM); lo streaming batte la CPU 26-60 volte; libero l'angolo "consumer, senza AMX, posto per esperto secondo r". D-R: nei MoE classici la banca da conoscenza; con micro-esperti non misurato. |
| 2026-11-17 | 0 | [`PLAN.md`](PLAN.md) | Piano iniziale: domande D-0, D-M, D-R; fasi 0-5; la fase 1 (gate del rung-1, misura r*) e la prossima. |
