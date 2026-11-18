# F1a: il gate 1 del rung-1, il risultato

2026-11-18. Fase 1a del [piano](PLAN.md), domanda **D-0**. Protocollo [`F1A_RUNG1_GATE_PROTOCOL.md`](F1A_RUNG1_GATE_PROTOCOL.md)
(cab3e1c), run `results/sparse_capacity/f1a_20261118_181303/` (`config.json`, `stages.jsonl`, `verdict.txt`).

**Esito: BANDIERA ROSSA, come predetto.** Sul P62 lo stadio con la banca (E) e peggio del denso (D) di 0,016 BPB
(+3,2σ).

## 1. I numeri

| Stadio | Passo | Parametri | Val interno (registrato / ricalcolato) | P62 BPB | Top-1 | T scelta | P62 meta B, T = 1 / T scelta |
|---|---|---|---|---|---|---|---|
| D | 137.328 | 10,92 M | 0,90371 / 0,90371 | 1,02803 | 0,5929 | 1,00 | 1,04827 / 1,04827 |
| E | 173.949 | 29,99 M | 0,91770 / 0,91770 | 1,04412 | 0,5903 | 1,00 | 1,06453 / 1,06453 |
| F | 183.104 | 29,99 M | 0,80851 / 0,80851 | 0,92766 | 0,6230 | 1,00 | 0,94796 / 0,94796 |

- **E − D:** +0,0161 BPB (+3,2σ), top-1 −0,0026. Sulla meta B a T scelta, +0,0163. Predizione: da +0,01 a +0,02.
- **F − D (lettura):** −0,100 BPB, top-1 +0,030. Predizione: circa −0,08. F ha la discesa del lr, D no.
- **Validita:** il val interno torna al quinto decimale nei tre stadi. I byte di P62 tornano (1.499.998 contati piu 2
  del primo token). F ritrova il P62 gia noto del run principale (0,9277).
- **Controllo banale:** la temperatura sceglie T = 1 in tutti e tre gli stadi, quindi non prende nulla. Il distacco
  E − D non e un problema di calibrazione.

## 2. Cosa dice

- Allo stesso lr, il rung-1 con la banca non batte la sua versione densa. Ha 2,7 volte i parametri e 36.621 passi in
  piu, ed e peggio sia in BPB sia in top-1.
- Il guadagno di F viene dalla discesa del lr, che D non ha avuto. Non e un argomento per la banca.
- Il confronto ha quattro differenze insieme: upcycle da un denso gia allenato, slot di recall, sequenze da 2.048 e
  passi in piu. Il risultato non dice quale delle quattro costa.
- Non dice neppure se una banca allenata da zero perde contro un denso con gli stessi token. Questa e la domanda D-0,
  e la fase 2 la misura con la base m1.

## 3. Raccomandazione

**Far partire la fase 2 come registrata (F2_BANK_PROTOCOL, 9c4baf4), senza modifiche.**

- La fase 2 toglie le quattro differenze della 1a:
  - banca allenata da zero, senza upcycle e senza recall;
  - stessa finestra per tutti i bracci;
  - m1 vede gli stessi token e ha la stessa precisione delle banche.
- La bandiera rossa alza la probabilita di un D-0 FAIL, ma non cambia il design: il gate della fase 2 sta gia contro
  m1, non contro un denso piu povero.
- Un esito sfavorevole e gia previsto dal piano. Se m4-m64 non scendono sotto m1 di almeno 4σ, D-0 e FAIL e il
  programma si ferma come scritto.
- **Costo:** circa 122 h di 3060, dopo la catena GPU di S2N.

**Cosa la smentirebbe.** Un motivo per credere che la fase 2 erediti la causa della 1a. Ad esempio, un difetto che
colpisce ogni banca allenata con la ricetta del rung-1 e non il denso. In quel caso perderebbero tutte le banche,
m1 compreso, e la base della fase 2 non distinguerebbe la capacita dalla ricetta. Il braccio `dense` della fase 2
(ternario, stessi token) misura proprio questo: se m1 sta sopra `dense` di oltre 2σ, la lettura di D-0 va fatta
anche contro `dense`, e lo si scrive nel risultato.
