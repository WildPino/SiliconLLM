# F2: la banca su E, il risultato

2026-11-26. Fase 2 del [piano](PLAN.md), domande **D-0** e **D-R**. Protocollo [`F2_BANK_PROTOCOL.md`](F2_BANK_PROTOCOL.md)
(regole del §6, regola di futilita del §10, commit 05cd90f). Run in `results/sparse_capacity/f2_<braccio>_s<seed>/`,
verdetto in `f2_verdict.txt`, log della catena in `f2_chain.jsonl`.

**Esito: D-0 PASS, D-R NON MISURABILE.** Da m1 a m64 la banca abbassa il BPB del codice di 0,041 (8,1σ), in modo
monotono, e il guadagno resta uguale dopo la temperatura. La sonda di ragionamento sta sotto il suo controllo piu
forte in tutti i bracci.

**Riga sulla tesi.** A nucleo fisso (4,6 M), 60 M token e calcolo degli esperti fisso, una banca 64 volte la MLP
abbassa il BPB. Le fasi 3 e 4 sono giustificate. Il risultato non dice nulla sulle banche oltre la VRAM, ne sul
ragionamento.

## 1. I numeri (media dei due seed; dense e dense-fp un seed)

| Braccio | Attivi (router compreso) | BPB val | Top-1 | P62 BPB | K 2.048 | R primaria | Esperti effettivi (min) | Ore |
|---|---|---|---|---|---|---|---|---|
| m1 | 11,2 M | 1,0392 | 0,6115 | 1,0835 | 0,028 | 0,062 | 1,00 | 6,8 |
| m4 | 12,0 M | 1,0191 | 0,6173 | 1,0630 | 0,039 | 0,026 | 0,99 | 9,3 |
| m16 | 15,1 M | 1,0029 | 0,6221 | 1,0496 | 0,062 | 0,128 | 0,96 | 12,4 |
| m64 | 27,8 M | 0,9987 | 0,6229 | 1,0448 | 0,072 | 0,190 | 0,91 | 24,1 |
| dense | 10,9 M | 1,0409 | 0,6117 | 1,0835 | 0,050 | 0,250 | - | 6,0 |
| dense-fp | 10,9 M | 1,0274 | 0,6156 | 1,0702 | 0,052 | 0,224 | - | 5,9 |

- **σ_BPB:** misurata 0,0019 fra i seed, usata 0,005 (il pavimento R1).
- **D-0:** guadagno m1 → m64 +0,0406 BPB (8,1σ), nessun passo che sale. Sulla meta B a T scelta, +0,0437 (8,7σ): la
  temperatura sceglie T = 1, quindi non prende nulla. Top-1 +0,011. Su P62 (lettura) −0,039 BPB, top-1 +0,011.
- **Regola di futilita:** dopo m1 s0 e m64 s0 ha dato CONTINUE, senza stampare numeri.
- **K, il positivo piantato:** sale di +0,015 (5,1σ), quindi la banca memorizza e il verdetto su R e leggibile. Pero
  discrimina solo la rubrica da 2.048 voci: quella da 256 e satura (≥ 0,99) e quella da 16.384 e a zero in tutti i
  bracci.
- **R:** le celle primarie stanno fra 0,03 e 0,25. Il controllo piu forte, la regola locale, fa 0,583. I seed sono
  molto lontani: m16 fa 0,308 e 0,002 su n9 L3, m1 0,162 e 0,010. σ_R = 0,10. La cella n12 L5 e a zero.

## 2. Predizioni (§7) contro esiti

| Predizione | Esito |
|---|---|
| D-0 PASS, guadagno 0,03-0,08, oltre 6σ | **giusta**: 0,041, 8,1σ |
| K sale molto, soprattutto a 2.048 voci | giusta nel verso e nella rubrica, sbagliata nella taglia: da 0,028 a 0,072 |
| D-R PIATTA; rischio dichiarato: R al livello della regola locale, esito NON MISURABILE | si e avverato il rischio, e peggio: R sta sotto la regola locale |
| dense-fp sotto dense di 0,02-0,05 BPB | **sbagliata**: 0,0135. Pesi ternari e attivazioni a 63 livelli costano meno del previsto |
| dense e m1 entro 0,02 | giusta: m1 sotto dense di 0,0017 (val), pari su P62 |

## 3. Cosa dice, e cosa no

- **La banca lavora su questa architettura.** Il guadagno e monotono in E, e quasi tutto arriva gia a m16 (−0,036 dei
  −0,041).
- **Pero si sta appiattendo.** Ogni passo moltiplica la banca per 4 e pesa meno del precedente: −0,020, −0,016,
  −0,004. L'ultimo, m16 → m64, sta sotto 1σ.
  - Il gate della fase 4 chiede "il BPB scende ancora con log E, senza saturazione precoce". Questo e il primo segnale
    contro, a 60 M token.
  - Con piu token per esperto la curva puo riaprirsi: m64 vede 0,94 M token per esperto, m16 3,7 M. Non e misurato.
- **Lettura delle curve (non un gate, aggiunta dopo il verdetto).** Si usa il BPB sul val delle curve (262k token, ogni
  1.000 passi), con la media dei due seed:

  | Passo | 1.000 | 2.000 | 3.000 | 4.000 | 5.000 | 6.000 | 7.000 | 7.324 |
  |---|---|---|---|---|---|---|---|---|
  | m4 − m16 | −0,0195 | −0,0001 | +0,0050 | +0,0096 | +0,0131 | +0,0129 | +0,0146 | +0,0152 |
  | m16 − m64 | −0,0698 | −0,0085 | +0,0005 | +0,0016 | +0,0021 | +0,0044 | +0,0052 | +0,0055 |

  - I distacchi crescono per tutto il run, in tutti e due i seed (m16 − m64 alla fine: 0,0051 e 0,0059), e non si
    sono fermati al passo 7.324.
  - Quindi l'appiattimento puo essere scarsita di token: le banche piu grandi partono piu lente e a 60 M token non
    hanno ancora finito di staccarsi.
  - Non lo prova. Il lr scende a coseno verso la fine, e una parte della crescita puo venire dalla ricetta.
  - Per saperlo servono molti piu token per esperto: lo dice l'abbozzo della fase 4 nel piano.
- **Banca piu router, non la sola banca** (§8). Da m1 a m64 i parametri attivi passano da 11,2 a 27,8 M, e 16,8 M sono
  il router. Manca un braccio a router congelato, quindi la parte del router nel guadagno non si separa.
- **La bandiera rossa della 1a** era un indizio contro la banca dopo l'upcycle. Allenata da zero, contro una base con
  gli stessi token, la banca vince: la causa della 1a sta nella ricetta del rung-1, non nella banca. Questo e
  compatibile con l'upcycle come causa, ma non lo prova. La condizione scritta nella 1a non scatta: m1 non sta sopra
  dense.
- **R non misura il ragionamento a questa scala.** Sta sotto la regola locale, e un seed puo stare a zero mentre
  l'altro arriva a 0,3. Quindi D-R resta aperta. I bracci densi stanno sopra le banche (0,25 contro 0,06-0,19), ma con
  σ_R = 0,10 e un solo seed non e una lettura.
- **Il ritmo e pulito.** Nessun altro lavoro sulla CPU durante i run (`s2n_cpu_running` sempre falso). In tutto 122 h
  dall'inizio alla fine, contro le 122 previste. m64 costa 3,5 volte m1 perche il calcolo e denso su E (§8).

## 4. Conduzione

- **m1 s0 parte due volte.** Alle 09:11 del 21-11 un primo avvio, poi interrotto, e morto dopo 3 minuti, senza
  checkpoint. Alle 09:14 il run e ripartito da zero con lo stesso seed (stessa loss al passo 0). `train.jsonl` tiene
  le due righe `start`; il run valido e il secondo.
- Tutti i run vengono dallo stesso commit, pushato e senza modifiche, e fanno 7.324 passi.
- Pesi finali fuori da git: `E:/SiliconLLM_Artifacts/sparse_capacity/f2/f2_<braccio>_s<seed>/final_model.pt`, con
  lo SHA-256 nella riga `done` di ogni `train.jsonl`.

## 5. Decisione (PLAN, fase 2)

D-0 PASS e R non misurabile: **la fase 3 (il metodo) puo partire; prima della fase 4 serve una sonda R nuova**
(protocollo, §6).
- La fase 3 e un abbozzo nel piano. Il dettaglio va scritto a fine fase, con il via dell'utente (PLAN §6).
- **Proposta per la fase 3.** Il trainer con la banca nella RAM in streaming (1b) rifa m16 s0 della fase 2. I gate
  sono quelli del piano: BPB entro 2σ, curve sovrapposte, ritmo almeno al 50% del solo-GPU. m16 e il braccio piu
  economico che porta gia la maggior parte del guadagno.
- **Per la sonda R nuova,** quello che questo risultato insegna: le celle devono partire sopra la regola locale per
  costruzione, e la varianza fra seed va misurata prima di fissare la soglia.
