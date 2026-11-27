# F3: il metodo, la banca nella RAM in streaming, il risultato

2026-11-27. Fase 3 del [piano](PLAN.md), domanda **D-M**. Protocollo: [`F3_STREAM_PROTOCOL.md`](F3_STREAM_PROTOCOL.md)
(registrato prima del run, `f91e0b4`). Verdetto di `sc3_verdict.py`, copiato in
`results/sparse_capacity/f3_verdict.txt`. Run: `results/sparse_capacity/f3_stream_m16_s0/`, commit `130a22b`, dal
26-11 18:31 al 27-11 07:34, 12,62 h, senza riprese.

**Esito: PASS.** Il trainer con la banca nella RAM allena come quello solo-GPU della fase 2, al 97% del suo ritmo.

**Riga sulla tesi.** E un test di meccanismo, come dichiarato. Dice che il metodo e giusto sul percorso che servira
oltre la VRAM (pesi ripresi nel backward, un buffer grande come uno strato) e quanto costa: 3,3 ns per parametro
della banca. Non mostra una banca oltre la VRAM: quella e la fase 4.

## 1. Gate

| Condizione | Misura | Soglia | Esito |
|---|---|---|---|
| Passo 0 uguale al riferimento | loss 7,7810078 e norma 0,71231025, differenza relativa 0 e 0 | ≤ 1e-6 | ok |
| Equivalenza del piano | ΔBPB val −0,0005 (−0,1σ); curva al peggio 0,0042 (passo 2.000) | ≤ 0,010 | PASS |
| Equivalenza stretta, da un lato | ΔBPB −0,0005 | ≤ 0,0054 | PASS |
| Ritmo | 1.321 contro 1.363 token/s: 0,969 | ≥ 0,5 | PASS |

- **BPB e top-1:** stream 1,0034 contro 1,0039; top-1 0,6222 contro 0,6217 (+0,0005).
- **Controllo banale (temperatura sulla prima meta del val):** 1,0257 contro 1,0265. Lo stesso scarto: la temperatura
  non sposta il confronto.
- **Curve** (stream − riferimento, ogni 1.000 passi): +0,0009, −0,0042, −0,0034, −0,0015, +0,0006, −0,0006, −0,0020,
  −0,0018. Restano dentro lo scarto fra i due seed della fase 2 (al massimo 0,0073).
- **La predizione** (PASS, |ΔBPB| ≤ 0,005, ritmo 0,93-0,99) e giusta.

## 2. Letture (non gate)

- **Tempi del passo** (mediana, s): 6,19 in tutto; raccolta degli esperti 0,176 (3 per strato), ritorno dei gradienti
  0,060, AdamW sull'host 0,093, AdamW sulla GPU 0,002.
- **Costo del metodo:** 0,330 s a passo, 3,27 ns per parametro della banca (dry: 3,14). E il 5% del passo a m16.
- **U:** fra 2.035 e 2.048 esperti per strato su 2.048. Il trasferimento e quasi pieno, come previsto.
- **P62:** 1,0501 contro 1,0499. **K:** 0,349 contro 0,350.
- **R:** 0,008 contro 0,254. Non e un segnale del metodo: R e bimodale fra i seed gia nella fase 2 (m16 s1: 0,001) e
  D-R e stato dichiarato non misurabile con questa sonda.
- **Proiezione per la fase 4** (lineare, limite inferiore del costo): a 20 GB di banca con lo stato +4,1 s a passo,
  ritmo contro il passo di m1 (3,62 s) al massimo 0,47; a 40 GB +8,2 s, al massimo 0,31.

## 3. Cosa dice

- **Il metodo e giusto.** Il passo 0 e identico e un passo dallo stesso stato e identico (dry, protocollo §8). Dopo
  7.324 passi il BPB e a 0,0005 dal riferimento, entro il rumore di seed.
- **Il costo non e il trasferimento sul bus ma il lavoro per parametro:** raccolta e copia (53%), AdamW sull'host
  (28%), ritorno dei gradienti (18%). Tutto scala con la banca, non con il calcolo attivo.
- **La predizione contro lo scopo regge:** a 20-40 GB il ritmo scende sotto lo 0,5 del piano. La fase 4, cosi com'e,
  fallirebbe il suo gate del ritmo.

## 4. Raccomandazione

**La sonda R nuova, come da piano, e poi il protocollo della fase 4 con due cose in piu.**
- Togliere almeno meta del costo per parametro con le leve gia note:
  - pesi in bf16 sul bus;
  - due passaggi dei pesi a passo invece di tre;
  - trasferimenti sovrapposti al calcolo;
  - Adagrad per riga sull'host, con il suo gate di qualita.
- Un dry che lo misura prima del run.
- **Il ritmo "solo-GPU" della fase 4** si misura contro il passo di m1, che ha lo stesso calcolo attivo tutto sulla
  GPU. Da registrare nel piano.

**Cosa la smentirebbe:** se a 20 GB il costo misurato, con le leve, restasse sopra i 3,6 s a passo.
