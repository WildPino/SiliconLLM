# F1a: il gate 1 del rung-1, il protocollo

2026-11-18. Fase 1a del [piano](PLAN.md), domanda **D-0**, solo CPU. Runner `benchmarks/sparse_capacity/sc1a.py`,
verdetto `sc1a_verdict.py` con le regole del §3, scritte qui prima della lettura di P62.

**Domanda.** Sul testo P62 code-val, il rung-1 con la banca batte la sua versione densa?

**Riga sulla tesi.** E un indizio su D-0, non la risposta (PLAN §5): i tre punti vengono dallo stesso curriculum, a
token diversi. La risposta pulita e la fase 2.

## 1. Cosa si confronta

I checkpoint di fine stadio del run principale del rung-1 (`D:/_ktmp/mainrun_final/stages_main_V2048_r26_CE/`):

| Stadio | Passo | Cosa e | lr |
|---|---|---|---|
| D | 137.328 | denso, MLP ternaria (QAT completo) | 1,5e-3 costante |
| E | 173.949 | D piu l'upcycle a E32 × h128 top-8 e lo slot di recall, sequenze da 2.048 | 1,5e-3 costante |
| F | 183.104 | E con lr × 0,1 per l'ultimo 5% | 3e-4 |

- **Confronto primario: E contro D.** Sono gli unici due allo stesso lr. F e una lettura: la discesa del lr a fine run
  abbassa il BPB da sola, e D non l'ha avuta.
- **Gia visto, e dichiarato:** i checkpoint contengono il BPB di fine stadio sul val interno (200k token). E vale
  0,9177, D 0,9037 (E peggio di 0,014) e F 0,8085. P62 non e mai stato letto su D ed E.

## 2. Misure e controlli

- **P62:** tutto il file (549.473 token, 1.500.000 byte), finestre da 512 come nel trainer, ogni byte una volta. Il
  conto dei byte deve tornare esatto. Accanto, l'accuratezza top-1.
- **Controllo piantato:** il runner ricalcola il val interno e deve ritrovare il numero registrato entro 0,002. Prova
  fatta prima del push: D 0,90371, E 0,91770 e F 0,80851, tutti uguali al registrato.
- **Controllo banale, la temperatura:** T scelta sulla prima meta delle finestre di P62, BPB sulla seconda a T = 1 e a T.
- **σ** = 0,005 (calibrazione R1); con un seme solo non c'e altro.

## 3. Esiti (registrati)

| Esito | Condizione | Decisione |
|---|---|---|
| VOID | il val interno non torna entro 0,002, o i byte di P62 non tornano | si corregge il runner |
| **BANDIERA ROSSA** | P62(E) − P62(D) > +2σ (+0,010) | prima della fase 2, una pagina con una raccomandazione all'utente (PLAN, fase 1a) |
| PARI | entro ±2σ | la fase 2 parte |
| VERDE | P62(E) − P62(D) < −2σ | la fase 2 parte |

**Predizione.** BANDIERA ROSSA, con E sopra D di 0,01-0,02 come sul val interno, e F sotto D di circa 0,08 per la
discesa del lr. E la predizione contro lo scopo (regola 5): cambia solo quando parte la fase 2, non se parte. La causa
piu probabile e il costo dell'upcycle, e la fase 2 parte da zero senza upcycle.

## 4. Cosa toglie al design

| Elemento | Qui | Perche |
|---|---|---|
| Un denso con gli stessi token di E | non esiste: il braccio `dense_paired` non e mai stato allenato | la fase 2 lo fa (m1 contro m64) |
| Stesso lr per F e D | no | F resta una lettura |
| Una sola variabile fra D ed E | upcycle, recall, sequenze da 2.048 e 36.621 passi in piu | e un indizio, non un gate su D-0 |
| fp16 su T4 (training) | fp32 su CPU | il controllo piantato riproduce il val al quinto decimale |
| Seeds | uno | σ viene da R1 |

## 5. Esecuzione

`python benchmarks/sparse_capacity/sc1a.py --run`, poi `python benchmarks/sparse_capacity/sc1a_verdict.py <cartella>`.
- **Risorse:** solo CPU, 4 thread, circa 50 minuti. La GPU resta a S2N.
- **Rifiuti:** il runner non parte se il protocollo e il codice non sono committati e pushati.
- **Artefatti:** `results/sparse_capacity/f1a_<data>/` (`config.json`, `stages.jsonl`), con gli SHA-256 di P62 e dei
  tre checkpoint.
