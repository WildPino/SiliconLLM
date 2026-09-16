# STRAT-01 — screen dei nuovi donor sparse/hybrid

**Stato: screening documentale, 2026-09-16.** Nessun peso scaricato, nessun
benchmark, test di qualità, export, training o uso di T4. Questo è il primo
passaggio R1.0 della [roadmap strategica](../../STRATEGIC_10B_20260916/ROADMAP.md),
non un `PASS` sperimentale né una preregistrazione retroattiva. Integra lo
[screen dei donor precedenti](TARGET_DONOR_FOLLOWUP_2026-09-16.md) senza
riaprire OLMoE, StdMoE14B, E63 o gli altri esperimenti chiusi.

## Identità e compatibilità

| Donor base pretrained | Revisione HF consultata | Parametri distinti nel safetensors metadata | Accesso/licenza dichiarata | Architettura e gap rispetto a `phase60/engine.c` |
|---|---|---:|---|---|
| [Granite-4.0-H-Tiny-Base](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base) | [`f95c8e83b06c12f877486f18d2e2d109ab252bd0`](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base/blob/f95c8e83b06c12f877486f18d2e2d109ab252bd0/config.json) | 6,939,037,248 | Pubblico; Apache-2.0 | 36 Mamba2 + 4 attention, MoE top-6/64 e shared FFN a ogni layer. L'SSM selettivo/conv nativo non è Mamba2 e shape/router/attention/loader non sono quelli del donor. |
| [LFM2.5-8B-A1B-Base](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base) | [`6cfb83b04b0f0f12a64cb38ab553275d3ffe95d8`](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base/blob/6cfb83b04b0f0f12a64cb38ab553275d3ffe95d8/config.json) | 8,467,856,832 | Pubblico; card `lfm1.0`, API `other`: leggere i termini prima dell'uso | 18 gated short-conv + 6 GQA attention; 2 FFN densi, poi 22 MoE top-4/32 con sigmoid, expert bias e rinormalizzazione. La conv nativa non equivale al blocco LFM; router/shape/loader non compatibili. |

I conteggi totali vengono dal campo `safetensors.total` delle API Hugging
Face [Granite](https://huggingface.co/api/models/ibm-granite/granite-4.0-h-tiny-base/revision/f95c8e83b06c12f877486f18d2e2d109ab252bd0)
e [LFM](https://huggingface.co/api/models/LiquidAI/LFM2.5-8B-A1B-Base/revision/6cfb83b04b0f0f12a64cb38ab553275d3ffe95d8)
alla revisione indicata. Sono *parametri memorizzati distinti*, non
byte/token, né una verifica indipendente del contenuto dei pesi. I due
checkpoint sono **sotto** il target indicativo di ~10B: possono al massimo
essere pilot o prove di forma, non una dimostrazione finale. Granite è più
piccolo, LFM più vicino al target; nessuna delle due qualità sul corpus del
progetto è stata misurata.

`benchmarks/phase60/engine.c` ha già SSM selettivo e convoluzione, ma usa
`D=256`, `L=6`, SWA locale, MoE top-8 e dimensioni compile-time. Preservare
esattamente questi donor richiede una **estensione/port** del runtime C; non
basta cambiare il file dei pesi. Se il vincolo è l'SSM v1 nativo *senza*
estensione, occorre invece sostituire e addestrare i mixer: è un'altra
milestone qualitativa, non una conversione lossless. Il
[`donor_engine.c`](../../../../benchmarks/donor_adaptation/engine/donor_engine.c)
Transformer è un runtime separato e non risolve automaticamente quel bridge.

## Contabilità dei pesi caricati per token

Ipotesi esplicite: batch 1, decoding autoregressivo, una lettura di payload
per peso lineare attivo, tutti gli expert selezionati valutati, head completa
per ogni token. L'embedding tied si conta **una volta in RAM**, ma la head
si applica comunque a ogni token. Sono esclusi bias/norm piccoli, scale,
metadata/padding, KV e stato/attivazioni, traffico amplificato, compute di
scan/router/softmax, sincronizzazione e sampling. Questa è una contabilità
di *pesi attivi*, non un'osservazione del traffico DRAM effettivo: cache e
riuso inter-token possono alterarlo.

| Organo, pesi/token | Granite | LFM2.5 |
|---|---:|---:|
| FFN routed | 566,231,040 | 968,884,224 |
| FFN shared | 188,743,680 | — |
| FFN dense primi 2 layer | — | 88,080,384 |
| Router | 3,932,160 | 1,441,792 |
| Proiezioni Mamba2 / conv | 526,417,920 | 301,989,888 |
| Kernel convolutivi | 479,232 | 110,592 |
| Proiezioni attention | 25,165,824 | 62,914,560 |
| Head vocab completa | 154,140,672 | 262,144,000 |
| **Totale dei grandi organi** | **1,465,110,528** | **1,685,565,440** |

Dimensioni congelate da config e controllate contro le implementazioni
ufficiali [Granite](https://github.com/huggingface/transformers/blob/main/src/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py)
e [LFM2 MoE](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2_moe/modeling_lfm2_moe.py)
(il codice linkato è `main`, non pin della libreria): Granite `D=1536,
L=40, 36 Mamba2, 4 attention, E=64, k=6, F=512, F_shared=1024,
V=100352`; LFM `D=2048, L=24, 18 conv, 6 GQA, E=32, k=4,
F_moe=1792, F_dense=7168, V=128000`.

Formule: routed `L_moe×k×3DF`; shared/dense `L×3DF`; router
`L_moe×ED`; attention `L_att×D×(2QO+2KVO)`; head `VD`.
Granite Mamba2: `36×[D×(2D+(2D+2×128×1)+48)+2D×D]`
per in/out projection e `36×(2D+256)×4` per depthwise conv.
LFM conv: `18×(3D²+D²)` e `18×D×3`.
Sostituendo `k` con `E`, i subtotali dei grandi organi **memorizzati**
sono 6,938,677,248 (Granite) e 8,467,755,008 (LFM), rispettivamente
360,000 e 101,824 sotto i totali safetensors: residuali compatibili con
norm, bias e altri tensori piccoli. Questo controllo indipendente evita di
confondere *totale stored* con *charged/token*; non certifica ogni tensore.

## Solo sensibilità del payload, non previsione di tok/s

`W8=1 byte`, `W4=0.5 byte`, `W2=0.25 byte` per peso lineare attivo,
uniformemente: sono **scenari**, non checkpoint quantizzati né formati
valutati. La colonna GB/s è il solo payload pesi a 50 tok/s; il tempo a
40 GB/s è un'ipotesi ideale `byte/40`, non un rate misurato su questi
layout/kernel. La roadmap riserva 14/20 ms al traffico per avere 30% di
margine a 50 tok/s.

| Donor | W8 GB/token | W4 GB/token; GB/s a50; ms a40 | W2 GB/token; GB/s a50; ms a40 |
|---|---:|---:|---:|
| Granite | 1.465111 | 0.732555; 36.628; 18.314 | 0.366278; 18.314; 9.157 |
| LFM2.5 | 1.685565 | 0.842783; 42.139; 21.070 | 0.421391; 21.070; 10.535 |

Con **40 GB/s ipotetici**, W4 già sfora il margine di 14 ms per entrambi;
LFM sfora perfino i 20 ms senza alcun altro costo. A 28 GB/s, il payload
W4 richiederebbe 26.16/30.10 ms e quello W2 13.08/15.05 ms
(Granite/LFM). W2 apre una *possibilità aritmetica* a 50, non una
promozione: deve preservare qualità, sostenere quel rate **nel suo kernel
reale**, e pagare stato, KV, router e glue. A 100 tok/s, anche W2 a 40
GB/s lascia solo 0.84 ms a Granite e sfora i 10 ms per LFM. La banda
effettiva per piccoli expert gather può essere molto inferiore a 40 GB/s;
non trasferire E63, che è un flusso **sintetico mixed-format**, a W8/W4/W2.

## Decisione e prossimo gate

- **Nessun donor promosso oggi a implementazione/training.** Non è una
  prova d'impossibilità, ma evita spesa T4 e un port fondato sul solo nome
  `A1B`.
- **Granite** è il pilot di compatibilità più interessante soltanto se si
  accetta un checkpoint ~6.94B e si definisce prima un formato/operatore
  che possa realisticamente avvicinare W2 senza distruggere la qualità.
- **LFM2.5** resta un'alternativa più grande, ma il W4 uniforme non entra
  nel budget ipotizzato neppure senza overhead. Prima di qualsiasi peso o
  implementazione, servono verifica dei termini di licenza e una proposta
  esplicita di riduzione traffico/precisione per organo.
- Il prossimo brief, se si decide R1, deve scegliere **un** donor/revisione,
  definire semantica C e baseline teacher, ipotesi W4/W2 per organo con
  scale e RAM loader, qualità paired sul corpus del progetto e stop gate.
  Non lanciare STRAT-02/T4 dal solo presente screen. R2 (training congiunto
  del denso) rimane il pivot se il bridge/qualità sparse non supera il gate.
