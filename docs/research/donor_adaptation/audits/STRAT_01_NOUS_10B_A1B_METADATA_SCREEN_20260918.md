# STRAT-01 — NousResearch 10B-A1B: metadata screen

**18 settembre 2026. Solo metadati; nessun peso scaricato, codice remoto eseguito, benchmark, T4 o port C.** Decisione: lead architetturale da verificare, **non** donor qualificato né autorizzazione implicita ad acquisire i pesi.

## Identità e fonti primarie

- Repository: [NousResearch/moe-10b-a1b-8k-wsd-lr3e4-1t](https://huggingface.co/NousResearch/moe-10b-a1b-8k-wsd-lr3e4-1t).
- Revisione fissata: [`e848be60cde857798081087287be1fffaab0c81f`](https://huggingface.co/NousResearch/moe-10b-a1b-8k-wsd-lr3e4-1t/tree/e848be60cde857798081087287be1fffaab0c81f), ultimo aggiornamento API 2026-04-01T18:45:48Z.
- [API metadata](https://huggingface.co/api/models/NousResearch/moe-10b-a1b-8k-wsd-lr3e4-1t/revision/e848be60cde857798081087287be1fffaab0c81f), [config](https://huggingface.co/NousResearch/moe-10b-a1b-8k-wsd-lr3e4-1t/blob/e848be60cde857798081087287be1fffaab0c81f/config.json), [safetensors index](https://huggingface.co/NousResearch/moe-10b-a1b-8k-wsd-lr3e4-1t/blob/e848be60cde857798081087287be1fffaab0c81f/model.safetensors.index.json).
- Il repository espone quattro shard safetensors, config e tokenizer, ma l'inventario API non comprende un `README.md`; `cardData.license` è assente. **Licenza del checkpoint e provenienza qualitativa restano da chiarire prima di promozione.** Non dedurre diritti dal nome dell'organizzazione o dal codice Qwen3.

L'API safetensors riporta **10.489.144.832 elementi BF16**. L'index dichiara **20.978.289.664 byte**, esattamente due byte per elemento. Sono pesi memorizzati, non attivi per token; il conteggio non dimostra che siano tutti parametri funzionalmente distinti o ben addestrati.

## Geometria attiva nominale

La config pin-nata dichiara `Qwen3MoeForCausalLM`, 32 layer, hidden 1536, 24 query head e 4 KV head di dimensione 128, 128 expert per layer con top-8, expert intermediate 512, vocabolario 151.936, head **untied**, contesto massimo 8192, e nessun layer MLP-only. Il ledger sottostante assume che ogni token esegua tutti i 32 layer, la completa Q/K/V/O attention, otto expert per layer, router e l'intera projection di output. Norm, input embedding lookup, RoPE, scale, KV e traffico effettivo sono esclusi. È derivazione dalla config e dai nomi dell'index, **non** misura del runtime.

| Organo | Calcolo | Pesi attivi/token |
|---|---|---:|
| Expert selezionati | `32×8×3×1536×512` | 603.979.776 |
| Attention Q/K/V/O | `32×1536×(2×24×128 + 2×4×128)` | 352.321.536 |
| Router | `32×1536×128` | 6.291.456 |
| Head untied | `1536×151936` | 233.373.696 |
| **Somma organi grandi** | | **1.195.966.464** |

| Ipotesi uniforme | Byte payload/token | GB/s di solo payload a 50 tok/s |
|---|---:|---:|
| W8 | 1.195.966.464 | 59,7983 |
| W4 | 597.983.232 | 29,8992 |
| W2 | 298.991.616 | 14,9496 |

Il W4 teorico è circa 63,80 MB/token sotto il ledger W4-all dello StdMoE già misurato (`661.782.528 B/token`), ma non crea ampio margine. Con il limite di progetto di 14 ms per streaming a 50 tok/s richiederebbe **42,71 GB/s per il solo payload**; con i 20 ms interi, **29,90 GB/s** prima di metadata, KV, decode, router/glue e compute. Il vecchio E30 a 36,30 GB/s è un riferimento di questa macchina, **non** una previsione del kernel Qwen3-MoE né un'impossibilità fisica generale. W2-BF16 PTQ fallì sullo StdMoE; non trasferire quel fallimento automaticamente a questa distribuzione di pesi, né trattare W2 come qualità preservata.

## Provenienza del pretraining: non confondere due evidenze

Il nome `...-1t`, le commit di training (HEAD mostra `step 18994`) e la presenza di pesi sono evidenza che il checkpoint esiste; **non** verificano da sole un numero finale di token, la qualità di questa precisa revisione o il suo stato finale. Una [discussione pubblica](https://huggingface.co/NousResearch/moe-10b-a1b-8k-wsd-lr3e4-1t/discussions/2) rimanda a una run Psyche, ma non è una model card né un audit del checkpoint.

Il [paper Nous Research su Token Superposition Training](https://arxiv.org/pdf/2605.06546) riporta una baseline 10B-A1B Qwen3-like addestrata su 1,05T token e una variante TST su 2T, con loss e benchmark zero-shot. **Non ho trovato nel paper un binding SHA o un link che identifichi questa revisione HF come uno dei checkpoint valutati.** Inoltre il paper descrive TorchTitan/B200 per gli esperimenti, mentre la discussione del repository rimanda a Psyche: non attribuire i numeri del paper a `e848be6` senza prova. La stessa architettura e l'organizzazione sono indizi, non identità sperimentale.

## Disposizione e prossimo gate

Questo è un lead più aderente al target di scala e alla semantica Transformer standard rispetto ai candidati sub-10B, ma non è pronto per una costosa acquisizione/port. Il prossimo controllo utile è **documentale e binario**: fonte ufficiale della licenza applicabile ai pesi, identificazione della run e del token/step finale per lo SHA pin-nato, e almeno una valutazione verificabile di quello stesso SHA. In assenza di questi, un download da ~21 GB e un port C rischiano di inseguire una qualità sconosciuta. Se il binding supera il gate, servono comunque teacher BPB locale, formato qualitativamente valido, byte/rate reali e parità del medesimo artefatto in `engine.c`; lo screen non promuove nessuno di questi gate.
