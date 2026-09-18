# STRAT-01 GigaChat 3.1 — pilot MTP GGUF/CPU (18 settembre 2026)

## Stato e confine della prova

**REFERENCE_PILOT, non gate finale.** Un target pretrained GigaChat 3.1
Q4_K_M e il suo layer MTP BF16 della revisione pin-nata hanno effettuato
speculative decoding nello stesso processo `llama.cpp` CPU. Il draft è stato
caricato da un GGUF *separato*, invocato e ha prodotto token accettati. Non
esiste ancora una prova di qualità del target Q4 rispetto al BF16, di logits
parity MTP rispetto a vLLM, di rate con build CPU ottimizzato, né un port nel
nostro `benchmarks/phase60/engine.c`. Il requisito ≥50 tok/s **non è passato**.

Questo addendum supera soltanto le precedenti frasi storiche «nessun GGUF
draft scritto» e «grafo MTP Q diretto ancora da implementare» dello
[screen metadata](STRAT_01_GIGACHAT31_10B_METADATA_SCREEN_20260918.md)
e dello [stato della roadmap](../../STRATEGIC_10B_20260916/STATUS_20260918.md).
Non trasforma i loro bound teorici in misure.

## Artefatti e provenienza

- Target produttore Q4_K_M: revisione GGUF `97045b260251cfa86f5ad25638fa2dd074153446`, file locale `benchmarks/donor_adaptation/density/results/strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf`, **6.474.702.976 B**, SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` verificato localmente. Ha 26 blocchi, 414 tensori e non include MTP.
- Sorgente MTP: checkpoint BF16 `ai-sage/GigaChat3.1-10B-A1.8B-bf16` revisione `189fff27a1dee68473960c3d5bca53e0e07a3191`; sidecar locale a 210 tensori, **1.614.456.643 B**, SHA-256 `c340ed41c1355441207357723b984fc336082e7c0e8a00deeb6f4fce15c46119`.
- Bundle minimo HF per il convertitore: `strat01_gigachat31_mtp_hf_bundle_189fff27_v1` sotto `benchmarks/donor_adaptation/density/results/`, con sidecar hardlinkato dopo controllo SHA, alias root per embedding/head e `model.norm.weight` distinto. Index a 213 chiavi. I quattro JSON sono vincolati alla revisione e, nella versione attuale del builder, anche ai SHA-256 sorgente.
- `config.json` upstream contiene `routed_scaling_factor: 1` intero; la versione installata di Transformers esige float. Nel bundle v1 il campo è stato normalizzato a `1.0`, stessa grandezza numerica; manifest locale registra SHA-256 sorgente `6a6b8260f08791c4968f70934903e9aa892a53ee3f2e2e05b61a68ea1ff33503` e locale `0c9f747879dcb35bac28ac4e864b849bd9995e0b8552405681878cb17d558dbe`. Il builder ora ripete la normalizzazione e fallisce se gli hash/il campo pin-nati non coincidono. Una verifica live dei quattro JSON pin-nati passa: il config materializzato dal builder ha SHA `0651934efaaf8ce56b593aaeb50d80be84742a9ed40b3cf71c9c1da577af888d`, diverso dal v1 **solo per un LF finale** aggiunto dalla prima modifica locale; il JSON parsato coincide. Il bundle v1 è stato materializzato *prima* di questa modifica al builder: l'export GGUF da un bundle fresco non è stato ripetuto.
- GGUF draft BF16 locale: `benchmarks/donor_adaptation/density/results/strat01_gigachat31_mtp_bf16_189fff27.gguf`, **1.620.709.600 B**, SHA-256 `20d80a73f833e3187adb8adda3e4f1a8d13e0500a796e69025e58473abb0f590`. Contiene 23 tensori: 20 operazioni fisiche del layer `blk.26`, root `token_embd`, `output`, `output_norm`; metadata `deepseek2.block_count=27`, `nextn_predict_layers=1`.

La patch `benchmarks/donor_adaptation/density/strat01_gigachat_llama_mtp.patch`
(SHA-256 `8676b2d32c0f7ada537ddb91c43f4997aa02a789378062bb2354419e8f2b522b`)
è riferita esclusivamente a `llama.cpp` commit
`5b335f413e4f73b0809c4fe39af894efbcc6a0d2`. Abilita export
DeepSeekV3 `--mtp`, Q diretto nel grafo MTP e il vero percorso/parametri del
draft standalone nel loader speculative. Senza la terza correzione la CLI
precedente caricava il target anche quando era indicato un draft separato.
Il checkout esterno temporaneo contiene inoltre un adattamento **solo di
build Windows LLVM-MinGW** (`CreateFileW`, `_WIN32_WINNT=0x0A00`), non incluso
nella patch scientifica. CMake ha riportato `GGML_SYSTEM_ARCH: UNKNOWN` e
`GGML_CPU_GENERIC`: non usare il suo rate come limite della CPU.

## Prove eseguite

Il convertitore patchato ha passato `--dry-run` e l'export BF16 reale; il
reader GGUF indipendente ha letto i 23 nomi e `nextn_predict_layers=1`.
Target e draft coincidono nei metadati numerici comuni salvo il numero di
blocchi (26 contro 27); il draft aggiunge NextN. **25 test offline** delle
quattro suite extractor, contratto GGUF, builder HF e patch sono verdi.

| Tentativo | Esito | Interpretazione |
|---|---|---|
| smoke, contesto 256 | prompt chat da 1.096 token rifiutato; CLI rimasta interattiva, PID mirato arrestato | VOID di setup; nessun decode |
| smoke, contesto 2048, 2 token | generazione `Hello!`; log standard senza prova di uso MTP | verifica parziale del target, non acceptance |
| smoke verbose, contesto 2048, 4 token | draft separato caricato, `draft-mtp` inizializzato e invocato, 1 proposta accettata su 1 | **FIRES**, campione troppo piccolo per stimare acceptance |
| prima richiesta da 64 token | CLI rifiuta argomento `una` per quoting PowerShell; nessun modello caricato | VOID di setup, non dato negativo |
| richiesta corretta da 64 token | 23 proposte accettate su 39 (58,974%), 64 output token, nessun errore runtime | osservazione valida su **un prompt**, non generalizzabile |

Ultimo run: log locale `strat01_mtp_accept64_seed42_20260918_v2.{out,err}.log`
nella directory `benchmarks/donor_adaptation/density/results/`; prompt in
italiano sulla lettura CPU dei pesi, seed 42, `n_predict=64`, `k=1`, default
temperature 0,8, 4 thread, polling dei thread disabilitato, contesto 2048,
CPU-only. La CLI ha aggiunto il chat template del modello: **1.149 token di
prefill**. Il log riporta `draft acceptance = 0.58974 (23 accepted / 39
generated)`, prefill 321.736,52 ms e decode 40.541,17 ms / 64 token, cioè
**1,55 output tok/s**. Questo rate include il draft e il build generic; non
è paired a un controllo no-draft, non è clean-box con il protocollo E63 e non
è una misura di `engine.c`. Una sola sequenza produce 39 proposte correlate:
non dedurre da 23/39 un intervallo binomiale o un rate su workload.

**Stima di traffico, non misura:** se `p=23/39` persistesse, un ciclo
speculativo `k=1` produrrebbe asintoticamente `1+p=1,58974` output token.
Il ledger teorico precedente dà **814.039.040 B** di pesi base W4 e
**256.311.296 grandi pesi MTP attivi**; lasciando il draft BF16 questi
ultimi valgono **512.622.592 B**. Senza riuso, i soli payload sarebbero
`(814.039.040+512.622.592)/(1+23/39) = 834.512.962 B/output token`,
ossia **41,73 GB/s ideali** a 50 output tok/s, contro i 36,30 GB/s del
riferimento locale E30. Quantizzando *anche il draft* a W4, la medesima
aritmetica scenderebbe a **29,63 GB/s ideali**; né l'acceptance dopo PTQ né
il riuso effettivo sono misurati. Il confronto non include scale Q4,
compute, cache/KV, verifica o output head separata e non è un verdetto
di impossibilità; indica quale trade-off misurare dopo.

## Gate successivi, in ordine

1. Verificare che una build x86 nativa e un controllo *paired* senza draft,
   stesso target/prompt/seed/campionamento, isolino il costo reale del draft;
   non trasferire l'1,55 tok/s generico al target di 50. La quantizzazione
   W4 del draft è il ramo successivo da testare, non un guadagno assunto.
2. Ripetere acceptance su prompt preregistrati e corpus non scelto dopo il
   risultato, contabilizzando proposte, accettazioni e token output. Misurare
   logits/hidden-state parity contro reference BF16 prima di attribuire
   eventuali mismatch ai pesi.
3. Solo se qualità e bilancio byte/tempo restano plausibili, progettare il
   port MLA+MoE+MTP in `engine.c`; l'operatore Q diretto e lo speculator di
   `llama.cpp` sono una reference, non il nostro motore.
