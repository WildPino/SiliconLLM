# STRAT-01 — GigaChat 3.1 Lightning 10B-A1.8B: metadata screen

**18 settembre 2026. Solo fonti primarie e metadati/intestazioni.** Nessun file di pesi completo scaricato né modello caricato; nessun kernel, task locale, T4, port C o throughput CPU. Decisione: donor licenziato e valutato dal produttore, **ma non promosso a port/benchmark pesante**: geometria attiva e meccanismo MTP richiedono una prova distinta di fattibilità batch-1 sulla CPU target.

## Identità e qualità dichiarata

- [Checkpoint BF16](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B-bf16), revisione pin-nata [`189fff27a1dee68473960c3d5bca53e0e07a3191`](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B-bf16/tree/189fff27a1dee68473960c3d5bca53e0e07a3191).
- [API metadati](https://huggingface.co/api/models/ai-sage/GigaChat3.1-10B-A1.8B-bf16/revision/189fff27a1dee68473960c3d5bca53e0e07a3191), [config pin-nata](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B-bf16/blob/189fff27a1dee68473960c3d5bca53e0e07a3191/config.json), [card pin-nata](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B-bf16/blob/189fff27a1dee68473960c3d5bca53e0e07a3191/README.md).
- La card dichiara **MIT**; l'API dà `cardData.license=mit`. L'API safetensors riporta **11.479.750.784 elementi BF16** (~22,96 GB payload); quindi il nome “10B” è arrotondato e non va usato per contabilità precisa. La [base dichiarata](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-base) ha anch'essa una card MIT, ma è un checkpoint distinto e non sostituisce la 3.1 instruct.
- La card pubblica valutazioni, tra cui HumanEval Plus 0,7317 e MMLU EN 0,7298 per 3.1 Lightning. **Sono risultati dichiarati dal produttore**, senza qui un rerun locale o un binding esplicito dei log di eval allo SHA sopra. Non sono BPB, task o generazione nel nostro protocollo.

## Forma e soglia di traffico

La config BF16 dichiara `DeepseekV3ForCausalLM`, hidden 1536, 26 layer base, 64 routed expert con top-4, un shared expert, `moe_intermediate_size=1280`, primo layer denso, MLA, head untied con vocab 128.256, e un layer MTP (`num_nextn_predict_layers=1`). L'[implementazione Transformers v4.57.3](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py) fissa le dimensioni delle proiezioni MLA, MoE e MLP usate qui sotto. L'index pin-nato conta 5.323 tensor key, layer `0..26`: il layer `26` contiene una projection `shared_head.head` e un `embed_tokens` propri ed è contabilizzato **separatamente**, non come un layer base gratuito. La card descrive circa **1,8B parametri attivi/token**, ma non specifica qui una definizione tensor-per-tensor per head/MTP.

| Organo base per un token autoregressivo | Formula da config + operatori | Pesi grandi attivi |
|---|---|---:|
| MLA Q/KV/O, 26 layer | `26×[1536×(32×192) + 1536×(512+64) + 512×32×(128+192) + (32×192)×1536]` | 650.051.584 |
| Expert routed, layer 1–25 | `25×4×3×1536×1280` | 589.824.000 |
| Expert shared, layer 1–25 | `25×3×1536×1280` | 147.456.000 |
| MLP dense, layer 0 | `3×1536×8960` | 41.287.680 |
| Router, layer 1–25 | `25×1536×64` | 2.457.600 |
| Head principale untied | `1536×128256` | 197.001.216 |
| **Totale grandi organi base** | | **1.628.078.080** |

Il ledger base è stato **riconciliato con le shape di tutte le 5.323 chiavi** leggendo esclusivamente le intestazioni safetensors dei sei shard (`model-00000`..`model-00005`), mediante richieste HTTP `Range` su una revisione immutabile: header di 103.400, 125.856, 126.672, 126.872, 126.872 e 50.408 byte. La somma delle shape è esattamente **11.479.750.784 elementi**, uguale all'API e all'index. Per i 25 layer MoE sono stati conteggiati esattamente quattro expert identici per forma per token, non tutti i 64. Il primo calcolo meccanico includeva erroneamente le matrici `shared_experts` anche in `dense`; il classificatore è stato corretto e la somma finale coincide con la formula indipendente in tabella. Non è stata fatta inferenza sui valori dei pesi o sul routing concreto.

La somma base esclude norm, bias, embedding di input indicizzato, KV/state, scale di quantizzazione e **l'intero costo del layer MTP 26**. È contabilità di forma dell'operatore, non un trace di byte letti. Il pure-payload W4 del base così definito è **814.039.040 B/token**, pari a **40,70 GB/s a 50 tok/s**; sul budget di progetto di 14 ms richiede **58,15 GB/s** solo per queste matrici. Anche questa stima più stretta supera il vecchio riferimento E30 di 36,30 GB/s, che tuttavia non è un bound universale per ogni kernel, riuso o layout.

## Layer MTP e soglia di acceptance (solo modello ideale)

Le shape effettive del layer `model.layers.26` comprendono una seconda head completa `shared_head.head.weight [128256,1536]` e una seconda tabella `embed_tokens.weight [128256,1536]` (quest'ultima indicizzata, non letta per intero a ogni draft). Il costo di **grandi matrici attive per un draft MTP** ricostruito dalle shape è:

| Organo MTP | Elementi |
|---|---:|
| MLA Q/KV/O | 25.001.984 |
| Quattro expert routed | 23.592.960 |
| Shared expert | 5.898.240 |
| Router | 98.304 |
| `eh_proj` | 4.718.592 |
| Head MTP separata | 197.001.216 |
| **Totale grandi organi MTP** | **256.311.296** |

A un ipotetico W4 uniforme sono **128.155.648 B/draft**; base+MTP leggerebbero **942.194.688 B/ciclo** se ogni matrice grande fosse caricata una sola volta. Non è una misura e non include il costo di verificare due posizioni, scale, cache o accessi irregolari. Per una proposta top-1 a ciclo, se `p` è la probabilità di accettazione e il ciclo produce idealmente `1+p` token, il riferimento E30 di 36,30 GB/s richiederebbe **`p≥0,298`** solo per collocare 50 token accettati/s dentro i 20 ms completi, o **`p≥0,854`** per rispettare il budget progettuale di 14 ms/accepted-token riservato allo streaming. Queste sono **condizioni necessarie sotto assunzioni favorevoli**, non sufficienti né previsioni di acceptance. Senza MTP, i soli 814,04 MB base richiederebbero almeno 22,43 ms al medesimo riferimento di banda, prima di compute. Il rapporto H100 pubblicato non misura `p` batch-1 sulla nostra CPU.

Usando invece il numero **nominale** di 1,8B attivi della card, W4 sarebbe 0,90 GB/token e 45 GB/s a 50 tok/s; W2 sarebbe 0,45 GB/token e 22,5 GB/s. Questi valori sono uno scenario separato, non da sommare al ledger base. Non trattare W2 come qualità valida: la PTQ W2 testata sullo StdMoE ha fallito, ma non decide la risposta di questo checkpoint diverso.

L'MTP **non regala due token a costo di uno**: il draft, la sua head aggiuntiva e la verifica hanno un costo, e contano solo i token effettivamente accettati senza cambiare la distribuzione target. La card riporta su **1×H100 SXM5, vLLM, concorrenza 32** BF16 2.866 output tok/s senza MTP e 3.346 con MTP, +16,7% in quel setting. Questo non è un test CPU batch-1, né una previsione di acceptance/throughput sulla nostra architettura. Se applicato ingenuamente al solo base-W4 di forma, un fattore 1,167 non basterebbe a trasformare 40,70 GB/s in 36,30 GB/s, **anche ignorando il costo del layer MTP**; questa aritmetica non adjudica MTP su CPU perché costo/riuso/accettazione possono cambiare.

## Il GGUF pubblicato non contiene MTP

Il repository [GGUF del produttore](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF), revisione [`97045b260251cfa86f5ad25638fa2dd074153446`](https://huggingface.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF/tree/97045b260251cfa86f5ad25638fa2dd074153446), pubblica BF16, Q4_K_M, Q6_K e Q8_0. L'[API tree](https://huggingface.co/api/models/ai-sage/GigaChat3.1-10B-A1.8B-GGUF/tree/97045b260251cfa86f5ad25638fa2dd074153446?recursive=true) riporta Q4_K_M **6.474.702.976 byte** (blob SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`) e BF16 GGUF **21.356.281.984 byte** (blob `e7a6409be0ac197babf21c48cfc8a96486d035c1e7a784feadd817b7b883c08e`). Dimensione del file **non** equivale a byte letti per token.

Ho analizzato via HTTP `Range` le intestazioni GGUF v3: BF16 e Q4_K_M hanno entrambi 47 metadata key, **414 tensor**, `deepseek2.block_count=26`, `general.architecture=deepseek2`, nessun nome `blk.26`, `mtp` o `draft`. Le intestazioni si chiudono al byte 6.102.892 in entrambi i file. Per semplicità il range richiesto era `0..8.388.607`: ha quindi trasferito anche **2.285.716 byte iniziali di payload per file** oltre l'header, non decodificati né usati. Il Q4_K_M è dunque un **artefatto base-only**: non può fornire il draft MTP del checkpoint safetensors, anche se la card generale menziona MTP. Non usare un test GGUF Q4 come prova della via speculativa. La fedeltà BPB di questo Q4 allo stesso BF16 non è pubblicata qui, e il binding di conversione allo SHA BF16 non è stabilito dalle sole intestazioni.

## Disposizione sperimentale

Il modello supera il problema di licenza dichiarata che blocca il lead Nous ed è plausibilmente addestrato/post-addestrato, ma è **più attivo** e usa operatori C nuovi (MLA, MoE, MTP). La card non dimostra un artefatto quantizzato che passi qualità né 50 tok/s CPU.

Il ledger base e MTP e il censimento del GGUF sono chiusi **a livello di forma**. Prima di acquisire ~23 GB BF16 o portare operatori restano due gate: (1) parità/qualità misurabile di un formato che **preservi MTP** oppure un nuovo export esplicito da questo SHA, senza considerare il Q4 base-only equivalente; (2) protocollo batch-1 di **token accettati per ciclo** e di riuso dei pesi verificati, non solo il rate di H100. Un gate favorevole può autorizzare un pilot, non il trasferimento dei benchmark H100. Se il traffico effettivo resta oltre budget senza riuso o acceptance misurati, non iniziare un port completo.
