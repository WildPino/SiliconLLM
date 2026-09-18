# STRAT-01 — GigaChat 3.1 Lightning 10B-A1.8B: metadata screen

**18 settembre 2026. Solo fonti primarie e metadati.** Nessun peso scaricato, kernel eseguito, task locale, T4, port C o throughput CPU. Decisione: donor licenziato e valutato dal produttore, **ma non promosso a port/benchmark pesante**: la geometria attiva e il meccanismo MTP richiedono una prova distinta di fattibilità batch-1 sulla CPU target.

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

Questa somma esclude norm, bias, embedding di input indicizzato, KV/state, scale di quantizzazione e **l'intero costo del layer MTP 26**. È contabilità di forma dell'operatore, non un trace di byte letti. Il pure-payload W4 del base così definito è **814.039.040 B/token**, pari a **40,70 GB/s a 50 tok/s**; sul budget di progetto di 14 ms richiede **58,15 GB/s** solo per queste matrici. Anche questa stima più stretta supera il vecchio riferimento E30 di 36,30 GB/s, che tuttavia non è un bound universale per ogni kernel, riuso o layout.

Usando invece il numero **nominale** di 1,8B attivi della card, W4 sarebbe 0,90 GB/token e 45 GB/s a 50 tok/s; W2 sarebbe 0,45 GB/token e 22,5 GB/s. Questi valori sono uno scenario separato, non da sommare al ledger base. Non trattare W2 come qualità valida: la PTQ W2 testata sullo StdMoE ha fallito, ma non decide la risposta di questo checkpoint diverso.

L'MTP **non regala due token a costo di uno**: il draft, la sua head aggiuntiva e la verifica hanno un costo, e contano solo i token effettivamente accettati senza cambiare la distribuzione target. La card riporta su **1×H100 SXM5, vLLM, concorrenza 32** BF16 2.866 output tok/s senza MTP e 3.346 con MTP, +16,7% in quel setting. Questo non è un test CPU batch-1, né una previsione di acceptance/throughput sulla nostra architettura. Se applicato ingenuamente al solo base-W4 di forma, un fattore 1,167 non basterebbe a trasformare 40,70 GB/s in 36,30 GB/s, **anche ignorando il costo del layer MTP**; questa aritmetica non adjudica MTP su CPU perché costo/riuso/accettazione possono cambiare.

## Disposizione sperimentale

Il modello supera il problema di licenza dichiarata che blocca il lead Nous ed è plausibilmente addestrato/post-addestrato, ma è **più attivo** e usa operatori C nuovi (MLA, MoE, MTP). La card non dimostra un artefatto quantizzato che passi qualità né 50 tok/s CPU.

Prima di acquisire ~23 GB BF16 o portare operatori: validare il ledger base contro i tensor shape effettivi dell'index, censire il **layer MTP 26** e il suo costo incrementale, definire il protocollo di **token accettati per forward** a batch-1 e verificare se esiste un formato già pubblicato di **quello stesso checkpoint** con qualità difendibile. Un gate di traffico/qualità favorevole può autorizzare un pilot, non un trasferimento dei benchmark H100. Se il floor resta oltre budget senza un meccanismo misurato di riuso o accettazione, non iniziare un port completo.
