# STRAT-01 GigaChat 3.1 — pilot MTP GGUF/CPU (18 settembre 2026)

## Stato e confine della prova

**REFERENCE_PILOT, non gate finale.** Un target pretrained GigaChat 3.1
Q4_K_M e il suo layer MTP BF16 della revisione pin-nata hanno effettuato
speculative decoding nello stesso processo `llama.cpp` CPU. Il draft è stato
caricato da un GGUF *separato*, invocato e ha prodotto token accettati. Non
esiste ancora una prova di qualità del target Q4 rispetto al BF16, di logits
parity MTP rispetto a vLLM, di rate clean-box con build CPU ottimizzata, né un port nel
nostro `benchmarks/phase60/engine.c`. Il requisito ≥50 tok/s **non è passato**.

Una successiva build nativa x86 ha permesso un primo confronto paired,
descritto sotto: il draft BF16 è risultato più lento del controllo e la
sequenza greedy non è identica. Rimane un **pilot n=1**, non una sentenza
statistica o un gate di qualità; la divergenza va spiegata prima di
ottimizzare l'acceptance.

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
| paired x86 nativo, greedy 64 token | baseline 12,78 tok/s; MTP BF16 11,01 tok/s; 25/38 proposte accettate (65,789%) | un solo run per braccio, **-13,85%** di rate decode; gli output divergono, quindi non è parità di qualità |
| ripetizione baseline greedy 64 token | stessi 64 token generati del primo baseline; tempi leggermente diversi | esclude la non-ripetibilità grossolana del solo controllo, non dimostra la causa della divergenza MTP |
| diagnostica paired da 48 token, verbosità default | testi divergenti, ma le righe aggiunte al sampler sono soppresse | VOID diagnostico: `llama-cli` usa verbosity error per default; non ripetere senza `-lv 3` |
| diagnostica paired da 48 token, `-lv 3` | prima differenza all'output token indice zero-based 25, ID 4734 contro 4164 | tracciamento del punto di biforcazione; logging strumentato, tempi non confrontabili |
| MTP diagnostico da 32 token con proposta soppressa a `n_gen=25` | stesso prefisso di 25 ID, poi target sceglie ancora 4164 con batch singolo | il solo batch da due token al passo divergente non è la causa sufficiente; rimane uno stato accumulato diverso |
| MTP diagnostico da 32 token con proposta consentita solo da `n_gen=25` | primi 25 ID e riepiloghi top-2/margine identici al baseline; al passo 25 target sceglie 4164 in batch da due | la diversa forma batch nel passo divergente **è sufficiente** a invertire top-1 in questo controllo; non esclude un effetto accumulato nell'altro controllo |

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

### Addendum: controllo paired nativo e divergenza greedy

Nel checkout temporaneo del medesimo commit `llama.cpp`, CMake sotto
LLVM-MinGW aveva inizialmente inferito `GGML_SYSTEM_ARCH: UNKNOWN` e compilato
`GGML_CPU_GENERIC`. Un adattamento **locale al checkout**, non parte della
patch scientifica, imposta `CMAKE_SYSTEM_PROCESSOR=AMD64` solo se mancante su
Windows; la nuova directory `build-cpu-native` conferma x86 e
`-march=native`. È una build ottimizzata di riferimento, **non** `engine.c`.

Entrambi i bracci hanno usato lo stesso target Q4_K_M, prompt italiano,
chat template da 1.149 token di prefill, seed 42, `--temp 0`, 64 output
token, 4 thread, `--poll 0`, contesto 2048 e CPU-only. MTP aggiunge il GGUF
BF16 separato con `draft-mtp`, `n_max=n_min=1`. Log locali:
`strat01_native_pair_base64_t0_seed42_20260918.{out,err}.log` e
`strat01_native_pair_mtp64_t0_seed42_20260918.{out,err}.log` in
`benchmarks/donor_adaptation/density/results/`.

| Braccio | Prefill | Decode (64 output) | Draft |
|---|---:|---:|---:|
| baseline | 18.739,00 ms / 1.149 | 4.931,37 ms; **12,78 tok/s** | — |
| MTP BF16 | 20.782,21 ms / 1.149 | 5.724,12 ms; **11,01 tok/s** | 25/38 accettate |

Il delta di rate decode è `11,01/12,78 - 1 = -13,85%` circa, senza
interleaving, controllo clean-box o intervallo di confidenza. Il baseline
rieseguito con la medesima invocazione (`...base64_t0_seed42_repeat_20260918`)
ha prodotto la **stessa sequenza di 64 token**; il run MTP diverge già nel
primo paragrafo, a partire da «Per comprendere come la CPU ...» contro
«Per comprendere il processo di generazione ...». Un confronto visivo del
testo, non una diff dei token-ID; le due uscite condividono l'intestazione.

Audit statico del runtime pin-nato: a `temp<=0` il sampler lascia il solo
logit massimo; `common_sampler_sample_and_accept_n` ricampiona ogni proposta
sui logits del target e accetta soltanto l'ID coincidente. Il server valuta
in un batch il token corrente **più** la proposta MTP, contro un solo token
nel baseline. Una variazione numerica del top-1 dovuta alla diversa forma
del batch è una *ipotesi* compatibile; non è ancora dimostrata né esclude un
bug di allineamento posizione/KV/verifica. La prova di ripetibilità del
baseline non discrimina queste cause. La precedenza sperimentale è una
traccia controllata dei logits attorno alla **prima** divergenza, confrontando
contesto, forma batch, KV e indice dei logits; non attribuire all'MTP qualità invariata né
usare il rate come previsione di un modello carved nel motore C.

Una prima diagnostica sul checkout temporaneo ha quindi aggiunto logging
del target top-2 grezzo **prima** della catena di sampling e dell'ID scelto.
I log `strat01_diag_{base,mtp}48_lv3_20260918.{out,err}.log` sono locali;
`-lv 3` è essenziale, perché la CLI altrimenti sopprime INFO. I primi
**25 ID di output** coincidono; all'indice zero-based **25** il baseline
produce **4734**, MTP **4164**. Al medesimo passo, 4734 è il top-1 grezzo
del baseline e 4164 il suo top-2 (margine 0,357017517); nel ramo MTP
4164 è top-1 e 4734 top-2 (margine 0,0479545593). La proposta draft a
quel passo era 4734 e il verificatore ha correttamente rifiutato il draft
perché il target del proprio batch sceglieva 4164. I log mostrano anche
una differenza nei margini del target già alla prima verifica dopo il primo
token: entrambi scelgono ID 50503, ma il margine è 3,49245834 nel
baseline contro 3,35231781 nel batch MTP. Questo è compatibile con
aritmetica dipendente dalla forma batch **prima di qualunque proposta
accettata**, ma non prova che il solo batching spieghi la biforcazione al
passo 25: restano da discriminare differenze numeriche, posizione e stato
KV. I tempi dei run strumentati non sono benchmark di performance.

Controllo di causalità successivo: una modifica **solo nel checkout
temporaneo** fa ritornare `get_n_draft_max()=0` quando `stats.n_gen==25`,
lasciando la speculazione precedente invariata. Il log
`strat01_diag_mtp32_single_at25_20260918.err.log` conferma
`STRAT01_SINGLE_ONLY n_gen=25`; in quel passo il target campiona da
`idx=0` e sceglie **ancora 4164**, con top-2 4734 e margine 0,09333992.
Il baseline a batch singolo aveva scelto 4734. Dunque la forma batch **al
solo passo 25** non basta a spiegare la biforcazione. Lo stato target dopo
le precedenti verifiche differisce già abbastanza da invertire il top-1:
può essere drift numerico accumulato dalle decodifiche multi-token o errore
di gestione posizione/KV/rollback; questo test non li distingue. La
ricostruzione automatica degli ID dal log `STRAT01_VERIFY` del run forzato
non include il passo singolo, quindi per quel passo fa fede la riga
`STRAT01_SAMPLE` e il testo emesso, non un conteggio derivato dalla sequenza
dei soli VERIFY. Anche questo run non è una misura di rate.

Controllo speculare: `get_n_draft_max()` ritorna zero finché
`stats.n_gen<25`, poi consente la proposta dal passo 25. Log locale
`strat01_diag_mtp32_only_at25_20260918.{out,err}.log`. Nei primi 25
output, gli ID **e tutti i riepiloghi raw top-1/top-2/margine** coincidono
esattamente con il baseline strumentato. Il log registra
`STRAT01_ONLY_AT25 n_gen=25`; il draft propone ID 4734, ma il target in
batch da due sceglie **4164** e lo rifiuta. Al passo stesso il margine
raw target fra top-1 4164 e top-2 4734 è **0,254646301**; nel baseline
a batch singolo il top-1 è 4734, top-2 4164, margine **0,357017517**.
Questo isola un effetto **sufficiente della forma batch al passo della
biforcazione** dopo un prefisso che coincide nei riepiloghi osservati.
Insieme al test precedente, dove dopo storia speculativa il batch singolo
sceglieva 4164, indica *anche* un effetto persistente della storia di
decodifica. Non abbiamo confrontato l'intero vettore dei logits, il KV o
gli hidden state: la causa a livello di kernel/stato resta non identificata.
Nessuno di questi run strumentati è un benchmark di rate.

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

**Priorità rispetto al gate finale:** con `k=1`, acceptance osservata
`25/38` e l'ipotesi volutamente favorevole che una verifica target costi
quanto un decode baseline e che il draft non costi nulla, il rate
proiettabile dal singolo baseline nativo sarebbe
`12,78 × (1+25/38) = 21,19 tok/s`; persino `p=1` darebbe
`12,78 × 2 = 25,56 tok/s`. Per arrivare a 50 con `p=25/38` sotto
*quelle stesse ipotesi* occorrerebbe un ciclo target equivalente ad almeno
`50/(1+25/38) = 30,16` iterazioni/s, cioè circa **2,36×** il riferimento
12,78. Sono scenari da **un prompt** e non un bound universale: costo di
verifica batch2, riuso dei pesi, quantizzazione del draft e un nuovo
engine possono cambiare il rapporto. Ma spiegano perché non è razionale
presentare MTP BF16 da solo come via già vicina a 50. Il prossimo
investimento sul ramo richiede una coppia *qualità + kernel target/draft*
plausibilmente entro l'envelope, non un altro sweep della stessa CLI.

La **parità greedy esatta non è il gate di qualità** della roadmap: il
verificatore seleziona il top-1 del target *nel batch effettivamente usato*,
che qui non è numericamente identico al target batch-1. Pertanto non
chiamare la speculazione lossless rispetto al baseline batch-1; d'altra
parte, la divergenza di una sequenza non prova un deficit di capacità.
L'eventuale candidatura va giudicata con BPB, task/rollout e rate sullo
**stesso artefatto e percorso di esecuzione**, rispetto al teacher fissato.

## Gate successivi, in ordine

1. Per attribuire il meccanismo della divergenza greedy al token 26, sia il
   batch a due token isolato sia la storia speculativa precedente con batch
   singolo al passo 25 possono invertire il top-1. Confrontare logits
   completi/hidden state e KV lungo il prefisso, posizioni e rollback,
   contro un target che usa lo stesso prefisso in batch singoli. Questa
   attribuzione è diagnostica, non sostituisce il gate BPB/task. Non
   trasferire né l'1,55 tok/s generic né il 11,01 tok/s native al gate 50.
2. Se il ramo supera prima un budget CPU plausibile, ripetere acceptance
   su prompt preregistrati e corpus non scelto dopo il risultato,
   contabilizzando proposte, accettazioni e token output. Misurare
   logits/hidden-state parity contro reference BF16 per attribuire i
   mismatch e BPB/task sul percorso effettivo per giudicare la qualità.
   La quantizzazione W4 del draft richiede un proprio gate, non è un
   guadagno assunto; il paired di rate va interleaved e clean-box.
3. Solo se qualità e bilancio byte/tempo restano plausibili, progettare il
   port MLA+MoE+MTP in `engine.c`; l'operatore Q diretto e lo speculator di
   `llama.cpp` sono una reference, non il nostro motore.
