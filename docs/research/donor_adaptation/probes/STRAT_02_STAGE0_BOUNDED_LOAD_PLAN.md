# STRAT-02 Stage 0 — piano di acquisizione e loader F32 a memoria limitata

**Data:** 2026-09-16. **Stato:** piano/preflight, non acquisizione di pesi.
La checklist del [brief](../briefs/BRIEF_STRAT_02_STDMOE_PRECISION_GATE.md)
resta aperta; questo documento non autorizza da solo il download o una
valutazione del checkpoint. [Formato](STRAT_02_STAGE0_WEIGHT_FORMAT.md),
[scoring](STRAT_02_STAGE0_TOKEN_SCORING.md).

## Provenienza e cap

La revisione fissata è `d2a4949c9d4ad6cf47fbac131f7e020077332b21` di
`allenai/StdMoE_1b14b_1T_Preanneal`. L'indice safetensors di 6.259 chiavi,
13.568.641.024 parametri F32 e 54.274.564.096 byte tensoriali è nella cache
locale; SHA-256
`934e6eb0b66c52b4f4a5d7ef7fde489f41d8fa697ea8ff52862c959377412339`.
La [lista write-once](../../../../benchmarks/donor_adaptation/density/strat02_weight_sources.json)
registra per ciascuno degli 11 shard dimensione e SHA-256 LFS dichiarati
dall'API del publisher alla stessa revisione. Somma file 54.275.336.216
byte (50,548 GiB); il contenitore aggiunge 772.120 byte ai tensori.
L'indice e gli asset tokenizer **non** sono shard peso.
Una query `HfApi.model_info` sulla **stessa revisione** ha restituito SHA
identico, `license=apache-2.0`, `private=false`, `gated=false`; la
[model card del publisher](https://huggingface.co/allenai/StdMoE_1b14b_1T_Preanneal)
conferma la licenza dichiarata e lo scopo di ricerca. È il checkpoint
distribuito, non una verifica dei diritti dei testi di valutazione PG-19.

Prima dell'acquisizione, `strat02_acquire_weights.py --preflight` riverifica offline
RAM/disco/libero e tutti gli hash di config, codice e indice. Consentire
solo questi 11 nomi, revision esatta, un download alla volta, nessun
`from_pretrained` implicito, file temporanei nella stessa cache e ripresa
idempotente. Il cap dei **payload shard validi nuovi** è esattamente la
somma degli 11 file (54.275.336.216 byte); non è un limite garantito sui
byte di rete, che possono includere header, retry o segmenti parziali. Il
downloader non aggiunge retry applicativi silenziosi e registra ciò che
riesce a osservare. Il cap di spazio libero richiesto è **almeno 65 GiB** prima del
download (file finali più lo shard parziale più margine) e il download si
ferma a errore di hash/size. Dimensioni finali, stato `cached`/`downloaded`
e hash letti localmente andranno nel run manifest; se i byte on-wire non sono
strumentati, vanno marcati `unknown`, non sostituiti con la dimensione dei
file. Non si cancella automaticamente nulla.
Il 16 settembre la lettura locale era 59,28 GiB RAM fisica disponibile su
79,95 GiB e 856,89 GiB liberi su D:. Il preflight del loader ha poi
misurato **7.283.021.737.984 byte liberi sulla cache HF in E:** e
63.191.719.936 byte di RAM disponibile: sono snapshot diversi e da
rifare al run. Il cap di download riguarda la cache E:, mentre gli
artefatti di output futuri potranno stare su D: con cap separato.

## Perché `from_pretrained` non è il loader di riferimento

I tensori F32 occupano **50,547 GiB**. Con 59,28 GiB liberi al preflight,
una sola altra copia completa sarebbe ~101,1 GiB solo di pesi, impossibile
sulla RAM disponibile. Per il teacher si istanzia `EmoForCausalLM` su device
`meta` con Transformers 4.57.1, codice remoto già pin-nato e
`USE_HUB_KERNELS=NO` impostato **prima** dell'import. Gli 11 safetensors
sono aperti `safe_open(..., framework='pt', device='cpu')` e restano vivi
tramite un `ExitStack`; `get_tensor` espone i tensori mappati. Con index
verificato e keyset esatto, `load_state_dict(..., assign=True, strict=True)`
assegna quegli storage al modello `meta` senza `.clone()`, `.to('cpu')` o
materializzazione preventiva. Si asseriscono 6.259 tensori, F32, parametro
totale, assenza di parametri `meta` e identità `data_ptr` fra tensori mappati
e parametri assegnati. Un mini-fixture locale `nn.Linear`+safetensors ha già
mostrato `same_storage=True`, `meta_remaining=False` e forward esatto; **non
dimostra** che l'intero checkpoint da 50,5 GiB entri nel peak RAM previsto.
Il codice Emo ha inoltre un buffer RoPE `inv_freq` **non persistente** che
resta `meta` dopo `assign=True`: il loader ricostruisce solo il modulo rotary
senza pesi tramite la stessa classe/config pin-nata su CPU, asserendo che
nessun altro buffer `meta` esista. Il test con Emo casuale minuscolo
riproduce il fallimento originale e ottiene parità esatta dell'hidden dopo
il ripristino; non è ancora una prova full-size.
Un secondo controllo senza pesi ha caricato il codice del donor con
Transformers 4.57.1, costruito il modello `meta` tramite il loader offline e
confrontato il suo `state_dict` con l'indice: **6259/6259 chiavi identiche**,
13.568.641.024 parametri. Non è un test di forward né di mapping full-size.

La lettura dell'intero file per verificare SHA-256 prima dell'uso è una
passata I/O obbligatoria, non una copia persistente F32. Il modello teacher
viene valutato da solo: mai due `EmoForCausalLM` F32 contemporaneamente.
Prima del primo documento si registra RAM disponibile, committed/working-set
del processo e page faults; si prova **un documento calibrazione** come
smoke di apparato e si ferma ai cap di memoria/tempo fissati sotto, OOM o identità
storage falsa. Nessuna osservazione su quel documento diventa gate di
qualità. Il forward usa `torch.inference_mode()`, batch 1, `use_cache=False`,
`output_hidden_states=False`, `output_router_logits=False`, e input EOS più
payload meno l'ultimo token. I logits completi del massimo span heldout
richiederebbero circa 1,13 GiB F32 (`3025×100352×4`), prima di temporanei
per logsumexp e attention. Il [core di scoring](../../../../benchmarks/donor_adaptation/density/strat02_score.py)
ora proietta l'hidden state in chunk da 128 posizioni (al massimo circa
51,4 MB F32 di logits/chunk) e accumula NLL in double; non conserva i logits
di tutto il documento. Non elimina il costo della head né degli hidden
state, e il picco RAM reale resta da misurare.

**Cap operativi fissati prima degli shard:** all'avvio richiedere almeno
55 GiB di memoria fisica *disponibile* e almeno 65 GiB di spazio libero
sulla cache target; durante il primo documento calibrazione leggere memoria
di sistema e processo almeno ogni 5 s. Interrompere con `VOID_RESOURCE` se
la memoria disponibile scende sotto 8 GiB, se working set o private commit
del processo supera 70 GiB, se appare OOM, o se il singolo smoke supera
60 minuti. Sono limiti di sicurezza/tempo, non soglie di qualità né stime di
tok/s. Registrare page faults e loro trend, ma non usare un contatore di
fault generici come soglia (include fault soft e dipende dall'OS). Il
monitor deve avere exit non distruttiva; non terminare altri processi per
far passare un run. Se il cap è violato, non ritentare con un loader diverso
senza nuovo piano preregistrato.

Il primo smoke usa esattamente la riga 0 di `calib.jsonl` (ID
`file:data/external/the_stack_python/cpython/Lib/test/test_sqlite3/test_userfunctions.py`,
`item_sha256=17020c6b2147d0d33555bb96dbde6302b9e4859523e8d0bd1a9d73bb0177d11e`).
Il [supervisore](../../../../benchmarks/donor_adaptation/density/strat02_bounded_smoke.py)
avvia un solo worker, campiona ogni 5 s e può terminare **solo quel worker**.
Su Windows deve lanciare direttamente l'interprete base, con i package
pin-nati nel `PYTHONPATH`: la `.venv` è un redirector il cui PID non misura
il worker vero. La parità degli import/versioni e dell'eseguibile campionato
è parte dei gate di preflight/monitoraggio, non una scelta di prestazione.
Il risultato della riga di calibrazione serve esclusivamente a verificare
l'apparato; non entra nel gate di qualità. L'acquisizione degli shard è
registrata [separatamente](STRAT_02_STAGE0_ACQUISITION.md).

Runtime osservato nel controllo senza pesi: Python 3.12.10,
Torch 2.6.0+cu124, Transformers 4.57.1, tokenizers 0.22.2,
NumPy 2.5.3, safetensors 0.8.0, huggingface_hub 0.36.2;
`USE_HUB_KERNELS=NO` prima degli import e nessun autocast. Il target
Transformers isolato vive in una directory temporanea locale: va
ricostruito e controllato per versioni/hash all'atto del run, non trattato
come ambiente persistente. La `.venv` ordinaria con Transformers 5.13.1
rimane esclusa.

## Candidato e comparabilità

Il codec W4/W2 produce codici e scale da tensori F32 a gruppi di 128 in
streaming per riga/tensore. Per lo scoring statistico si può materializzare
un **riferimento F32 ricostruito dal packed** in nuovi safetensors su disco,
poi usare lo stesso loader mappato, dopo aver rilasciato completamente il
teacher. La ricostruzione deve decodificare *solo* il packed congelato; non
può rileggere gli originali per correggere l'errore. Questo file F32 è un
oracle di qualità dell'operatore quantizzato, **non** il deployment artifact
né una prova di RAM/token/s in C. La verifica di parità è
`packed -> scalar decode -> reconstructed F32 -> model output` su dati
piantati; il futuro kernel C dovrà eguagliare il medesimo operatore.

Lo spazio disco per teacher F32, W4 ricostruito e mixed ricostruito può
superare **150 GiB** oltre ai due packed e ai file temporanei; il cap va
ricontato prima di generare ogni arm, non giustificato con il solo spazio
per il primo download. Tutti gli output sono write-once con hash e nessuna
pulizia distruttiva automatica. Se il loader non tiene il vincolo RAM, la
variante GPU/T4 o lo streaming per layer richiede un nuovo piano e va
comunicata all'utente prima di impegnare T4×2; non si cambia silenziosamente
precisione teacher a FP16 per farlo entrare.
