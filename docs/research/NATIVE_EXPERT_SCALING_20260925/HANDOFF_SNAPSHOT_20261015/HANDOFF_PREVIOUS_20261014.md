# SiliconLLM — HANDOFF operativo e scientifico

**Aggiornato il 14 ottobre 2026, Europe/Rome. Repository: `D:\_THINGS\Progetti\SiliconLLM`. Branch: `research/native-expert-scaling`. HEAD osservato prima di questo handoff: `bd88f5e`.**

Questo documento trasferisce il contesto della ricerca, il significato delle evidenze e il punto preciso di ripresa. Il goal scientifico è **INCOMPLETO**. Lo stato restituito dallo strumento del goal durante questa consegna è **`paused`**. I documenti storici che riportano `ACTIVE` descrivono lo stato precedente e non devono essere usati per inferire lo stato corrente di un processo.

Il precedente `HANDOFF.md`, 83.691 byte e ultima modifica filesystem 30 luglio 2026, è conservato integralmente in `docs/research/NATIVE_EXPERT_SCALING_20260925/HANDOFF_SNAPSHOT_20261014/HANDOFF_PREVIOUS_20260730.md`. Il suo incipit risaliva al 3 luglio. Contiene il patrimonio delle fasi originali; le sue istruzioni di “next” e le sue dichiarazioni di completamento non sostituiscono quelle attuali.

## Indice

1. Punto esatto di ripresa e audit interrotto.
2. Obiettivo del proprietario e criteri di completamento.
3. Istruzioni persistenti, risorse, Git e processi.
4. Mappa delle fonti e come interpretare una prova.
5. Motore originale: patrimonio reale e limiti del trasferimento.
6. Donor, architettura, corpus e precisione effettiva.
7. Storia della ricerca prima del percorso Falcon completo.
8. Geometria del readout, informazione e incompatibilità degli operatori.
9. Dalla ricorrenza locale agli input veri e al chatbot completo.
10. Componenti C, ABI, compilatore iniziale e readout.
11. Limiti dei generatori contestuali e delle banche di output.
12. Operandi FFN reali, pilota affine e difetto nonlineare esatto.
13. Sottoinsiemi fissi, oracle per token, PCA e metrica source-active.
14. Custodia, audit falliti, aggiornamento NVIDIA e stato dei risultati.
15. Piano concreto verso il goal, scelte condizionali e idee ancora ipotetiche.
16. Inventario, comandi di ripresa, verifiche e trappole da evitare.

## 1. Leggere questo prima di eseguire qualsiasi cosa

### 1.1 Stato reale dell'ultima attività

L'ultimo esperimento è il selettore di neuroni FFN con geometria d'ingresso derivata dalle **derivate del remainder del donor**, bilanciata con il secondo momento FIT. Il producer è terminato correttamente. Il successivo audit completo è stato avviato, ma dopo l'interruzione della conversazione non risultano né il processo corrispondente né una ricevuta finale né il risultato dell'audit.

| Oggetto | Stato al controllo del 14 ottobre, circa 09:19–09:22 |
| --- | --- |
| Producer `ffn_active_router.py` | Terminato, ricevuta `exit_code=0` |
| Risultato producer | Presente, 412.240 byte; SHA256 `aef9eccc6b4fa2f2e66820e0fb4288000eb4f368b5e4f23b2bbdb04d61015189` |
| Producer PID / creazione | `6508` / `1791960298.124822`; non è un worker da rilanciare |
| Durata / picco OS producer | 595,687 s / 1.933.545.472 byte |
| Audit lanciato | `ffn_active_router_audit.py`, PID `24844`, avvio 08:59:42 locale, sessione terminale `23212` |
| Ultimo avanzamento osservato dell'audit | Site 12, 384 delle 576 predizioni controllate, 395,953 s; prima site 0, 192 predizioni |
| Audit dopo l'interruzione | PID assente dal controllo WMI; sessione `23212` restituisce `Unknown process id` |
| JSON di adjudication / terminale audit | Entrambi assenti al controllo |
| Closure finale dell'esperimento | Non eseguita e non qualificata |
| Qualificazione scientifica attuale | **Risultato producer preliminare; audit NON CONCLUSO** |

Non è corretto attribuire un `exit_code=1` all'audit: manca la ricevuta del worker. Non è corretto chiamarlo PASS: 384 controlli non sostituiscono 576 controlli e la chiusura. Il fatto osservato è l'assenza del processo e delle ricevute dopo l'interruzione; non è stata ricostruita una causa di terminazione a livello OS.

I file chiave sono sotto `docs/research/NATIVE_EXPERT_SCALING_20260925/`:

- `FFN_ACTIVE_ROUTER_PROTOCOL_20261014.md`;
- `FFN_ACTIVE_ROUTER_STATE_20261014.md` — lo stato di lancio originario è conservato sotto l'aggiornamento di handoff;
- `ffn_active_router_binding_20261014.json`;
- `ffn_active_router_binding_20261014.preparation_terminal.json`;
- `ffn_active_router_result_20261014.json`;
- `ffn_active_router_result_20261014.terminal.json`.

Il codice è in `benchmarks/native_expert_scaling/ffn_active_geometry.py`, `ffn_active_router.py`, `ffn_active_router_audit.py`, `ffn_active_router_close.py`. Freeze del codice `b34cd94`; binding commit `58c33ec`; ultimo stato commit `bd88f5e`. Binding 2.325.345 byte, SHA256 `86a6df9cb9bbf93feb1f53c25380eead0355f07a1012c13c754e02529d53ba54`.

### 1.2 Prima azione alla ripresa della ricerca

1. Verificare nuovamente file, hash dichiarati, processi e stato goal. Questo handoff è una fotografia, non un lock.
2. Conservare l'osservazione dell'interruzione come osservazione, senza inventare un terminale del worker.
3. Se adjudication e ricevuta non sono comparsi e nessun worker corrispondente è vivo, avviare **un solo audit completo**, sul producer già congelato, senza rifare il fitting o le predizioni.
4. Se l'audit termina PASS, eseguire `ffn_active_router_close.py`, previa verifica che non esista già una closure. Se fallisce, conservare il primo difetto e diagnosticarlo prima di qualunque riparazione.
5. Scrivere il report finale dell'esperimento e aggiornare `INDEX.md`, `METHOD.md`, `PRIOR_EVIDENCE.md` e lo stato. Il risultato non ammette comunque un candidato ai gate locali scelti su FIT: vedi sezione 13.
6. Prima di ulteriori griglie di rank/celle, implementare e congelare il **cost screen dei 24 layer ricorrenti reali** proposto in `NATIVE_24_LAYER_COST_SCREEN_NEXT_20261014.md`.

Durante la produzione di questo handoff non vengono avviati nuovi esperimenti numerici. La consegna richiesta è il contesto per il nuovo modello, non una nuova campagna.

## 2. Obiettivo: una pipeline pretrained CHATBOT → engine C

La fonte autoritativa è `C:\Users\giosa\.codex\attachments\6359009a-bac9-4073-a714-3fb7f7537b80\goal-objective.md`. Il percorso originariamente menzionato con attachment `3d644d97-f09b-47e7-ae36-100b38c3a29d` non era più disponibile. Una copia completa dell'obiettivo è nello snapshot associato a questo handoff.

Si cerca un metodo riproducibile che prenda **un LLM già addestrato come chatbot**, ne trasferisca conoscenza e comportamento in una geometria efficiente e lo esegua nel motore C. L'architettura desiderata separa la capacità totale dal costo attivo: nucleo SSM/SWA compatto e riutilizzabile, molte funzioni o esperti utili in RAM, consultazione selettiva, precisioni differenziate e LUT/ternari dove convenienti.

Il proprietario ha ribadito che il numero `n` può diventare molto grande, idealmente limitato dalla RAM. Un donor da 100B dovrebbe poter ospitare molta più capacità condizionale di uno da 10B; il routing deve reggere una scelta molto più ampia senza far crollare qualità o velocità. Il rapporto “10 volte n” è un'intenzione di scaling da verificare, non un teorema o un numero già misurato.

### 2.1 Cosa conta come completamento

| Requisito | Evidenza attuale | Cosa manca |
| --- | --- | --- |
| Conversione riproducibile da pretrained | Provenienza, catture, compilatori, pacchetti C e bridge Falcon qualificati | Pipeline target compatta e completa, con scelte iniziali documentate |
| Qualità chatbot rispetto al donor | Bridge ricorrente full-width: 14/16 task e buoni KL/disagreement sul corpus noto | Qualità della pipeline CPU target, dati non consumati, generazione propria e contesti dichiarati |
| ≥50 token utili accettati/s batch 1 | Vecchie misure del motore e componenti; una misura Switch originale storica >50 | Velocità end-to-end del **medesimo artefatto** che supera i gate di qualità |
| Target migliorativo 100 token/s | Intenzione | Misura sullo stesso artefatto, budget totale 10 ms/token |
| Grande n utile | Alcuni esperimenti storici di utilità, routing sintetico e costi finiti | Capacità distinta e causalmente utile al crescere di n, con costo attivo contenuto |
| CPU LUT corretta | Kernel e readout interi qualificati | Routing selettivo realmente eseguito in C: IDs **e masse**, precisione, qualità e costo |
| Traffico DRAM reale | Ledger di operandi, misure warm/evicted e costi componenti | Contatori o altra misura del traffico effettivo del modello finale |
| Generalità 10B/100B/famiglie | Evidenze specifiche su piccoli donor e metadati di altri | Valori reali, conversione e qualità di altre famiglie/scale; non basta leggere gli header |
| Costo di conversione sostenibile | Compressioni lossless e strumenti locali | Master, gradienti, optimizer e costi che crescono O(n), dati, GPU-ore e storage end-to-end |

Una implementazione generica del donor che conserva tutto il costo denso non soddisfa la tesi architetturale. Un piccolo student rapido che perde il chatbot non soddisfa il goal. Duplicare parametri senza nuova utilità non prova capacità trasferita. Token/s grezzi di un modello che fallisce tutti i task non sono token/s utili.

### 2.2 Le due domande da mantenere collegate

1. Quale geometria target conserva informazione e capacità utile con poco lavoro attivo?
2. Quali trasformazioni, approssimazioni o adattamenti permettono di trasferirvi il donor?

Il bridge full-width risponde parzialmente alla seconda domanda perché mostra che una ricorrenza nativa derivata dal donor può comporsi in un chatbot ancora utile. I router/sketch FFN rispondono a una parte della prima. Nessuno dei due, isolatamente, è il risultato finale.

## 3. Istruzioni del proprietario che persistono

### 3.1 Metodo e autonomia

- Scegliere autonomamente l'esperimento motivato dalle evidenze, e rivalutare periodicamente l'intero progetto.
- Ridurre i difetti a problemi espliciti di algebra, geometria e informazione; distinguere identità esatte, approssimazioni e misure.
- Non spostare le soglie dopo aver visto FIT o DEV. Non scegliere il candidato usando DEV se il protocollo sceglie su FIT.
- Il messaggio che chiedeva **Fail-Fast è stato esplicitamente revocato**. Non saltare gli audit perché la qualità è catastrofica: possono emergere fallimenti meccanici silenziosi e intuizioni importanti.
- Completare la verifica prevista, conservando fallimenti e riparazioni. Ciò non autorizza una ripetizione infinita di campagne senza una nuova variabile.
- `graphify` è **opt-in**: nessun aggiornamento ordinario del grafo. Il proprietario ha chiesto di eliminarne l'obbligo pratico quando non porta vantaggi. Il costo di update non deve rallentare il lavoro normale.
- `donor-adaption` è una linea congelata perché si era allontanata dal goal. Il proprietario autorizza consultarne le evidenze, soprattutto GigaChat, quando utili; non riprendere automaticamente quella roadmap.

### 3.2 Risorse e run lunghe

- È autorizzato l'uso di E:, mantenendo directory ordinate, binding, dimensioni, hash e tracciabilità.
- **Nessun subagente, nessun Luna.** L'ultima istruzione è “Ignora Luna, fai tutto te”. Non creare task o monitor delegati.
- Per run di ore: comunicare in anticipo motivo, budget, risorse e condizioni di arresto; lanciare una run posseduta e limitata; **mettere il goal in pausa**, come richiesto dal proprietario. Non continuare a campionare ragionamento per ore nell'attesa.
- Non c'è una T4 attualmente allocata. L'ipotesi di usare anche un mese o più di T4 è una disponibilità progettuale, non una risorsa attiva.
- Annunciare uso e budget della RTX 3060 locale prima di impiegarla. Considerare giochi e applicazioni dell'utente; non chiuderli.
- Non creare automazioni o processi di monitoraggio ricorrenti senza richiesta.
- Nel lavoro in corso, aggiornamenti brevi e significativi almeno circa ogni minuto; non narrare polling invariati.

### 3.3 Git e lavoro estraneo

L'`AGENTS.md` del proprietario impone nessuna firma, attribuzione automatica, nome di assistente/modello/fornitore nei messaggi, branch o trailer. Conservare l'identità Git del proprietario. Nuovi branch descrittivi, senza prefissi di attribuzione. I nomi tecnici di donor, librerie e protocolli sono leciti e necessari.

Non fare staging globale. Tre file tracked erano già modificati da lavoro estraneo:

- `benchmarks/donor_adaptation/configs/_manifest.json`;
- `benchmarks/donor_adaptation/density/build_document_holdout.py`;
- `docs/research/RESEARCH_INDEX.md`.

Esistono moltissimi untracked estranei, inclusi `AGENTS.md`, `KAGGLE.md`, log e artefatti GigaChat. Non eliminarli, non includerli in un commit generale, non fare reset/clean. I commit della ricerca sono sempre limitati ai file espliciti; firma automatica disabilitata per il comando quando necessario, senza cambiare l'identità configurata.

### 3.4 Processi e filesystem Windows

I PID sono riutilizzabili: verificare PID + tempo di creazione + command line, o il vero terminale completato. Storicamente c'erano un publisher `pythonw` PID 21640 e figlio 9832, estranei; non terminare processi per somiglianza di nome. Il gioco dell'utente era presente durante alcune catture.

`Start-Process` deve avere `-WindowStyle Hidden` per helper non interattivi. Per worker GPU è stato usato `DETACHED_PROCESS=8`; `CREATE_NO_WINDOW` aveva provocato un difetto di guardia relativo a `conhost`. Non reinventare il launcher senza leggere la storia.

Nessuna rimozione o move ricorsiva su percorsi calcolati non verificati. Usare PowerShell end-to-end e `-LiteralPath`; non passare elenchi generati da PowerShell a `cmd /c` per cancellare. La sandbox è disabilitata e l'approval policy è `never`; non fornire `sandbox_permissions` ai comandi.

## 4. Fonti autoritative e vocabolario delle evidenze

La directory principale della ricerca è `D:\_THINGS\Progetti\SiliconLLM\docs\research\NATIVE_EXPERT_SCALING_20260925`.

| Fonte | Funzione |
| --- | --- |
| `INDEX.md` | Ingresso breve e prossimo passo, aggiornato anche per l'interruzione audit e la consegna |
| `METHOD.md` | Procedimento in costruzione, prerequisiti, fasi e limiti; non un diario da riscrivere ogni volta |
| `PRIOR_EVIDENCE.md` | Ledger molto ampio; top recente più autorevole delle vecchie frasi live/next in basso |
| `*_PROTOCOL_*.md` | Domanda, binding, scelte e soglie congelate prima della run |
| `*_RESULT_*.md` | Interpretazione del risultato e scope; risalire ai JSON per numeri esatti |
| `*_binding*.json` | File, hash, dataset, runtime, freeze, budget e scelte |
| `*_result*.json` | Osservazioni del producer, non automaticamente audit indipendente |
| `*_adjudication*.json` | Controlli indipendenti dichiarati e loro esito |
| `*.terminal.json` | Identità del worker, durata, picco OS, errore e descriptor del risultato |
| `*_session_closure*.json` | Chiusura dei processi, hash e famiglia di artefatti |
| `ADMISSION_*`, `RETENTION_*` | Ammissione/custodia degli esperimenti numerati storici, non completamento del goal |

**Numeric PASS** significa che il programma e i risultati corrispondono alla matematica e ai dati dichiarati. Può accompagnare **quality FAIL**. Un `COMPLETE/ADMITTED` locale non implica che il chatbot, il costo fisico o lo scaling abbiano passato. “Oracle” indica accesso a informazione costosa o privilegiata: può diagnosticare un limite, non fornire un router gratuito.

I campioni storici già usati come `consumed validation` non diventano nuovo held-out rinominandoli DEV. Le interpretazioni distinguono corpus noto, dato non consumato, FIT e DEV. Un hash attuale non ricrea la custodia completa di un worker storico che non aveva congelato tutte le dipendenze transitive.

Lo snapshot `HANDOFF_SNAPSHOT_20261014` conserva `INDEX.md`, `METHOD.md`, `PRIOR_EVIDENCE.md`, obiettivo completo e handoff precedente come si presentavano alla consegna. I riferimenti relativi dentro le copie puntano logicamente alla directory originale: leggere i documenti live quando servono i link. Lo snapshot preserva il contenuto, non attesta nuovi audit di tutti gli artefatti esterni.

## 5. Il motore originale: perché vale ancora la pena tornare a lui

`engine.c` e `benchmarks/phase60/engine.c` sono il patrimonio iniziale. La configurazione originale usa circa D=256, N=96, DN=512, dt rank=16, H=8, EH=128, K=8, L=6: cinque SSM e una SWA a finestra 128. Usa FFN gated dReLU, ternari packed, LUT `pshufb`, sparsità ed esperti granulari. È un nucleo appreso in un dominio sandbox; non è il donor Falcon trasferito.

Il vecchio handoff conserva i risultati TinyStories, byte/BPE, ricordo sintetico e fasi 42–61. Principali risultati utili:

- Ternary-LUT `pshufb` 4,2–5,0× rispetto a matvec F32 in microbench Zen 2, bit-exact sul riferimento intero; qualità MLP QAT nel sandbox con costo dichiarato +0,028 BPB, single-seed.
- Gated dReLU produce forte sparsità: circa 92% hidden / 79% gate; la perdita +0,0006 BPB è dentro la variabilità osservata, non un teorema di costo zero.
- La working set sweep trova una discontinuità intorno a 16 MB per CCX. Il layout sequenziale e la granularità dei gather sono fondamentali; non basta comprimere il numero di bit.
- Il set attivo cambia quasi come una base rate indipendente da un token all'altro. Hot pool temporale e prefetch basati sul token precedente non sono stati dimostrati. La prevedibilità utile era soprattutto in-place.
- MoE granulare E32×h128, top-8: BPB 0,8589 contro dense 0,8799 nel setup dichiarato; nessun esperto morto e buon bilanciamento. Si tratta di uno studio piccolo, single-seed, non di trasferimento 100B.
- Phase60 C: riferimento F32 176,1 token/s; LUT, skip e exp deterministico portano fino a 848 token/s; variante granulare E4 circa 701,7 token/s. Gate end-to-end mantengono la parità nel sandbox.
- Una LUT bit-exact aveva coesistito con un overflow di staging che peggiorava il sistema di circa +0,15 BPB. È la lezione permanente: **correttezza dei componenti non implica correttezza della composizione**.
- Phase61 ternarizzazione delle proiezioni SSM fallisce il gate: arm all 0,8940 BPB, in/out 0,8972, gate 0,8899. Gli organi di controllo sono sensibili alla precisione in quel regime.

Quelle velocità sono una prova della convenienza di alcuni kernel e di una architettura appresa per loro. Non attestano costo indipendente da qualunque n, DRAM fisica a pool arbitrario o conservazione della conoscenza di un chatbot. Il proprietario accetta estensioni del motore, purché conservino l'obiettivo capacità/costo e abbiano una via di conversione verificata.

La linea recall originale ha studiato SSM con un indice selettivo e IVF-PQ/two-stage. È distinta dalla linea weight-streaming/MoE e non è ancora automaticamente fusa nel chatbot Falcon. Il vecchio handoff descrive gli esperimenti MQAR, normalizzazione, drift e costi; non importarne la scala 128K come prova del contesto del nuovo modello.

## 6. Donor Falcon-H1: contratto matematico e numerico

### 6.1 Identità e geometria

Donor principale: Falcon-H1 1.5B chatbot locale, source revision `80ebc50d`, directory `results/native_expert_scaling/falcon_1p5b_source_repair1_20261008`. Checkpoint di circa 3,1 GB: **411 tensori BF16, 1.554.863.488 parametri**. Source hidden=2048, 24 decoder layer, FFN intermediate=4608, vocabolario=65537, RMS epsilon=1e-5.

La ricorrenza Mamba del source ha 3072 canali, N=256, 48 head da 64, un gruppo, causal conv di larghezza 4 con bias. `in_proj` è 6704×2048 con layout `[gate3072, x3072, B256, C256, dt48]`. `out_proj` è 2048×3072. `A_log` e `D` sono per-head, 48 valori. Ci sono full skip e normalizzazione post-gate propria.

Il source ha attenzione ibrida; la ricorrenza C qualificata usa N=96 derivato dal donor, ma mantiene il residuo a 2048 e molti operatori source. Il decoder originale è, schematicamente:

1. norm iniziale;
2. SSM e attenzione con i loro moltiplicatori;
3. somma al residuo;
4. RMSNorm pre-FFN;
5. FFN SwiGLU con moltiplicatori;
6. somma al residuo.

L'output del ramo C è **prima** del moltiplicatore SSM-out: il decoder lo applica una sola volta. Raddoppiarlo o ometterlo cambia il modello.

### 6.2 FFN esatto e scalari

In notazione reale, senza bias:

```text
u = U x
g = alpha G x
F(x) = beta D [u * SiLU(g)]
alpha = 0.4419417382415922
beta  = 0.13020833333333331
```

Gli scalari Python hanno un effetto F32 prima di alcuni rounding BF16. Non convertirli prematuramente a BF16. Il moltiplicatore embedding Python è circa 5,656854; una prova su 33.026 valori BF16 ha mostrato 772 differenze usando uno scalare BF16 anticipato. La catena corretta è definita dal programma del source e dai witness salvati, non dalla sola formula reale.

Moltiplicatori source rilevanti: SSM-in 0,625; SSM-out circa 0,11785113019775793; head 0,01953125; attention-out 0,234375. MuP sul vettore preconv completo usa valori BF16 `[0.353515625, 0.25, 0.1767578125, 0.5, 0.353515625]` sulle parti dichiarate.

RMSNorm source: convertire hidden a F32 per varianza/rsqrt; applicare normalizzazione; tornare BF16 al punto stabilito; moltiplicare gamma BF16 nell'ordine source. Cambiare l'ordine altera i bit. Ogni risultato distingue modello reale F64 proxy, F32, BF16 e programma C.

### 6.3 Runtime locale congelato

- Python 3.12.10: `C:\Users\giosa\AppData\Local\Programs\Python\Python312\python.exe`.
- Invocazione standard: `-I -S -B -X utf8`.
- Site esplicito `results/native_expert_scaling/chatbot_source_runtime/site`, con junction di singoli package verso `.venv/Lib/site-packages`.
- NumPy 2.4.6, psutil 7.2.2, Torch 2.6.0+cu124, Transformers 5.13.1 nelle catture dichiarate. Controllare binding e ABI reali prima di una nuova run.
- Ryzen 5 3600X, 80 GiB RAM, RTX 3060 12 GiB; CPU 6 thread e affinity 0..5 nei protocolli recenti. Desktop condiviso, scheduler non isolato.
- Driver storico 610.47; **componenti NVIDIA cambiati il 14 ottobre**, quindi non assumere questo driver come runtime GPU corrente.
- LLVM-MinGW clang: `C:\Users\giosa\AppData\Local\Microsoft\WinGet\Packages\MartinStorsjo.LLVM-MinGW.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\llvm-mingw-20251216-msvcrt-x86_64\bin\clang.exe`.
- Flag kernel recenti `-O3 -mavx2 -mfma -march=znver2 -ffp-contract=off -shared`, link `-lm`; reference scalare con vectorization/SLP disabilitati. Cap compiler 45 s.

Il loader source riuscito usa `FalconH1ForCausalLM.from_pretrained` locale, `trust_remote_code=False`, `dtype=torch.bfloat16`, `attn_implementation='eager'`, CUDA/eval/no-grad. `chatbot_falcon_ssd_tiles.py` installa patch verificata, chunk 128, fast path false. `true_generator_input_capture_v3.py` verifica i 411 parametri: nomi, shape, valori BF16/F32, device, identità, `_version`, no-grad.

Run GPU: seed0, deterministic algorithms, TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `CUDA_VISIBLE_DEVICES=0`, HF offline/telemetry off, tokenizer parallelism off, thread/interop dichiarati. Non caricare Torch per gli audit CPU che possono usare il lettore BF16 diretto `falcon_auxiliary_s6_transport.tensor_reader`.

### 6.4 Corpus, split e gate

Corpus principale `results/native_expert_scaling/chatbot_broad_capture_repair1_20261009/corpus.json`: 48 record, 12 domini.

| Split | Casi | Posizioni etichettate | Posizioni di storia |
| --- | ---: | ---: | ---: |
| FIT | 24 | 4422 | 11148 |
| DEV | 24 | 4386 | 11399 |
| Totale | 48 | 8808 | 22547 |

Massimo history 1507. I record conservano token/posizioni/source BF16 logits `[labels,65537]`, dominio e split. I task canonici sono 16, binding `recovered_categorical_native_chat_binding_20261012.json`; followup sulla propria risposta: nome animale **Milo**.

Gate whole-model congelati:

- task almeno 12/16, ogni categoria almeno 2;
- blank al massimo 1, stop valido e nessuna perdita di token speciali;
- DEV media case KL ≤1 e disagreement ≤0,20;
- ogni dominio KL ≤2 e disagreement ≤0,35;
- tutti 16 controlli prefix esatti;
- contatori interi esatti, mass defect ≤1e-6;
- followup su own-answer corretto.

Per i nuovi piloti FFN esiste un gate **locale diverso**: media case ≤0,10, worst case ≤0,20, domini DEV ≤0,20; scelta del minimo K/rank/count ammesso da FIT. Questo non cambia il gate source/proxy 1% né i gate del chatbot.

**RESERVED non aperto:** 104 record nuovi +64 più vecchi e longalign di lunghezza circa 8186–20874. Non aprirli per tuning o comodità: sono una futura verifica, con protocollo da congelare. I teacher logits storici derivano anche da una famiglia originariamente fallita; la custodia riparata dei byte non rende retroattivamente riuscito il worker fallito.

## 7. Storia precedente: scaling, Switch, ridondanza, proiezione compatta

Questa sezione evita di perdere mesi di lavoro senza trasformare il handoff in un elenco opaco di centinaia di file. Il ledger integrale è `PRIOR_EVIDENCE.md` (174.430 byte nello snapshot); i protocolli, ADMISSION e RETENTION delle campagne sono la fonte dettagliata.

### 7.1 Scaling preliminare e utilità di n

Sono stati raccolti dati su piccoli donor Qwen/Switch e metadati GigaChat. Alcuni studi storici di n=128→1280 con piccoli Qwen BF16 mostrano utilità in setup specifici; altri hanno qualità mista. Non generalizzare a n arbitrario, 10B o 100B.

Un router strutturato sintetico con rank64 e n=273408 ha avuto costo circa 0,44 ms; confronto flat più piccolo circa 1,39 ms. È un'evidenza di implementazione e fattorizzazione del punteggio del router, senza dimostrazione di massa del donor, capacità distinta o DRAM del modello.

I budget Phase64 E128/E256 sono **modellati**, non velocità di quei modelli addestrati. Il path esperti reale incorpora gather, dispatch e kernel: circa 4,2 GB/s integrato contro 17 GB/s kernel t6 nelle evidenze storiche. Non sostituire il costo con una banda ideale.

Un costo originale CPU Switch n128 warm storico era circa 54,8773 token/s, lower circa 53,4759, citato nella reassessment dopo 499. Quel risultato non appartiene al nuovo artefatto trasformato e non gli trasferisce il gate ≥50.

### 7.2 METH499: migliorare l'encoding non risolve funzione e routing

Riferimenti `METH_499_FACTOR_PROTOCOL_20261006.md`, `METH_499_FACTOR_RESULT_20261006.md`, `METH_499_WHOLE_ALGEBRA_AND_NEXT_20261006.md`, `ADMISSION_499_20261006.json`, `RETENTION_499_20261006.json`.

499 fattorizza esplicitamente output non pesato `F` e massa `p`, usando un prior/feature e una correzione vincolata sulle righe development. La soluzione è verificata con condizioni KKT indipendenti, join canonici UID/occorrenze, quantizer fisici e source target originali. Non divide impropriamente `roundF32(pF)` per p per ricostruire F.

Sul dominio consumed weighted, errore dei coefficienti I8 11,210455%→0,287945%; native oracle 14,759822%→9,724962%. L'errore fitted F64 è ancora 9,718714% contro 9,518808% di 498: encoding migliore, generalizzazione non migliorata.

Sul coupled vero l'errore consumed è circa 90,597547%; ID fidelity 22,701495%; errore RMS massa winner circa 64,792158%. Choice counterfactual e mass increment sono separati: una funzione accurata con IDs o masse errati resta sbagliata. Rare class e singoli domini falliscono anche quando la media encoding è buona.

Development non determina le componenti nel kernel delle feature: nella formulazione di 499 si contavano 53.943 direzioni di coefficienti non determinate, 41.428.224 scalari per 768 output. Il prior le sceglie, ma non prova conoscenza trasferita fuori dalle righe osservate. Non replicare una fit lunga identica senza cambiare l'informazione o la geometria.

### 7.3 METH522: limite della sola copia di supporti, non della ridondanza generale

`METH_522_REDUNDANCY_LOWER_BOUND_RESULT_20261007.md` studia branch di massimo 512 neuroni originali e budget logico rho2: ≤6144 incidenze per parent da 3072. Supporto esatto sufficiente per preservare anche max/quantizer, ma non necessario per un'approssimazione.

Su development 11721 UID /127 parent esposti: 10 parent hanno supporti >512 (massimo 764); 29 hanno bound di copie >6144; 19 di questi sono ulteriori casi senza supporti oversized. 288 UID oversized. Audit di tutte 63262 coppie del clique e ogni decisione della traccia. Parent 0 senza osservazioni non compilato/promosso.

Esempio parent113: supporti singoli ≤483, ma clique certifica almeno87 branch e 32159 incidenze, rho≥10,468424. Questo confuta **quella ricetta** B512/rho2 per supporto esatto, non ogni RAM, più branch attivi, altre larghezze, funzione folded, cancellazione o distillazione.

L'idea del proprietario — duplicare foglie per consultare branch piccoli più completi, anche raddoppiando i parametri — resta una via legittima. Devono essere verificati conservazione dei contributi, costo attivo, specializzazione e storage/traffico fisico. Il conteggio logico delle copie non è DRAM misurata.

### 7.4 Compressione lossless di master e supervisione

`ORIGINAL_MASTER_TRANSITION_RESULT_20261010.md`: 92 tensori, 721.008.128 parole F32; transizione via XOR verificata byte per byte, 54,342 milioni parole cambiate. Raw circa2,884GB→186,158MB, **15,4924×**. Copy35/compressed719/raw0 chunk nella transizione osservata; fallback raw testato. Non è una garanzia sul rapporto futuro né elimina master, backward o optimizer O(n).

Il compilatore categorical loss conserva esattamente momenti/entropy/IDs di 8808 label, 72 controlli dense/compiled e gradienti: payload circa18,144MB contro1,1545GB logits BF16, **63,628×**. È una compressione della supervisione per una geometria head fissa, non una compressione lossless universale del chatbot o delle attivazioni.

### 7.5 Tentativi del piccolo target e fallimenti whole-model

La linea originale comprimeva hidden2048→256, cambiava numero di layer, FFN, norm, routing e precisione insieme. Una campagna categorical recuperata produceva task0/16, DEV KL circa94,6 e disagreement99,8%, pur con circa159 token/s grezzi. La sua grande famiglia di circa138GB e le ricevute sono state auditate: non scartarla senza leggere i difetti, ma non chiamarla utile.

Un altro shadow all24 con D256 produceva0/16, DEV KL22,62 e disagreement100%; la proiezione manteneva circa13% energia input. Non prova un'impossibilità di tutti gli encoder a256 o di tutti i target compatti: prova il fallimento di quella geometria e composizione.

Le run di recovery on-history della linea originale avevano ancora native DEV KL circa6,52 e0/16 su due arm dopo24 update. La ricerca si è quindi spostata verso preservazione funzionale degli operatori e controllo delle coordinate, prima di ulteriore training identico.

## 8. Algebra del readout e informazione persa

Riferimento fondamentale `TARGET_READOUT_GEOMETRY_20261010.md`. La proiezione rettangolare non è un semplice cambio di coordinate. Con h∈R2048, P∈R2048×256, z=Pᵀh e Q=PPᵀ, head W, gamma diagonale G e RMS r:

```text
l_D = W G h / r(h)
l_C = W_C G_C P^T h / r_C(P^T h)
DeltaW = W_C - W P

l_D - l_C = W G (I-Q) h / r
            + W (G P - P G_C) z / r
            + W P G_C z (1/r - 1/r_C)
            - DeltaW G_C z / r_C
```

Quattro cause diverse: direzioni eliminate, gamma proiettata incompatibile con una diagonale, denominatore norm alterato e head adattata. La somma è un'identità nel regime ortonormale/reale dichiarato; i rounding e il difetto reale di base sono ulteriori. I termini possono cancellarsi e non sono quote additive di KL.

Una rotazione quadrata può preservare la norma e trasferire opportunamente una gamma densa coniugata. Eliminare colonne cambia la funzione. Copiare WP e `diag(PᵀGP)` inizializza un target; non dimostra trasporto della funzione.

Softmax ignora il common shift. Per piccoli delta logits:

```text
KL(q || softmax(l+delta)) ≈ 0.5 delta^T [diag(q)-q q^T] delta
J_RMS(h) = W G [I/r - h h^T/(d r^3)]
```

La geometria locale del residuo è JᵀFisherJ, non solo energia Euclidea. È una Taylor locale, non un bound valido automaticamente quando KL è6,52 o94. Argmax/disagreement resta discontinuo vicino ai tie.

Per un encoder fisso Z=f(H), il miglior decoder probabilistico illimitato soddisfa `inf_p E KL(q(Y|H)||p(Y|Z)) = I(Y;H|Z)`. Più esperti dietro lo stesso encoder non ricostruiscono informazione condizionale davvero persa. I dati finiti qui non stimano quella mutua informazione e un fixed-head assay non ne calcola l'infimum. Non convertire una perdita di energia in una percentuale di conoscenza.

### 8.1 SwiGLU e dReLU non sono la stessa funzione

L'identità `g*u = sum_(s,t=±1) s*t*ReLU(sg)*ReLU(tu)` conserva il bilineare. Il donor SwiGLU è `g*u*sigmoid(g)`. Le quattro repliche sign dReLU senza sigmoid non conservano il neuron source, anche con copertura completa. Routing normalizzato, selezione top-K, ternarizzazione e AQ aggiungono altre approssimazioni. Più copie non correggono automaticamente un operatore cambiato.

La futura ridondanza deve scrivere esplicitamente, per f=Σ_j f_j e expert g_r=Σ_j a_rj f_j, la ricostruzione Σ_(r∈S(x)) w_r(x)g_r. Conservazione esatta richiede Σ_r w_r a_rj=1 per ciascun contributo rilevante. Se router usa masse normalizzate, le scale devono essere coerenti. Per originali disgiunti il top-K può eliminare informazione; duplicare può aiutare la copertura, al prezzo di storage e controllo.

## 9. Ricorrenza: dalle approssimazioni locali alla composizione utile

### 9.1 Studi locali prima della cattura degli input veri

La linea usa basi source-trained e controlli di ricorrenza, non un core casuale:

- `RECURRENT_ONLY_CHANNEL_SKIP_RESULT_20261012.md`: mantenere full skip migliora le approssimazioni di ricorrenza a384 canali, ma i gate locali ancora falliscono; ricorrenza e skip possono cancellarsi.
- `RECURRENT_OUTPUT_BASES_RESULT_20261012.md`: FIT recurrent-output bases migliorano site23, ownnorm mean circa0,06357/worst0,08701, ma site0 circa0,27308 e site12 circa0,79273 restano fail.
- `COUPLED_RECURRENCE_SENSITIVITY_RESULT_20261012.md`: derivazioni/JVP/energie e head sensitivity qualificate; remainder tangent site12 circa30% FIT/33% DEV. Un obiettivo finite è più appropriato di un fit tangent medio in quel setup.
- `FINITE_RECURRENCE_GRADIENT_RESULT_20261012.md`: gradienti/retraction/FD completi e un update FIT; ownnorm DEV site12 circa0,792727→0,789261, ancora fail.
- `WHITENED_RECURRENCE_PREFLIGHT_RESULT_20261012.md` e `WHITENED_CANDIDATE_RESULT_20261012.md`: whitening preserva inizialmente la funzione e migliora condizionamento; alpha0,1 scelta FIT, DEV ownnorm circa0,775702, ancora fail. Non continuare epoch identiche senza nuova causa.
- `RECURRENT_AXIS_COMPOSITION_RESULT_20261013.md`:384 canali×state96, ownnorm DEV mean0,068813/worst0,096949 al site23, stato circa21,33× più piccolo di source full; source generator/gate/out densi ancora presenti.
- `COMPOSED_OPERATOR_MIXED_RESULT_20261013.md`: C mixed F64 decode/gate/RMS→BF16→AVX F32, tutti timestep e92M score bit-exact; scan circa10µs, norm/out circa1157µs, totale supplied operands circa1248µs, input GEMV separato circa2535µs. Non è una pipeline layer completa.
- `FULL_CHANNEL_NATIVE_RESULT_20261013.md`:3072 canali×96, scan circa59µs e supplied total circa1300µs; 8× canali non implica8× costo whole, perché i generatori/readout dominano. Local ownnorm gatepass ai tre site; source-denominator fault trattenuto.

Questi numeri appartengono a workload e periodi diversi: non sottrarli per inventare un speedup composto. Il progresso pratico è una ricorrenza qualificata che può essere testata su input realmente causali.

### 9.2 Recuperare h dall'inverso del gate era una scorciatoia errata

`GATE_INPUT_OPERATOR_RESULT_20261013.md` verifica full rank2048 delle matrici gate, con condition circa27/71/55. Rank pieno non significa che invertire attivazioni BF16/rounding dia l'hidden originale esatto. Il recovery inverse-gate aveva errore circa0,5%; un caso Numina finalstate1,1258% falliva.

`TRUE_GENERATOR_INPUT_CAPTURE_RESULT_20261013.md` cattura il **vero h** di ingresso al generatore in tutti24 layer, su storie proprie. Verifica tutti411 parametri BF, nuovi input e replay. Dataset circa4,233GB,1152 array; cattura GPU circa334s, audit CPU circa80s nei protocolli. Il caso Numina vero input porta finalstate circa0,006441%, senza cambiare il core.

La lezione è specifica: input ricostruiti e input osservati non sono intercambiabili. Per FFN servirà un'altra cattura: h prima di SSM non è x dopo attenzione/residuo/RMS pre-FFN.

### 9.3 Head e probability-image: progressi di diagnostica, non chatbot finale

La linea head ha qualificato momenti source, supporti, affine rays, gradienti GPUF64 e oracle probability su tutte8808 label. Riferimenti `NATIVE_HEAD_AFFINE_RAYS_RESULT_20261012.md`, `NATIVE_HEAD_GPU_PREFLIGHT_RESULT_20261012.md`, `ALL_LABEL_NATIVE_PROBABILITY_GAP_RESULT_20261012.md`, `DUAL_ENTROPY_MIXTURE_RESULT_20261012.md`, `SOURCE_HEAD_FEATURE_MOMENTS_RESULT_20261012.md`.

Un oracle native probability raggiunge DEV case KL circa0,0483465 e disagreement2,3003%, rispetto al candidato precedente2,93425/41,09%; non è ottimo certificato e non è un modello causale. Winner coverage98,66% non assicura distribuzioni: winner-only rays potevano avere KL circa9,6007. Le dual entropy family esplorate non forniscono un ostacolo positivo ai budget. Non trasformare questi risultati in una head pronta del target.

### 9.4 Bridge full-width: primo risultato chatbot composto della linea nativa

Report `FULL_WIDTH_NATIVE_CHAT_RESULT_20261013.md`, finale commit `9448d9b`. Codice `full_width_native_chat.py`: `NativeBridge` sostituisce il forward Mamba in tutti24 layer con contesti C persistenti propri, conservando residuo2048, FFN/attenzione/head originali sul source GPU.

- Primo layer: U32 token ID reale→tabella F0 completa, senza padding/teacher lookup di hidden.
- Layer successivi: source hidden del decoder corrente, SSM-in, source in_proj e MuP→righe BF16 da6704→CPU native state.
- Stato96, conv e ring sono propri; reset solo all'inizio di una nuova sequenza.
- Attenzione originale usa `DynamicCache`; `get_seq_length(site)` deve coincidere con gli step consumati.
- Output BF native copiato con ownership corretta→GPU; moltiplicatore SSM-out applicato una volta.
- Head source calcolata sulle posizioni richieste, moltiplicatore head e BF/F32 nell'ordine verificato.

Risultati sul corpus noto:

| Metrica | Risultato |
| --- | ---: |
| Task | 14/16 |
| Followup own-answer | Milo corretto |
| DEV case KL | 0,02155979257 |
| DEV disagreement | 0,04884743007 |
| Massimo dominio KL / disagreement | 0,051393 /0,1110444 |
| Controlli prefix | Tutti16 esatti |
| Sequenze / token consumati / native posizioni | 97 /24169 /580056 |
| Parole output e intere auditate | 1.187.954.688 per famiglia; mismatch0 |

Audit completo verifica state/ring/output, ricorrenza indipendente e molti intermedi: GG maxrel8,94934e-5; B'/C' circa1,72e-7/2,94e-7; scan1,365e-6; state1,909e-6; fields e ownnorm0. Tutti411 parametri source prima/dopo esatti. I due errori task `arithmetic_div4` risposta2 e `history_color` rispostaGreen restano fail; nessuna eccezione creata per promuoverli.

Famiglia E `full_width_native_chat_20261013`:4805 file,17.235.894.707 byte. Producer circa3576s, audit circa2658s; producer OS~4,03GB/GPUalloc3,187GB/reserved3,265GB. Tutti sei comandi e closure riusciti. Audit SHA `af08c1f0800339ad689d0c5101f111577f91e97033b8b94096200ba472453744`; closure SHA `2e5ec100a7d580818cc4edc757999ddce9792bd3de84e037a3eafe868cdf3129`.

**Limiti:** GPU source FFN/attention/head, generatore contestuale source denso; non CPU finale, non useful n e non ≥50. Le188 pin producer storiche non includevano tutte le dipendenze transitive (`composed_operator_mixed.py`, corpi tokenizer/Jinja);496 pin supplementari audit non sanano retroattivamente il gap. L'attenzione GPU è testimoniata dalle tracce/cache, non da una reimplementazione aritmetica indipendente della GPU.

## 10. Componenti disponibili in C e compilazione source

### 10.1 Source-only F0 per il primo layer

`TOKEN_GENERATOR_COMPILER_RESULT_20261013.md`, finale commit `13a29c8`. Compila embedding, first norm, in_proj e MuP per **tutto il vocabolario**, dai soli coefficienti source prima di confrontare risposte.

- `F0.BF16`:65537×6704,878.720.096 byte;439.360.048 parole.
- `H0.BF16`:65537×2048,268.439.552 byte;134.219.776 parole.
-48 field archive e conv param; famiglia1.449.937.450 byte.
-Ogni parola riprodotta da audit CPU dell'equazione dichiarata;151M causal field words esatte.
-Source CUDA H0 storico differisce357 parole in11 casi; max0,0092775%, sotto gate norm0,01%. Differenza trattenuta, non cancellata.
-Field worst0,0433294% e incremental worst0,03745%, sotto gate1%.

Il lookup selezionato costa13408 byte per ID. Il vocabolario contiene65537 righe: **non usare U16 per gli ID token**. Il fatto che molti ID osservati stiano sotto65536 non autorizza la truncation dell'ultimo.

### 10.2 Token→own state

`TOKEN_STATE_RESULT_20261013.md`, finale `0f8f21d`. F0 reale, conv causale,3072×96 state, skip, own gate/RMS, quantizer I16 e readout accettato. Su tutte48 storie, blocco completo e ogni continuazione token-by-token sono bit-equal. Output extra worst0,06174%, finalstate0,01502%; source FIT mean/worst6,506/8,875%, DEV7,639/9,341% nel gate locale storico.

State-ready batch1,60 call: median1012,90µs warm,1044,05µs source-evicted. Misura **solo primo ramo**; non moltiplicarla per24 per chiamarla benchmark e non invertirla in token/s del chatbot. Famiglia E1.763.186.800 byte; tutti5 comandi chiusi.

### 10.3 All24 contextual-row ABI e parametri

`ROW_STATE_RESULT_20261013.md`, finale `32b1c5d`. `row_state.c`, `row_state.py`, auditor/closer; DLL92672 byte, SHA256 `49f8400e5fc3d2ae3bf30d2de32d36f76c863c82a0343ba59b0f7bfbbbaaa447`. Famiglia E2.131.959.652 byte.

All24 pacchetti `site00..23.parameters.npz` contengono:

- Ws/Vs96×256F32, A_log/D48F32, gamma3072F32;
- convw3584×4F64, convbias3584F64, dtbias48F64;
- q2048×3072I16 nel range[-364,364], scales2048F64=`maxabs(O_row)/364`, round-nearest-even.

Audit source-only: tutti150.994.944 coefficienti q e1.686.912 core/conv; duality WᵀV96 max2,27e-14. Tutte48 initial-layer histories /22547 posizioni esatte al token branch. All24 su16 righe FIT arbitrarie verificano parametrizzazione, non erano da sole prova dell'input reale dei layer successivi; quella prova arriva dal bridge.

ABI concreta:

```text
TokenState:
  uint64 seen
  uint32 last_id
  uint16 ring[3*3584]
  float32 state[3072*96]

ts_abi = [65537,6704,3072,96,3584,3,2048]
rs_abi = [6704,3072,96,48,2048,3]

ts_block: context, n, uint32 token IDs, uint16 F0 table, parameter pointers,
          output/intermediate pointers
rs_block: context, n, uint32 provenance tags, contiguous preconv BF16[n,6704],
          same parameter/output pointers
snapshot: own state float32, ring uint16, seen uint64
last: UINT32_MAX when seen == 0
```

`row_state.interface()` restituisce DLL/create/destroy/tokens/rows/snapshot. n ammesso1..1507. Pointer parameter grouping: w=[Ws,Vs,Alog,D,gamma], conv=[cw,cb,dtbias], f=[Bp(n,96),Cp(n,96),latent(n,3072)], bits=[fields(n,6704),norm(n,3072),output(n,2048)]. r(n,3072)I16, dots(n,2048)I64, a(n)F64, raw(n,2048)F64.

**Ownership:** mantenere vivi gli array NumPy originali finché C usa i pointer. Fare cast su temporanei e conservare solo l'indirizzo è un use-after-free possibile. La replica deve usare il wrapper verificato. Nonfinite input è rifiutato prima di modificare stato.

Numerica C: conv BF→SiLU BF, delta con bias/softplus threshold20/BF; B/C AVX F32; decay `expf(dt*(-expf(Alog)))`; FMA state +Dx; norm post-gate F64/gamma→F32/BF; quantizer RNE(norm/maxabs*32767); dot esatto I64; scala input F64 **prima** della scala riga F64; F32/BF finale.

### 10.4 Readout source full rank: precisione, interi e LUT

La matrice source O2048×3072 resta full rank numerico2048 ai site esaminati. `TERNARY_READOUT_PRECISION_RESULT_20261013.md` sceglie FIT p6 tra p4/5/6; p4 e p5 falliscono site12. Sei plane trit rappresentano q[-364,364], scales riga, rounding dichiarato. Extra1% raw/BF e donor-local gatepass su144 casi; coefficienti/packed/spectra/output interamente auditate. Non rifare p4/p5 senza una nuova variabile.

Tre backend fisici vanno distinti:

| Backend | Precisione | Costo misurato | Esito |
| --- | --- | --- | --- |
| Five-trit pattern/F64 LUT native p6 | Tutti output riprodotti | circa5,1–5,3ms vs dense1,16–1,22ms |4,2–4,6× più lento, reject accelerazione di questo backend |
| I16 `PMADDWD`→I64 | Dot/scaling/BF esatti all'audit |0,674–0,858ms vsdense1,488–1,617ms |1,85–2,29×; backend accettato per il bridge |
| Byte LUT `pshufb`,6 trit weight planes×4 radix16 activation digits | Ogni dot/BF identico a I16 |6,86–7,73ms vsI16 0,807–0,875ms |8,50–9,10× più lento; reject questa implementazione24-path |

I16 deploy usa12.599.296 byte di pesi+scales, **non è esecuzione ternary-LUT**. PMADDWD accumula chunk128: worst int32 circa1,526B <2³¹, estende a I64 prima di sommare chunk. Full dot massimo circa36,640B <2^53, quindi conversione F64 esatta. Cambiare ordine delle scale può cambiare BF finale.

Byte LUT packed circa9,45MB, ma24 path leggono circa37,75MB di operandi logici per input. Byte su disco, byte logici e traffico DRAM fisico sono quantità diverse. Nessuno di questi benchmark evicted misura automaticamente PMU DRAM. Un backend LUT fallito non confuta tutte le LUT, ma non consente di ereditare i848 token/s di Phase60.

Riferimenti `INTEGER_READOUT_RESULT_20261013.md`, `EXACT_INTEGER_LUT_RESULT_20261013.md`, finali `8d2e0c5` e `7d3b200`. I16 tutti144 casi/207M input integer/138M output dot e BF auditati; V1 denominatore diagnostic errato e V2 variable-shadow in final hashing trattenuti; V3 PASS completo. Famiglie E2,910GB e2,527GB.

## 11. Limiti misurati di input generator e output bank

### 11.1 Token-only prototype + UNA correzione lineare condivisa

`TOKEN_CONTEXT_RANK_RESULT_20261013.md`, finale `2f8d2d3`. Veri input own-history ai site1/12/23; FIT token means BF16 + fallback globale, fattorizzazione residual pesata GG, rank0/128/256/512/1024 e sei metodi.2248 token distinti FIT;24,730239% token DEV sconosciuti, mantenuti nel report.

Tutte le sei varianti falliscono FIT ≤1024. Rank256 errori DEV field mean52,22/61,52/49,56%; rank102431,25/33,80/28,82%; nessun24/24 case passa gate1%. Residual energy17,09/48,61/45,06% è energia, non quota di conoscenza.

Il bound certifica rank necessario FIT almeno **1408/1857/1857**, per qualunque token-only prototype più **un unico** residual lineare comune nella famiglia dichiarata, includendo allowances di BF/prototype/source. η circa0,0078278789; source floor circa0,00469; weighted RMS ammissibile circa0,04487. Conditional mean minimizza la covariance; altre scelte aggiungono PSD.

Non è interval arithmetic certificate né un limite universale a funzioni nonlineari/condizionali o encoder causali nuovi. La Jacobian di un generatore lineare G è G in ogni cella: suddividere input e assegnare lo stesso map non crea nuova compressione. Non rifare global/token rank grid identica.

### 11.2 Rank finite e banche di output condizionali

`SOURCE_FUNCTION_RANK_RESULT_20261013.md`: sui FIT osservati input rank256 errors24,87/33,64/31,13%; output rank25653,27/54,40/34,15%. Input rank1024 circa6,65/13,07/11,18%; output rank1024 circa16/19/12%. Sono RMS e matrici finite nel regime dichiarato, non impossibilità di un union of conditional subspaces.

`CONDITIONAL_OUTPUT_RESULT_20261013.md`: banche FIT tree/leaf means e rank output16cell/r128 oppure64cell/r32; tutte277M predizioni/IDs/masses/SVD qualificate. Anche l'oracle target-aware esaustivo verifica5.411.280 errori: DEV16/r128 circa0,6733/0,7587/0,4490,64/r32 circa0,7414/0,8483/0,5191. Tutti case1% falliscono anche su FIT.

**Deduzione limitata ma utile:** ottimizzare solo il router non può riparare la mancata copertura di quelle immagini affini fisse. Cambiare il readout, il rank, la famiglia di funzione o le coordinate è una nuova ipotesi; cambiare soltanto il ranking lasciando quelle banche uguali non risolve l'oracle failure.

## 12. Operandi FFN veri e difetto affine

### 12.1 Cattura all24 degli operandi realmente usati

`FFN_OPERAND_RESULT_20261013.md` e capture/audit protocol. Sei field BF16: x2048 **dopo SSM/attenzione/residuo/RMS pre-FFN**, up4608, gate_raw4608, gate_scaled4608, features4608 (vero input del down), output2048 dopo beta.

All24×48×22547 history positions:6912 raw file,12.190.531.584 parole BF,24.381.063.168 byte raw. Con1152 state NPZ,48 label F32,48 receipts, metodo e parameter witnesses: **8163 file,28.079.135.477 byte**, cap32GiB, famiglia `E:/SiliconLLM/results/native_expert_scaling/ffn_operand_capture_20261013`.

`StrictSink` osserva byte nuovi e confronta parent trace, non inietta hidden teacher. Native inH/preconv/output, state/ring/attention cache digest e577.249.896 label F32 nuovi sono esatti al parent. Tutti411 source coefficienti e identità/versions immutati; hook osservatori non cambiano FFN.

Binding v2 SHA `9903f7f0719909406225f095963753f8edb776247257050a0868eb0e0510b3e4`,9900 pin framework/numerical `.py/.pyd/.dll`. Non è ogni file OS/stdlib e non sana il parent storico. La prima prep chiedeva10GiB free e fallì prima CUDA; corretto gate actual `memory.free≥7GiB` con6GiB own alloc cap +1GiB margine, non `total-used` WDDM. GPU reale producer peak3.187.002.368 allocated /3.265.265.664 reserved.

Producer worker28776/holder31288, circa4765,922s/4828,344s; audit worker24928/holder13028 circa1103,5s/1109,329s. Tutti sei comandi chiusi0, originale prep fault trattenuto. Gate scalare/features0; upmax4,26599e-5, gate4,74706e-5; down+scalar max **0,004683556826**, sotto1%.

La causa dello scarto down non è isolata. Raw down non è stato catturato separatamente: non attribuirlo come fatto a BF16 reduced-precision reduction, allo scalare o a un'implementazione GPU. Audit indipendente copre aritmetica salvata; cache attention digest è witness, non replica indipendente dell'attenzione.

### 12.2 Pilota affine full-output

`FFN_AFFINE_PILOT_PROTOCOL_20261013.md`, codici `ffn_affine_pilot.py`, auditor. Site0/12/23, K1/16/64, FIT-only Euclidean kmeans++, seed0, max25 Lloyd.64 celle non sono un grande router strutturato: nearest-centroid flat su2048, un U32 ID e massa1.

Con centro μ, uc=Uμ, gc=αGμ, s=SiLU, l'approssimazione:

```text
A_mu(x) = beta D [s(gc)*(U x) + uc*s'(gc)*(alpha G x - gc)]
```

Jμ è2048×2048; nel pilota è **implicito**, calcolato attraverso i coefficienti source. Non è esportato o misurato come banco fisico. Salvare una matrice D² può usare meno coefficienti attivi di3DI, ma non prova né qualità né latenza.

Tutte nove varianti falliscono FIT mean≤0,10/worst≤0,20.64celle FITmean/DEVmean: site0 0,621696/0,621892;site12 0,836879/0,927598;site23 0,126993/0,134470. Ogni proxy/source control1% passa. Audit432 predizioni e144 controlli, tutte distanze/route, derivative central FD1e-4 max1,563e-9, saved F32pred max2,644e-8 PASS. kmeans replay condiviso non è prova di ottimalità del clustering.

Famiglia E441file/1.711.100.700B; producer449,188s e audit455,703s. Binding SHA `38affb161447013d7d1b9f03b5352112d34e2e31617fbf7bb3f9a472051bde3d`; risultato SHA `b19243cfdf292ece9b14830d01b1cfb12ddc57ff7a7e598237a79d2721ef9e8b`.

### 12.3 Decomposizione esatta gate/cross

`FFN_AFFINE_DEFECT_RESULT_20261013.md`, `ffn_curvature_split.py` e auditor, closure `ffn_affine_defect_close.py`, finale `91b540c`.

Per du=u−uc, dg=g−gc:

```text
a      = s(gc)*u + uc*s'(gc)*dg
Rgate  = uc*[s(g)-s(gc)-s'(gc)*dg]
Rcross = du*[s(g)-s(gc)]
F      = beta D [a + Rgate + Rcross]
s''(g) = sigmoid(g)*(1-sigmoid(g))*[2+g*(1-2*sigmoid(g))]
Qcross = s'(gc)*du*dg
Qgate  = 0.5*uc*s''(gc)*dg^2
```

Tutti144 casi e identità/energy/cross verificati, max6,820e-16. Una correzione prima dell'esecuzione ha sistemato la norma del termine: `||term||/||truth||`, non `||term-truth||/||truth||`; nessun risultato osservato prima del fix.

| Site | Affine DEV | Norma gate | Norma cross | Affine+cross esatto | Quadratico completo |
| --- | ---: | ---: | ---: | ---: | ---: |
|0|0,621901|0,097974|0,610236|0,097974|0,566415|
|12|0,927585|0,264810|0,992441|0,264810|0,506929|
|23|0,134429|0,149612|0,147118|0,149612|0,233021|

Cross inner/reference-energy mean:0,002198/−0,098310/−0,013526. Il cross domina a0/12, ma a23 gate e cross si cancellano parzialmente. Correggere un termine o aggiungere Taylor quadratico può peggiorare. Non sommare i quadrati delle norme medie per ricostruire l'energia: serve il cross innerproduct per ogni caso.

Questa diagnostica usa tutti neuroni/source, non è un kernel economico. Producer291,562s, audit307,984s; closure verifica6300 pin distinti, famiglie/receipt/protected. Result SHA `2f46557c1cd713b4a2a817f81de8d8a97c91e1ed69d45f2ef90973a77998d4d6`; audit SHA `0390c4780cba7eed865ca15bbcf23fcbf8a8e3a94f5a0920bb0b1d90308c951d`; closure SHA `ce6818aec7a213a3cc8e8bf687017d21a155f089fda815ea245c124be8f09d1f`.

## 13. Dalla correzione esatta alla scelta economica: ultimi quattro esperimenti

Il problema comune è conservare l'affine full-output e correggere il remainder nonlineare di pochi neuroni. Tutte queste prove riusano le64 celle FIT qualificate, gli input pre-FFN veri e i tre site0/12/23. L'affine è ancora un reference evaluator implicito; non leggere queste prove come costo fisico di una FFN sparse completa.

Per ogni neurone i, definire il **remainder combinato**:

```text
r_i(x) = u_i*s(g_i) - s(gc_i)*u_i - uc_i*s'(gc_i)*(g_i-gc_i)
w_i = || beta D[:,i] ||_2^2
score_i(x) = r_i(x)^2 * w_i
prediction_S(x) = A_mu(x) + beta sum_(i in S) D[:,i]*r_i(x)
```

La score è una regola per contributo individuale. Le colonne di D possono interagire/cancellarsi; i top score non sono il subset globalmente ottimo. Non chiamare un successo della score un limite inferiore di sparsità universale.

### 13.1 Sottoinsiemi fissi per cella: fallimento della regola FIT

`FFN_EXACT_SUBSET_RESULT_20261013.md`; codice `ffn_exact_subset.py`, auditor/diagnostic/V2/closer. Freeze `18e7da7`, binding `e9a9fa5`, finale `90c3586`. Binding SHA `16acd77970ce6cbe6ceb00a23cc0b37a2723d4bb8bc95ed2bb9e7df4117393cb`.

Ranking per cella: media FIT token-weighted di score_i, stabile per indice a parità. Empty FIT cell usa global fallback, senza DEV. Counts32/128/512/2048; controllo4608 escluso dalla selezione di una rappresentazione economica.

| Site | Decisione su FIT | Risultato DEV pertinente |
| --- | --- | --- |
|0|Nessun budget≤2048 ammesso;2048 FITmean0,108113/worst0,130057|2048 mean0,243356/worst0,398866|
|12|Nessun budget≤2048 ammesso;2048 FITmean0,244910/worst0,282628|2048 mean0,404537/worst0,506711|
|23|FIT sceglie128, mean0,085335/worst0,164889|Mean **0,101090835 >0,10**, FAIL;512/2048 locali passano ma non promossi usando DEV|

All4608 ricostruisce il proxy smooth fino a circa6,89e-16. Producer988,953s/OS1,199GB; E723file/2.785.911.606B. Result SHA `56e8a090da1230bb1bc8f838952a949c55dc352bc9701f1d56baf10f7f57e140`.

**Audit fault istruttivo:** V1 fallisce un'uguaglianza di ranking prima delle predizioni. In11 celle site0 le score producer sono esattamente zero e il tie-break per indice è corretto. Algebra indipendente equivalente introduce rumore fino a5,774e-29, che altera l'ordine di valori nulli. Celle8,15,23,24,26,31,40,48,56,59,63; dieci singleton FIT e cell56 con24 righe. Non dichiarare sbagliato il producer per una differenza di floating algebra a zero.

La diagnostica salva score/rank/count completi,7.080.504 byte su E.34/11148 FIT e26/11399 DEV token ricadono in quei supporti. Il conteggio non misura l'energia causata dal difetto e non spiega automaticamente tutta la perdita. Rivela che il training della scelta non contiene alcuna preferenza fuori dall'anchor per quelle celle.

V2 conserva il V1 fallito, riproduce **esattamente la score/rank letterale** e verifica separatamente l'algebra indipendente,720 predizioni/3modelli/144controlli. PASS completo424,859s, picco1.418.113.024B. Audit SHA `0233675a7d84b8258adacfd95762d7c617cbec8c9f74db87d3a5fe1f2de89b31`; closure SHA `08a040cd70f7d7471fe27a40d20273526e5f6a0ca09a5bdd43250ce1f87696d6`. Nessun gate allentato, nessuna cella esclusa. La prima diagnostica non aveva ricevuta autonoma PID/create; limite dichiarato, V2 riproduce ogni array.

### 13.2 Oracle per-token: esiste una selezione utile per questa regola

`FFN_REMAINDER_ORACLE_RESULT_20261013.md`; freeze `cb5f4fe`, binding `dc78161`, final `8724aef`. Seleziona top score **vera di quel token**, dopo aver calcolato tutte originali U/G e nonlinearity. Non usa nuovi centri/training/GPU/RESERVED.

Count guida è il primo tra quelli testati che passa FIT mean0,10/worst0,20. Gli stessi count passano poi DEV e domini:

| Site | Count FIT | FIT mean/worst | DEV mean/worst | Confronto con fixed-cell a stesso count |
| --- | ---: | --- | --- | --- |
|0|512|0,061027 /0,075778|0,062173 /0,076069|0,400908→0,062173, errore circa6,45× minore|
|12|2048|0,042296 /0,048495|0,048711 /0,054676|0,404537→0,048711, circa8,30× minore|
|23|32|0,066187 /0,124634|0,072805 /0,126221|0,113219→0,072805, circa1,56× minore|

A site12 oracle512 DEVmean0,225681 fallisce ancora. La scelta di2048 è costosa: non fingere che tutti layer richiedano solo32 o512 neuroni. L'oracle mostra perdita della regola aggregata fissa nella famiglia esaminata; non prova un router economico o un lower bound assoluto sul count.

Producer470,750s, OS1.242.435.584B, E720file/3.143.325.180B. Result SHA `a419e5e9c5c8b8780542d9ac4f9fdb5921876203a4ecc12c294811979e9d4c5a`. Selected IDs U16 (qui neuroni0..4607), labels cell U32, mass F64=1, predictionF32; tutte le scelte letterali archiviate.

Audit V1 espande D[:,selected] in tensor token×count×2048, producendo circa **3.014.385.991.680 byte di working tensor payload cumulativo**; non è traffico PMU o costo batch1. Time guard fallisce1800,141s; piccoOS2.985.328.640B. Risultato producer preservato.

V2 cambia **layout dell'audit**, non scienza/soglie/dati: projection sum per blocchi256colonne source, evitando weights espansi per token. Tutte720predizioni/144controlli/15curve/domains e ogniID sono verificate,509,563s/OS1.549.312.000B. MaxF32pred2,663e-8, remainder3,985e-16, identity6,682e-16. Score absolute Frobenius difference1,116e-9 rientra nel gate `||delta_score|| ≤1e-10||literal_score||+1e-24`; non chiamare1e-9 l'errore relativo.

Audit SHA `d447153780666625277e48183d8d14c5df0568058d54101cfff226a143c9b0e0`; closure SHA `fef3d33eaaac7af90a1e40b8330c6882608c00ab302b01a5b0046270f7c7d4ce`. Shared BLAS/reader/BF e regola literal espliciti: indipendenza dichiarata, non “nessun codice comune”.

### 13.3 Sketch PCA FIT-only: ID scelti senza full source, ma candidato fallisce

`FFN_SKETCH_ROUTER_PROTOCOL_20261013.md`, `FFN_SKETCH_ROUTER_RESULT_20261014.md`; codice `ffn_sketch_router.py`, auditor/closer. Freeze reale **`5be0cdae9b80e23f8fdc866dfb0727bc940c82d8`**, binding corretto v2 `592a77d`, chiusura environment finale `677203b`.

PCA del secondo momento **non centrato** FIT di δ=x−μ, rank64/128/256/512. U_projected=U P512, G_projected=αG P512; il selettore riceve δ, fattori projected, center constants e col-energy, **nessuna attivazione originale completa**. Le IDs sono congelate prima del reference evaluator full source che applica la correzione esatta scelta.

Count fissati dall'oracle512/2048/32, non riscelti su DEV. Risultati:

| Site | Scelta FIT | Risultato |
| --- | --- | --- |
|0|Nessun rank≤512 passa FIT|rank512 FIT0,10713144/0,144444; DEV0,16943819/0,234448|
|12|Nessun rank≤512 passa FIT|rank512 FIT0,13893633/0,159494; DEV0,22480053/0,282644|
|23|FIT sceglie64|FIT0,09979830/0,196467; DEV0,11000081/0,178872, FAIL|

Site23 rank256/512 hanno DEVmean0,09947464/0,09266366, ma non sono scelti con DEV. Nessun candidato ammesso. Questo respinge la PCA d'energia input nella ricetta provata, non ogni selettore low-rank/nonlineare.

Producer598,453s/OS1.650.401.280B; audit originale611,594s/OS1.755.299.840B, tutti576prediction/144controlli/3modelli PASS. E579file/3.103.272.156B; payload3.102.686.640B. Spectral basis archiviata qualificata; un LAPACK diverso non deve essere bit-identico nel basis per parlare di stesso subspace, ma una riproduzione literal dello stesso artefatto deve usare i fattori salvati.

Result SHA `cb9fdda4e777220d087273332e90b7d38e07fb2ede6db31413f17f5a237e9299`; audit SHA `33d5971f25caffc212db9474e0092b0e1d7f6b5a33273e35e6cd3d6e4f0c3b2e`. MaxF32pred2,655e-8; moment/eigen witnesses e coefpass. Non è un kernel C, non ci sono measured speed/DRAM del selettore.

### 13.4 Source-active balanced geometry: ultimo producer, audit da completare

Si cambia la variabile motivata dalla PCA failure: **quali direzioni di input contano per la variazione del remainder**, invece di quanta energia δ hanno. `FFN_SOURCE_ACTIVE_METRIC_NEXT_20261014.md` e protocol attuale spiegano il surrogato. Nessun rank extra, count extra, cambio gate o selezione DEV.

Con s=SiLU e g=αGx:

```text
grad r_i = a_i U_i + alpha b_i G_i
a_i = s(g_i) - s(gc_i)
b_i = u_i*s'(g_i) - uc_i*s'(gc_i)
w_i = ||beta D[:,i]||^2
M = E_FIT sum_i w_i * (grad r_i)^T*(grad r_i)

M = U^T diag(w E[a^2]) U
    + alpha^2 G^T diag(w E[b^2]) G
    + alpha {U^T diag(w E[ab]) G + transpose}
```

M è PSD in aritmetica esatta e viene accumulata senza allocare gradiente per ogni token. È un surrogato per variazione del remainder, **non gradiente della score**, non obiettivo top-K ottimo e non include covarianza output cross-neurone.

Σ=E_FIT δδᵀ non centrato; media e centered covariance salvate per verificarne l'identità. Cutoff eigenΣ relativo1e-12, solo supporto trattenuto. Balanced T=Σ^(1/2) M Σ^(1/2), top eigenvectors V. Encoder E=Σ^(-1/2)V, decoder B=Σ^(1/2)V, EᵀB=I. δ_hat=δ E Bᵀ. Questi fattori sono **biortogonali**, non una proiezione ortogonale Euclidea.

Uproj=UB, Gproj=αGB. ID function standalone usa solo δ,E,projections/center constants/col-energy. L'originale source full viene calcolato solo **dopo** la scelta IDs per il reference evaluator. Lo stato di sviluppo soddisfa questa separazione di informazione; il costo dell'intera prediction include ancora full affine/reference, quindi non è la velocità del target.

Controlli congelati:

- prime history FIT,16 righe uniformemente selezionate, RNG0 e4 direction action;
- central FD1e-4 a centro fisso, errore normalizzato con norma del **derivato source feature completo**, non con remainder derivativo magari zero, gate1e-5;
-4 gradient Gram diretti vsM action, gate1e-10;
-coefficients, PSD, symmetry, spectrum, square roots, duality, moment/covariance e source witness gate1e-10;
-all576 F32prediction gate1e-6, metadata1e-9;144 source proxy1% controlli;
-before/after hash di input e output; budget CPU6, audit1800s, OS8GiB, output4GiB.

Geometria producer:

| Site | Supporto Σ | Drop | Condition Σ sul supporto | Max FD relativo |
| --- | ---: | ---: | ---: | ---: |
|0|2048|0|164141057,42441052|3,1116919211e-11|
|12|2048|0|4023381,075977966|2,3351247765e-11|
|23|2048|0|2715582669,1792827|9,4910799080e-11|

Fattori con inverse-root molto condizionata sono un rischio di precisione/physical deployment da misurare. Il buon F64/control non certifica F32, AQ o ternari in C. Non eliminare dimensioni per stabilizzare dopo aver osservato DEV senza nuovo protocollo FIT.

**Curve del producer ancora non completamente adjudicate:**

| Site | Rank | FIT mean /worst | DEV mean /worst |
| --- | ---: | --- | --- |
|0|64|0,342514 /0,447844|0,388192 /0,535542|
|0|128|0,238390 /0,339245|0,312970 /0,447219|
|0|256|0,147885 /0,196544|0,223796 /0,319098|
|0|512|**0,098412530 /0,127617496**|**0,137283700 /0,178026976**|
|12|64|0,284836 /0,316598|0,376078 /0,466854|
|12|128|0,229290 /0,258849|0,316667 /0,410778|
|12|256|0,175611 /0,198730|0,252030 /0,310319|
|12|512|0,124525 /0,140929|0,186774 /0,239840|
|23|64|**0,095835675 /0,187636042**|**0,105606825 /0,172211771**|
|23|128|0,087125 /0,170017|**0,100026053** /0,166579|
|23|256|0,078786 /0,146916|0,093889 /0,157830|
|23|512|0,072619 /0,137468|0,086271 /0,147959|

Site0 FIT sceglie512, DEVmean fallisce. Site12 nessunFITrank ammesso. Site23 FIT sceglie64, DEVmean fallisce.128 DEV è0,100026053, **non arrotondare a0,10 per passarlo**.256/512 locali passano ma non sono il candidato FIT scelto. Perciò nessuna selezione ammessa ai tre site, anche se il miglioramento su PCA è reale nelle osservazioni producer.

Famiglia E `ffn_active_router_20261014`:579file/3.460.561.182B. Payload3.459.961.776B:modelli772.720.128B +predizioni/IDs2.687.241.648B. GPU_calls/source_model_forwards/optimizer_updates/RESERVED_queries=0. I dati source sono letti dai coefficienti e operand archive, non query di un nuovo modello.

Il senso della curva è un progresso nella variabile giusta, con fallimento del candidato scelto. Non è ragione sufficiente per continuare automaticamente rank1024, altri centri o più modelli. Prima chiudere l'audit, poi valutare il budget24-layer e l'ostacolo fisico complessivo.

## 14. Custodia e fallimenti meccanici: cose da non “ripulire”

### 14.1 Typo del freeze PCA

La prima prep PCA indicava erroneamente freeze `6b70810`; in realtà il codice congelato era `5be0cda`. Prep originale PID32060,19,969s/exit0 conservata. Correzione **prima di fitting**, nuovo prep PID21896,19,156s/exit0, binding v2 2.322.842B SHA `6f616c231e12faa1fd7d210f31a2daea3e56c9fd757526ba382237b813810154`.

`ffn_sketch_router_revision_correction_20261013.json` conserva entrambi; config e pin identici, cambia solo il freeze. Non rinominare il vecchio errore come se fosse sempre stato corretto.

### 14.2 Duplicate audit PCA respinto

L'audit originale era già completo. Una tool chain ha letto file esistenti e poi lanciato un audit incondizionatamente, senza una decisione intermedia. Tentativo duplicate PID20728/session33133, creazione osservata01:27:40 locale, exit1. La guardia sull'esistenza del risultato lo respinge prima dei calcoli; il `finally` tenta una terminal write exclusive e fallisce anch'esso sul file già presente.

Il helper `write()` apre con modalità **`'x'`**, quindi non sovrascrive. La ricevuta originale success è preservata anche come `ffn_sketch_router_adjudication_20261013.completed_terminal.json`, byte-identica, SHA `bec93e479c1a92fc0b10d8509611cdddfef539ae9de0dc3abdfbfc19b5c590b6`. Il duplicate è registrato in `ffn_sketch_router_duplicate_attempt_20261014.json`, come osservazione, non come falso worker receipt numerico.

**Regola operativa:** separare read/check dal launch; il launch deve essere dentro la condizione che ha verificato assenza file e assenza worker. Una lista di tool sequential non diventa condizionale solo perché il primo legge lo stato.

### 14.3 Drift NVIDIA del14 ottobre

La closure stretta PCA fallisce su `nvidia-smi.exe`, PID6380,1,828s/exit1; receipt conservata. Inventario completo attuale8044pin,19.704.477.495byte dichiarati,727,532s di cold IO: **esattamente quattro differenze**, tutte componenti NVIDIA non usate dagli algoritmi CPU di quel producer/audit.

| File `C:\Windows\System32` | Bytes prima→dopo | SHA256 prima | SHA256 attuale al confronto |
| --- | --- | --- | --- |
|nvidia-smi.exe|1682152→1682664|bfa7baa612540e4c8ad2f84501b15bf6614f7d428e9af630f2a517686dadefa2|a070678c20a19bb16c9fe102e69e7cb26de16ef9e1582f36ed6bd91fd9489a96|
|nvcuda.dll|4714728→4714728|b53d1e3e49b2d1bc33480243017af6847e4809b7acdd7c121ea294f05afe6fd3|83cb61c63706ead0c2938aa62f3c9abe6098362ec3fd2425b77b2b178d4ae052|
|nvml.dll|1408232→1408744|21e72408146f394bf2848bd52d147d5f9f9fb0f160dfeb6bee35d0967c419198|df44b5c207f289db4ac9ae4b9a1754c5a9901b2f7d01b160830cd29c10a28308|
|nvapi64.dll|572368→572368|5b8d83a4732c62f460605b1cf88dbdbda08f08e790fb5eb9051b5bd2684da4c7|6863bf52f439b9fb40fc0c279bad439d9b4a487c18e2a5590744fa4e13160c7d|

AST dei moduli CPU non importa Torch/Transformers/CuPy/subprocess né chiama NVCLI; reader/hash non esegue i file NVIDIA. Tutte altre pin di codice, Python, NumPy, psutil, dati, source e output sono esatte. Non si è reinstallato il vecchio driver.

`ffn_sketch_router_close_environment.py` permette **solo quei quattro percorsi**, conserva descriptor prima/dopo e failed original closure, ri-verifica gli altri pin/artefatti/protected e chiude lo scope. Closure environment JSON8489B, SHA `4d3a93b0246bee904d6d886905fc91bce6b7aadc1441225a40b176908bdee3a3`; PID5756,23,453s/exit0.

Decisione `HISTORICAL_CPU_SKETCH_AUDIT_QUALIFIED_WITH_EXPLICIT_UNUSED_TOOL_DRIFT`. Flag `all_historical_pins_current_exact=false`; `all_CPU_numerical_and_artifact_pins_current_exact=true`. Non è un bypass indiscriminato dell'audit. **Nessuna prova di continuità GPU attuale**; un prossimo job GPU necessita nuovo binding/preflight reali. Source-active binding conserva questa closure storica e aggiorna esplicitamente le quattro pin.

### 14.4 Hash protetti e campo di validità

Valori protetti storicamente congelati:

| File | SHA256 |
| --- | --- |
|`engine.c` e `benchmarks/phase60/engine.c`|5f948fc0dcd28b1647a2a2c73067bdc0a76a7d41ae7f34395ede8120aeaa83ce|
|`benchmarks/donor_adaptation/configs/_manifest.json`|fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d|
|`benchmarks/donor_adaptation/density/build_document_holdout.py`|c5c9ed40864989592664c15790e69e41a5ebbad73d00ee7ac067d98e2baf15e9|
|`docs/research/RESEARCH_INDEX.md`|99b5b11c865d000f387c17120b918458aa8cac8263628d7d040abddcd7eb5273|
|`nibble_pack.h` descriptor nel binding|915ffd79dd1db0b2d3dc35fb79bfb7ae41da44dbbc062daf2c31f3f9e0001d23|

Queste sono le identità dei descriptor di custodia, non un ordine di ripristinare file dell'utente se li modifica successivamente. In handoff non si rilegge l'intera famiglia28GB o87GB solo per ricopiare numeri; le closure complete già fatte rimangono le fonti. Per una nuova run seguire le pin previste, e distinguere filesystem corrente da hash storici.

## 15. Letteratura già studiata e limiti dell'attribuzione

Non serve ripetere una ricerca web generica prima di ogni esperimento. La ricerca primaria è in `STRATEGIC_REVIEW_20261004.md`, `LITERATURE_CONVERSION_FOLLOWUP_20261010.md`, `CONDITIONAL_REDUNDANCY_LITERATURE_NOTE_20261013.md`, più le note nel ledger. In questo handoff si riportano le conclusioni di quelle letture: non è stata svolta una nuova verifica dei paper o una replication.

| Linea | Cosa offre alla ricerca | Cosa non dimostra per noi |
| --- | --- | --- |
|MOHAWK /Phi-Mamba|Allineamento mixing, blocchi e distillazione staged con operatori circostanti preservati|Conversione simultanea2048→256,24→6layer, SwiGLU→dReLU, topK e ternari a costo basso|
|Mamba in the Llama|Riutilizzo proiezioni attention/FFN e distillazione chatbot|Il nostro budget locale o uno student minuscolo senza training sostanziale|
|Sparse/Granular Upcycling|Copia/partizione source con coverage e scale d'inizializzazione, continued training|Funzione conservata con shard arbitrari e routing normalizzato non corretto|
|Net2WiderNet|Duplicazione function-preserving sotto ipotesi elementwise, divisione outgoing per copie|Specializzazione utile immediata o accelerazione a topK su qualunque copia|
|PEER/product keys|Punteggi fattorizzati per tanti piccoli esperti; scelta efficiente nella famiglia vincolata|Adattamento automatico del router donor non fattorizzato, masse corrette e DRAM CPU costante|
|SliceGPT|Distinzione tra rotazione e cancellazione di dimensioni, invarianti compatibili|Che una proiezione rettangolare arbitraria con gamma diagonale preservi il nostro chatbot|
|Quantizzazione SSM /ternary training|Sensibilità di operatori e necessità di recovery/precisioni differenziate|Trasporto esatto del pretrained nel vecchio core o velocità LUT universale|

`LITERATURE_CONVERSION_FOLLOWUP` distingue errori locali su input teacher da composizione su input student. Il bound telescopico `||F_student-F_teacher|| ≤Σ_i ε_i Π_(j>i)L_j` richiede coordinate compatibili, veri bound e Lipschitz lungo gli stati incontrati; qui non sono stati certificati. Una MSE media non è un bound uniforme.

Due fonti recenti sono state lette **solo a livello di abstract primario**:

- Expert Upcycling, arXiv2604.19835v2: duplicazione di esperti/router, topK fisso, continued pretraining rompe simmetria; abstract riferisce7B–13B. Non è nostra replica o prova CPU.
- DivMoE, arXiv2610.11317v1: problema della granular single-source slicing, composizione di esperti specializzati di dominio, routing a gruppi distinti e training ulteriore. Full metodologia non ancora studiata; non è teorema zero-training.

I numeri dei paper restano risultati degli autori. Il corpus locale, il codice C e la qualità del nostro artefatto devono essere misurati separatamente. Se una decisione dipende da un dettaglio non letto, aprire la fonte primaria e registrare livello di evidenza; non attribuire alla letteratura una garanzia che non contiene.

## 16. Piano di ripresa concreto, con decisioni e gate

### 16.1 Priorità A — chiudere l'ultimo risultato

**Incertezza:** il producer source-active e tutte le576 predizioni corrispondono alla matematica dichiarata? **Reuse:** binding, source coefficients, capture all24,64celle e oracle già qualificati. **Decisione:** se PASS, classificare miglioramento geometrico ma fallimento del candidato FIT; se fault, isolare il difetto meccanico senza scartare la curva. **Costo:** audit6thread≤1800s/8GiB, nienteGPU/RESERVED. Closure dopoPASS, poi documentazione.

Non ripartire dal producer perché è fermo il terminale audit. Non fare curve nuove durante un audit incompleto. Non saltare site23 o gli hash finali per risparmiare tempo: il Fail-Fast è stato revocato.

### 16.2 Priorità B — il budget reale del nucleo24-layer

Proposta già scritta ma non congelata/eseguita: `NATIVE_24_LAYER_COST_SCREEN_NEXT_20261014.md`. È un file untracked della ricerca corrente: conservarlo e non scambiarlo per una prova.

**Domanda:** quanto dei20ms/token per useful50 consuma l'implementazione ricorrente che ha effettivamente preservato il chatbot noto? Il branch iniziale~1,013ms non misura tutti24 e non giustifica né ottimismo né impossibilità.

Protocollo da implementare/congelare prima di osservare tempi:

1. DLL `row_state` qualificata e tutti24 canonical parameter packets.
2. Set congelato di storie e posizioni reali FIT/DEV, corte/lunghe, dalle tracce own-history qualificate.
3. Fresh native contexts; generare prefix states con le vere righe, **mai caricare stato teacher per lanciare il timing**.
4. Una continuazione token singolo per layer,24 chiamate **sequenziali**; verificare output integer/raw/BF/stato/ring sulle continuation archiviate.
5. Non parallelizzare i layer sfruttando righe già precalcolate: nel vero decoder esiste dipendenza. Threading intra-layer consentito solo se dichiarato e verificato.
6. Report distribuzione whole24 e per-layer, parametri fisici/addresses/footprint, warm e source-evicted dichiarati, condizioni macchina e overhead del wrapper.
7. Il benchmark fornisce righe preconv già note **per isolare il costo ricorrente**: esclude generatori contestuali, FFN, attenzione, head, tokenizer e trasporto. Non chiamarlo end-to-endTPS.
8. Misurare traffico reale se contatori accessibili; altrimenti riportare exact operand ledger e lasciare DRAM **unresolved**. CLFLUSH non è un contatore.
9. Non concorrere con altri job pesanti della ricerca durante timing; preservare processi dell'utente e registrare sharing residuo.
10. Freeze benchmark/audit/binding/posizioni/budget; solo dopo lancio.

**Decisione:** se la ricorrenza sola consuma≥20ms nella configurazione misurata, occorre una vera modifica della implementazione/core prima di espandere banche FFN o donor. Non è teorema di impossibilità del goal. Se resta budget, assegnarlo esplicitamente a generatori/FFN/route/attention/head e precisione, sullo stesso artefatto.

### 16.3 Priorità C — un artefatto FFN realmente eseguibile, se il costo lo giustifica

Non basta diminuire l'errore di una score offline. Un nuovo protocollo dovrebbe verificare:

- funzione affine full-output **fisica**: matrix/factors, spazio, quantizzazione, decode e costo;
- costo di cell selection, projected feature, topK IDs e masse; sort completo4608 non gratuito;
- source-active inverse-root precisione, especially condition≈2,716e9 site23;
- read selected input row U/G e output columns D con layout contiguo/ragged dichiarato;
- correzione combined remainder e cross-neuron interaction, sourceBF control;
- spent bytes perselected expert, shared bytes, pagine/working set e DRAM osservata;
- effettivo vantaggio rispetto al source FFN senza creditare full original projection usata per reference;
- composizione del candidato nel decoder su proprie storie, prima di estenderlo a24site;
- qualità/task/DEV e costo sullo **stesso** artifact.

Possibili nuove variabili, ancora **ipotesi**, da scegliere solo con un difetto motivato: obiettivo direttamente topK/score-weighted o output Fisher; metric cross-neuron; residual encoder nonlineare o condizionale; coverage/celle con supporto FIT insufficiente; rumore/calibrazione robusta; banking joint tra affine e remainder; training che recupera perturbazioni compositive. Nessuna è stata dimostrata dalla curva attuale.

Un risultato0,100026 non autorizza “arrotondiamo e va bene”. Un rank maggiore scelto dopoDEV è nuova ricerca, non riparazione dello stesso protocollo. Se cambia regola di selezione, fissare su FIT criterio motivato e usare nuovi dati per la verifica finale.

### 16.4 Priorità D — ridurre core/generatori senza perdere la causalità

Se il backbone wide domina, esplorare channel/layer/shared-core reduction, generatori condizionali o coordinate sufficienti. L'invariante è donor-derived parameter, own state e qualità. Non tornare a un random core solo perché veloce.

Separare le operazioni modificate: encoder/residual width, recurrence, layer merging, norm, FFN activation, selector, precision, decoder. Reuse bridge full-width come controllo. Un staged intervention su un solo site/blocco può attribuire il difetto, ma deve poi comporsi. I lower bound global-linear già qualificati impediscono di giustificare un'altra grid identica di tokenmean +lowrank.

### 16.5 Priorità E — allCPU, n utile e generalità

Quando un primo percorso qualitativamente utile e a budget emerge:

1. Eliminare la dipendenza GPU dei componenti che appartengono all'inferenza finale: contextual generators, FFN, attention/SWA e head, con nuovi ABI/export/audit dichiarati.
2. Congelare il medesimo artifact; own-history, task, stop, cache/reset, continuation e local-to-whole gates.
3. Run end-to-end CPU batch1 di almeno50 **token utili accettati/s**, accounting di prompt/prefill/decode, warm/cold/context/hardware e acceptance se speculative. Il denominatore deve contenere il lavoro reale necessario, non solo il kernel selezionato.
4. Usare RESERVED solo dopo scelte e protocollo congelati, documentando che non hanno guidato tuning.
5. Crescere n con **funzioni distinte** derivanti da capacità donor, misura ablation/necessità/causalità/quality, routing IDs+masses e DRAM. Repliche identiche sono storage ridondante e non capacità nuova.
6. Verificare altri donor/famiglie, poi10B e100B quando risorse vere disponibili; metadata/header di80B non sono valori/conversione.
7. Documentare criteri di applicabilità e varianti: source architecture, operator compatibility, rank/support/precision, dati/budget di recovery e failure mode.
8. Contabilizzare costo di adattamento: master/grad/optimizer/CPU/GPU/storage/tempo, incluso tutto ciò che cresce con n. Compressione di delta non equivale a ottimizzatore O(1).

### 16.6 Stop delle ripetizioni improduttive e rivalutazione

Ogni nuova attività deve dire: quale incertezza, quale evidenza riusa, quale decisione cambia, costo e arresto. La documentazione non deve diventare un obiettivo separato che assorbe tutta la ricerca.

Un problema operativo osservato è stato il prolungamento eccessivo del ragionamento testuale, con speculazione ripetitiva e mancata osservazione tempestiva di job già terminati. Il correttivo è concreto: ragionamento breve collegato a un controllo o modifica, strumenti, risultato, decisione; nessuna attesa riempita da pagine di ipotesi. Per run di ore usare la pausa richiesta dal proprietario. Questo preserva il rigore: gli audit continuano a essere completi e i fallimenti restano visibili.

## 17. Mappa artefatti, storage e commit da usare

Base codice `D:/_THINGS/Progetti/SiliconLLM/benchmarks/native_expert_scaling`. Base docs `D:/_THINGS/Progetti/SiliconLLM/docs/research/NATIVE_EXPERT_SCALING_20260925`. Base risultati grandi `E:/SiliconLLM/results/native_expert_scaling`. Source e corpus piccoli/referenziati possono restare sotto `results/native_expert_scaling` nel repository.

Le dimensioni sono byte dei file della famiglia indicata, storicamente controllati dal report/closure. Non sommarle come uso libero attuale del disco e non assumere che ogni variante sia un unico model deployabile.

| Famiglia E | File /bytes noti | Ruolo e stato |
| --- | --- | --- |
|`full_width_native_chat_20261013`|4805 /17235894707|Bridge known-corpus completo; quality/numeric PASS, nonCPU finale|
|`ffn_operand_capture_20261013`|8163 /28079135477|Operandi actual all24, audit/closure completi|
|`row_state_20261013`|2131959652|All24 ABI/packets qualificati|
|`token_state_20261013`|1763186800|Primo layer token own-state/COST componente|
|`token_generator_compiler_20261013`|1449937450|Tutto F0/H0 source-only|
|`integer_readout_20261013`|2909894600|Backend I16 qualificato; usare path esatto del descriptor nel report|
|`exact_integer_lut_20261013`|2526781296|Backend byteLUT preciso ma più lento; descriptor authoritative|
|`ffn_affine_pilot_20261013`|441 /1711100700|AffineFITfail, numeric e defectauditPASS|
|`ffn_exact_subset_20261013`|723 /2785911606|FixedsubsetFIT/DEVfail, V2 auditPASS|
|`ffn_exact_subset_rank_diagnostic_20261013`|7080504|Array del fault rankingzero, trattenuti|
|`ffn_remainder_oracle_20261013`|720 /3143325180|Oraclelocalpass, auditV2PASS, no cheaprouter|
|`ffn_sketch_router_20261013`|579 /3103272156|PCA candidatefail, completeaudit/environmentclosure|
|`ffn_active_router_20261014`|579 /3460561182|Producerdone, auditinterrotto/nonconcluso|

Alcuni nomi familiari nella tabella riassumono l'esperimento; per path effettivo di DLL/output usare sempre `path` assoluto nei descriptor JSON. Non creare directory nuove con lo stesso nome per “sistemare” un path presunto.

### 17.1 Commit recenti che delimitano scienza e custodia

| Commit | Contenuto |
| --- | --- |
|`b34cd94`|Freeze source-gradient metric/balanced selector/audit/close|
|`58c33ec`|Binding current runtime/input prima dei numeri source-active|
|`bd88f5e`|Stato della run source-active; ora live text superato|
|`677203b`|PCA result e closure environment esplicita|
|`17f55d1`|Source/custody reconciliation environment|
|`5be0cda`|Freeze PCA source-linear selector reale|
|`592a77d`|Binding PCA corretto prima del fitting|
|`8724aef`|Oracle completo/auditV2/closure|
|`6a15e65`|Layout audit oracle V2 blocchi256|
|`90c3586`|Fixedsubset risultato completo con zero-score repairaudit|
|`a99c170`|Fixedsubset V2 literal replay +algebra indipendente|
|`91b540c`|Affine defect fullaudit/closure|
|`b22ee53`|Freeze affine pilot e qualified FFN source corpus|
|`9448d9b`|Whole bridge full-width risultato completo|
|`32b1c5d`|All24 contextual-row export/ABI completo|
|`0f8f21d`|Token own-state branch completo|
|`13a29c8`|Source-only token compiler completo|
|`8d2e0c5`|I16 readout completeV3|
|`7d3b200`|Exact byteLUT costo/qualifica completa|
|`2f8d2d3`|Tokencontext global-linear rank bound e report|

Per cronologie anteriori usare git log e i documenti di risultato, non indovinare il ruolo di un commit dal suo numero. Non fare checkout distruttivi in questa working tree sporca. Il branch donor indicato dall'utente può avere spelling storico diverso: verificarlo con `git branch -a` prima di usarlo come ref.

## 18. Comandi di ripresa e controllo, senza launch implicito

Eseguire da `D:\_THINGS\Progetti\SiliconLLM`, PowerShell. Questi sono comandi operativi; il blocco audit è da usare **quando si riprende la ricerca**, dopo lettura dello stato goal e verifica che un altro worker non sia partito. Non è stato lanciato dalla consegna di questo documento.

### 18.1 Stato piccolo e veloce

```powershell
Set-Location -LiteralPath 'D:\_THINGS\Progetti\SiliconLLM'
git branch --show-current
git log -6 --oneline
git status --porcelain -uno
$researchDocs = 'D:\_THINGS\Progetti\SiliconLLM\docs\research\NATIVE_EXPERT_SCALING_20260925'
Get-Content -LiteralPath "$researchDocs\INDEX.md" -TotalCount 160
Get-Content -LiteralPath "$researchDocs\ffn_active_router_result_20261014.terminal.json"
Get-Item -LiteralPath "$researchDocs\ffn_active_router_adjudication_20261014.json" -ErrorAction SilentlyContinue
Get-Item -LiteralPath "$researchDocs\ffn_active_router_adjudication_20261014.terminal.json" -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match 'ffn_active_router_audit.py' } |
    Select-Object ProcessId, CreationDate, CommandLine
```

Se la ricerca del processo mostra il comando di verifica stesso, distinguerlo dal Python worker. Non usare una regex larga per decidere che esiste un worker vivo. Il prossimo controllo deve confrontare eseguibile e command line reali.

### 18.2 Audit completo sull'esistente producer

```powershell
$researchRoot = 'D:\_THINGS\Progetti\SiliconLLM'
$researchPython = 'C:\Users\giosa\AppData\Local\Programs\Python\Python312\python.exe'
$researchDocs = Join-Path $researchRoot 'docs\research\NATIVE_EXPERT_SCALING_20260925'
$auditResultPath = Join-Path $researchDocs 'ffn_active_router_adjudication_20261014.json'
$auditTerminalPath = Join-Path $researchDocs 'ffn_active_router_adjudication_20261014.terminal.json'
if ((Test-Path -LiteralPath $auditResultPath) -or (Test-Path -LiteralPath $auditTerminalPath)) {
    throw 'Audit output or receipt already exists: inspect it before any launch.'
}
$auditWorkers = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and
    $_.CommandLine -match 'benchmarks[/\\]native_expert_scaling[/\\]ffn_active_router_audit.py'
}
if ($auditWorkers) { throw 'An audit worker is already present: observe it.' }
Set-Location -LiteralPath $researchRoot
& $researchPython -I -S -B -X utf8 'benchmarks/native_expert_scaling/ffn_active_router_audit.py'
if ($LASTEXITCODE -ne 0) { throw 'Audit failed: preserve and inspect its receipt.' }
```

Questo script è una guardia locale, non un lock globale contro un altro launch simultaneo. Il programma protegge gli output con exclusive write. Usare il terminal tool per avere la sessione e osservarla; prima del lancio registrare PID/create/command e log se il launcher lo prevede. Non terminare job estranei.

Se result o terminal sono presenti ma non qualificati, il codice non può essere rilanciato in-place perché write è exclusive: non cancellarli. Conservare il difetto; progettare repair/v2 e binding espliciti con output nuovi come nei casi precedenti.

### 18.3 Closure solo dopo audit PASS

```powershell
$researchDocs = 'D:\_THINGS\Progetti\SiliconLLM\docs\research\NATIVE_EXPERT_SCALING_20260925'
$auditPath = Join-Path $researchDocs 'ffn_active_router_adjudication_20261014.json'
$receiptPath = Join-Path $researchDocs 'ffn_active_router_adjudication_20261014.terminal.json'
$closurePath = Join-Path $researchDocs 'ffn_active_router_session_closure_20261014.json'
$auditObject = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
$receiptObject = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
if ($auditObject.decision -ne 'INDEPENDENT_STORED_PASS' -or
    -not $auditObject.all_numeric_pass -or
    $auditObject.checked_predictions -ne 576 -or
    $receiptObject.exit_code -ne 0) { throw 'Full audit is not qualified.' }
if (Test-Path -LiteralPath $closurePath) { throw 'Closure exists: inspect it.' }
Set-Location -LiteralPath 'D:\_THINGS\Progetti\SiliconLLM'
& 'C:\Users\giosa\AppData\Local\Programs\Python\Python312\python.exe' -I -S -B -X utf8 'benchmarks/native_expert_scaling/ffn_active_router_close.py'
if ($LASTEXITCODE -ne 0) { throw 'Closure failed: preserve and inspect its receipt.' }
```

Il closer verifica hash, tutti579 output, file family/bytes e identità dei processi; il primo check PowerShell non sostituisce questi controlli. La closure scrive nel JSON `goal_status=ACTIVE_INCOMPLETE` come campo di scope scientifico; ciò non modifica automaticamente lo stato paused dello strumento goal.

### 18.4 Git scoped se è utile fissare il risultato

```powershell
$researchPaths = @('HANDOFF.md')
git update-index --add -- $researchPaths
if (-not $?) { throw 'Scoped index update failed.' }
git diff --cached --check -- $researchPaths
if (-not $?) { throw 'Scoped diff check failed.' }
git -c commit.gpgSign=false commit --only -m 'Document research state and exact continuation plan' -- $researchPaths
if (-not $?) { throw 'Scoped commit failed.' }
```

Per snapshot o nuovi risultati aggiungere **esplicitamente** i loro path; controllare l'identità Git configurata. Nessun `git add .`, nessun reset, nessun clean, nessun push/PR implicito. Non aggiungere coauthor/trailer. I comandi non rappresentano commit già creati in questa consegna.

## 19. Errori di interpretazione da evitare

| Errore frequente | Lettura corretta |
| --- | --- |
|“Il motore faceva848TPS, quindi Falcon può fare848TPS”|Workload/operatori/model diversi; riprezzare stessoartifact|
|“Il bridge passa14/16, quindi il converter è finito”|SSMCPU composto utile, FFN/attention/headGPU e wide generators ancora attivi|
|“Il router restituisce ID plausibili”|Serve la scelta corretta, masse corrette, qualità composita e costo fisico|
|“Tanti parametri in RAM = tanta conoscenza”|Dimostrare funzioni distinte/necessità, replication e random weights non bastano|
|“PCA manteneva energia, dunque informazione utile”|Head/Fisher/operator sensitivity e conditional information sono differenti|
|“AuditPASS implica qualitàPASS”|Molti risultati perfettamente riprodotti sono scientificamente falliti|
|“Audit parziale384/576 è abbastanza”|Manca site23, hashfinali e ricevuta; non è qualificato|
|“Il processo è sparito: è fallito1”|Senza worker receipt non si conosce exitcode/rootcause|
|“Il17GB trace è un model17GB”|È famiglia di dati/audit, distinta dall'artefatto di inferenza|
|“Packed9MB implica9MB letti per token”|Multipass/lookup/indice/cache possono cambiare logical eDRAM|
|“Evicted uguale DRAMmisurata”|È condizione iniziale cache; trafficoeffettivo unresolved senza misura|
|“Il source-active FD è piccolo, quindi ternario andrà bene”|F64smooth control non certifica inverse-rootquantized/Cphysical|
|“Rank256 site23 passaDEV: promuoviamolo”|ProtocolloFITsceglie64; cambiare selezione dopoDEV richiede nuova ricerca|
|“0,100026 è10%”|È sopra0,10; arrotondamento display non cambia eligibility|
|“Gli hash attuali sistemano custodia storica incompleta”|Sono prove attuali; gap storico resta dichiarato|
|“NVIDIA cambiato: ignoriamo ogni hash”|Soloquattrofile non eseguiti, scopeCPUesplicito; GPUcontinuitànonprovata|
|“Hashfallisce: reinstalliamo il vecchio driver”|Preservare ambienteutente e classificare drift; nonripristinosilenzioso|
|“Luna può monitorare le ore”|Istruzione revocata: nessunsubagente, comunicare epausare|
|“Fail-fast ferma gli audit0/16”|Regola revocata esplicitamente; auditcompleto necessario|
|“Si può rifare sourcecapture per sicurezza”|Reusequalificato; nuovaquerycostosa solo se incertezza concreta|

## 20. Come rendere il prossimo passo produttivo

Il risultato migliore raggiunto non è una nuova percentuale di compressione: è una catena che parte da un chatbot vero, usa i suoi parametri, esegue ricorrenze C con stato proprio e conserva comportamento sul corpus noto. Questa catena dà un controllo da cui togliere costo senza perdere la funzione.

Il difetto attuale FFN è localizzato: gli affine-cell perdono soprattutto interazioni nonlineari; la scelta per-token vera conserva gran parte del remainder con count differenti ai tre site; due encoder economici provati non danno una selezione FIT→DEV ammessa. Il source-active migliora la PCA ma non supera i gate del candidato scelto. Serve decidere la prossima trasformazione in rapporto al costo whole, non migliorare indefinitamente un proxy.

Il nuovo modello deve poter rispondere prima di una run: **quale funzione stiamo preservando, quale informazione ha il selettore, quali byte e operatori vengono davvero eseguiti, quale criterio era fissato prima dei risultati, e quale gate del goal cambierà**. Se la risposta manca, fermare il lancio e formulare un protocollo concreto.

Il punto di ripresa è quindi: **audit source-active completo sull'esistente producer → closure/report → cost screen24-layer qualificato → scelta motivata della riduzione o del componente condizionale → stessa pipeline con qualità e ≥50 usefulTPS → scaling n utile e generalità**. Tutto ciò che segue il producer source-active resta da fare secondo i limiti precisati sopra.

## 21. Materiale di continuità preservato con questo handoff

Directory `docs/research/NATIVE_EXPERT_SCALING_20260925/HANDOFF_SNAPSHOT_20261014`:

| File | Contenuto e utilizzo |
| --- | --- |
|`HANDOFF_PREVIOUS_20260730.md`|Vecchio handoff integrale83.691B: fasi originali e sandbox; storico, non prossimo passo attuale|
|`goal-objective.md`|Obiettivo completo8.385B, copia byte-identica dell'attachment corretto|
|`INDEX.md`|Indice al momento precedente alla riconciliazione handoff; preserva il vecchio testo live|
|`METHOD.md`|Procedimento46.318B nella versione di ingresso alla consegna|
|`PRIOR_EVIDENCE.md`|Ledger174.430B: dettagli delle linee anteriori e collegamenti a report/protocolli|
|`evidence_inventory.json`|Inventario con path relativi, byte eSHA256 dei documenti di ricerca e JSON di result/adjudication/closure/terminal/admission selezionati; HEAD e timestamp di acquisizione|
|`active_router_interruption_observation.json`|Osservazione dello stato audit, con limite esplicito: nessun exitcode né causa del worker inventati|
|`handoff_document_verification.json`|Verifica della documentazione: lunghezza, riferimenti, descriptor producer, snapshot; non nuovo audit numerico|
|`verify_handoff.py`|Script standard-library che esegue la verifica documentale, senza Torch/GPU/esperimenti|

L'inventario è una mappa dei file repository osservati, non un nuovo hash audit completo del runtime o del discoE. Le date delle entry e gli hash di snapshot non devono essere confusi con una nuova ammissione scientifica. Le copie delle vecchie istruzioni sono intenzionalmente conservate per tracciabilità; la ripresa corrente è descritta all'inizio di questo documento e nell'indice aggiornato.
