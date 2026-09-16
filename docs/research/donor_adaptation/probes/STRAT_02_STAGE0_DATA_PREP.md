# STRAT-02 Stage 0 — corpus documentale congelato, nessuna qualità letta

**Data:** 2026-09-16. **Stato:** preparazione dati `VALID`; il gate di
precisione STRAT-02 è ancora `PROPOSED`, non eseguito. Nessun peso donor,
training, T4, export C o benchmark tok/s.

Il [brief](../briefs/BRIEF_STRAT_02_STDMOE_PRECISION_GATE.md) aveva fissato
il preset `r1_preflight_v1` prima del build. Il
[generatore](../../../../benchmarks/donor_adaptation/density/build_document_holdout.py)
è al commit `44cda68`, SHA-256
`63cd0726a021306240729e94a91f322277eb7a31d32369877e9dd274309eb910`.
Ha eseguito una scansione completa delle sorgenti locali e creato una sola
volta il [manifest](../../../../benchmarks/donor_adaptation/density/corpus/strat02_document_holdout_v1/manifest.json),
SHA-256 `56f3d707040785b21e657d7ba721814fb32fe63e85feb418fafe38889d8ca749`.
Gli span raw JSONL restano sul disco locale, esclusi da Git per non
ridistribuire automaticamente testo di terzi.

| Split | Documenti | Code | Technical general | Prose | Byte degli span UTF-8 |
|---|---:|---:|---:|---:|---:|
| Calib | 48 | 16 | 16 | 16 | 362,405 |
| Heldout | 96 | 32 | 32 | 32 | 746,161 |

Hash SHA-256 locali del JSONL: calib
`f1ed84f64284d2cd6ffd59f2373849f41a3c33bdb327eaa75f0f2b7e0e3d998f`;
heldout `450da27e25755bb7c71215e148f5863d197af6893385033e6100bb212deafd0e`.
Questi sono **byte di span**, non il denominatore BPB effettivamente scored:
il runner dovrà fissare BOS, token iniziale/stride e byte della continuazione.

Controlli: self-test toy passed (leakage di chunk, duplicato cross-split,
documento corto e write-once piantati); `verify --check-source-files` passed;
un controllo indipendente sui due JSONL ha trovato `0` ID-documento e `0`
content-SHA identici fra split, ricontando tutte le tre categorie e i byte.
Lo scan ha visto `39,064` documenti unici dopo deduplica esatta globale,
scartando `6` duplicati esatti; il manifest registra `109` file sorgente
selezionati e i loro hash. Disponibilità dopo filtro `min_source_bytes=4096`:
calib `1433/3776/28499` e heldout `1407/3849/100`
per code/technical-general/prose. I rifiuti per lunghezza sono nel manifest.

**Confini:** la deduplica esatta non esclude near-duplicates o sovrapposizioni
semantiche. Le licenze/provenienze delle fonti locali devono ancora essere
verificate prima di promuovere la suite. L'import isolato del codice StdMoE
e il modello `meta` sono controlli strutturali separati, non teacher forward.
Mancano ancora protocollo BPB/generazione/tasks interamente fissato,
baseline F32, quantizer W4/W2, parità dell'operatore, loader a memoria
limitata, qualità e rate sullo stesso artefatto. Il build dati non autorizza
automaticamente il download dei 54 GB di pesi.

### Inventario preliminare di provenienza (non clearance)

I 109 file sorgente registrati nel manifest provengono da: 25 file del
repository locale `kubernetes_website`, 23 `mdn`, 23 `cpython`, 13 `django`,
3 `numpy`, 8 `pip`, 1 `requests` e 13 shard Parquet `pg19` (questi ultimi
contengono molti documenti). I file di licenza locali esistono per i primi
sette gruppi; i loro SHA-256 sono rispettivamente
`d07988f4be47912f75a73b44da8a9c0b602bf6c8ced69de858b848f379ecd973`,
`97a45d9a5c27c90ad7f058b20a870f11a24717f0eb4585f141948d25b74c4d89`,
`db693914a7f6d42f1d3e09c10eda1482e5d94ed4c70a769476ae8d722a9be1ce`,
`7e493fa7ce2cfdb8dc8a97d5e912f81fc5e0ddc58c5a236e98de2ee55ca978a8`,
`1be1df33863f97a7bc1c4d67980bd6c69c9a6fef0a5ee76e6ad6cb91e56e8491`,
`afbb3c587ad82d668258516cfb0364d964670097a226fcc980e3217a76e16e8d`,
`88046bf22d5b4f4b8cc85079ae6aae5424a3a1999db952ed152828ff325b2c6d`.
Questo localizza le dichiarazioni di licenza; non prova che ogni singolo file
o traduzione sia coperto senza eccezioni né autorizza ridistribuzione di
testo raw. [Kubernetes](https://github.com/kubernetes/website/blob/main/LICENSE)
dichiara CC BY 4.0; [MDN](https://github.com/mdn/content/blob/main/files/en-us/mdn/writing_guidelines/attrib_copyright_license/index.md)
distingue prose CC BY-SA e campioni di codice. Il progetto
[The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) avverte
che il codice conserva le licenze originali e gli obblighi di attribuzione.
PG19 deriva da libri di Project Gutenberg, secondo
[DeepMind](https://github.com/google-deepmind/pg19/blob/master/README.md),
ma non è presente un file LICENSE nella copia locale `data/external/pg19`:
la provenienza/redistribuibilità a livello di libro resta da chiarire.
Gli span JSONL rimangono per questo solo locali e ignorati da Git.
