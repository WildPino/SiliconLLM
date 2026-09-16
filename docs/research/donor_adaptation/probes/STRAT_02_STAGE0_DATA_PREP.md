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
