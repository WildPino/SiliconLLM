# STRAT-02 Stage 0 — provenienza degli span effettivamente scelti

**Data:** 2026-09-16. **Stato:** inventario read-only; nessun peso o
forward eseguito. Questo non è un parere legale né un permesso di
ridistribuire testo. L'uso previsto è valutazione locale e pubblicazione di
soli hash/metriche, non degli span raw.

L'[inventario preliminare](STRAT_02_STAGE0_DATA_PREP.md) contava 109 file
*sorgente* nel manifest. Qui conto invece i **144 documenti selezionati** nei
due JSONL congelati (`calib` 48, `heldout` 96), senza stampare il loro testo.
Il manifest ha SHA-256
`56f3d707040785b21e657d7ba721814fb32fe63e85feb418fafe38889d8ca749`;
i JSONL hanno SHA-256 `f1ed84f64284d2cd6ffd59f2373849f41a3c33bdb327eaa75f0f2b7e0e3d998f`
e `450da27e25755bb7c71215e148f5863d197af6893385033e6100bb212deafd0e`.
Ogni riga contiene `source_path`, `source_document_id`, hash del file e del
contenuto sorgente, offset, lunghezza e hash dello span; l'audit tokenizer
verifica identità, split e assenza di overlap esatto. I path dei testi raw
restano locali e ignorati da Git.

| Categoria | Fonte selezionata | Calib | Heldout | Dichiarazione di fonte/licenza da conservare |
|---|---|---:|---:|---|
| Code | CPython | 8 | 15 | `data/external/the_stack_python/cpython/LICENSE` (PSF e avvisi inclusi) |
| Code | Django | 5 | 8 | `data/external/the_stack_python/django/LICENSE` |
| Code | pip | 2 | 6 | `data/external/the_stack_python/pip/LICENSE.txt`; **una** riga usa `pip/src/pip/_vendor/packaging/pylock.py`, quindi conservare anche `pip/src/pip/_vendor/packaging/LICENSE*` |
| Code | NumPy | 1 | 2 | `data/external/the_stack_python/numpy/LICENSE.txt`; verificare eccezioni locali per i path selezionati |
| Code | Requests | 0 | 1 | `data/external/the_stack_python/requests/LICENSE` |
| Technical general | MDN | 6 | 17 | `data/external/markdown_corpus/mdn/LICENSE.md`; prose e snippet possono avere condizioni distinte, come spiega la [guida MDN](https://developer.mozilla.org/en-US/docs/MDN/Writing_guidelines/Attrib_copyright_license) |
| Technical general | Kubernetes website | 10 | 15 | `data/external/markdown_corpus/kubernetes_website/LICENSE`; [publisher LICENSE](https://github.com/kubernetes/website/blob/main/LICENSE) |
| Prose | PG-19 / Gutenberg | 16 | 32 | [PG-19 README](https://github.com/google-deepmind/pg19/blob/master/README.md): libri estratti da Project Gutenberg, pubblicati prima del 1919; il metadata del dataset dichiara Apache-2.0 |
| **Totale** | | **48** | **96** | |

Le licenze dei sette repository di codice/Markdown sono presenti localmente
e i loro SHA-256 sono registrati nel [data-prep](STRAT_02_STAGE0_DATA_PREP.md).
Il file `pylock.py` vendorizzato da pip richiede di conservare anche il notice
di `packaging`; la licenza top-level di pip da sola non basta a documentarlo.
Nessuna riga selezionata è stata riclassificata o sostituita dopo il build.

**Caveat PG-19 importante:** il metadata Apache-2.0 di DeepMind descrive il
dataset, ma non trasforma automaticamente i diritti di ogni libro sorgente.
[Project Gutenberg](https://www.gutenberg.org/policy/terms_of_use.html) avverte
che lo status del pubblico dominio negli Stati Uniti non garantisce quello
fuori dagli USA. Perciò non attribuisco ad ogni singolo span una licenza
universale né propongo di committare o distribuire il JSONL. Gli ID di riga
PG-19 e i titoli/URL originali rimangono ricostruibili dai Parquet locali;
se servirà **pubblicare i testi** o eseguire in una giurisdizione con requisiti
diversi, occorrerà una verifica dei singoli libri o un nuovo corpus
preregistrato. Questa riserva non va silenziosamente convertita in clearance.

Questo inventario chiude la domanda *da dove provengono gli span scelti* e
quali notice conservarne nell'audit; non chiude da solo gli altri gate Stage 0:
runtime esatto, parser task, sandbox HumanEval, peak RAM full-size e futura
parità del teacher restano indipendenti.
