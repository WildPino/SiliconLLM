# STRAT-02 Stage 0 — protocollo baseline teacher heldout

**Congelato il 17 settembre 2026 prima dello scoring heldout reale.**
Il [terzo smoke](STRAT_02_STAGE0_SMOKE.md) ha dimostrato un solo forward
pretrained su una riga di **calibrazione**; i suoi 0,35806 BPB non entrano
nel baseline heldout e non sono un gate di qualità.

## Input e output vincolanti

- Checkpoint StdMoE revisione
  `d2a4949c9d4ad6cf47fbac131f7e020077332b21`, shard F32 con hash e
  indice in [`strat02_weight_sources.json`](../../../../benchmarks/donor_adaptation/density/strat02_weight_sources.json).
  La snapshot esplicita è nella cache `E:\AI_Cache\huggingface\hub`;
  `C:\Users\giosa\.cache\huggingface` è una junction verso E:, non una
  seconda copia indipendente.
- Manifest del corpus SHA-256
  `56f3d707040785b21e657d7ba721814fb32fe63e85feb418fafe38889d8ca749`;
  `heldout.jsonl` SHA-256
  `450da27e25755bb7c71215e148f5863d197af6893385033e6100bb212deafd0e`.
  Usare **tutte e sole le 96 righe nell'ordine JSONL originale**, 32 per
  categoria, 181.385 payload token e 746.161 byte UTF-8. L'audit deve
  ricostruire l'hash delle liste di token ID
  `5a4cf31ec7db0f3bc541f2e75a78b133a50cd3d6174abb26ef81e08c45e65289`.
- Tokenizer GPT2TokenizerFast esatto con Transformers 4.57.1,
  tokenizers 0.22.2, EOS 100257. Nessun troncamento, stride, documento
  saltato o EOS finale contato. Seguire il
  [protocollo BPB](STRAT_02_STAGE0_TOKEN_SCORING.md): score di ogni
  payload token da `[EOS]+payload[:-1]`, poi
  `BPB = sum(bits_d) / sum(bytes_d)`; non media dei BPB dei documenti.
- Persistenza write-once di una riga JSONL per documento con `source_document_id`,
  `category`, `tokens`, `bytes`, `bits`; tempi per documento solo
  diagnostici. Nessun testo grezzo negli output. Il baseline è valido solo
  con 96 righe complete, ID/byte/token binding passato e aggregato
  riconciliato; una parte incompleta resta `INCOMPLETE`, mai baseline.

## Vincoli operativi e interpretazione

Un unico processo worker verifica tutti gli 11 hash una volta, carica un
solo teacher mmap con pointer sharing completo, `eval` e
`torch.inference_mode()`, e scorre i 96 documenti senza ricaricare i pesi.
Il parent campiona **l'interprete reale** ogni 5 s: RAM fisica disponibile
almeno 8 GiB, working set e private commit worker non oltre 70 GiB;
all'avvio servono almeno 55 GiB disponibili e 65 GiB liberi nella cache.
Il limite wall dell'intera suite è **6 ore**; OOM, guard rail superato,
errore numerico o output incompleto sono inconclusivi e non autorizzano
selezione ex post di un subset. Nessuna T4 è prevista per questa prova.

Questa misura fornisce il denominatore del futuro confronto paired W4/W2.
Non dimostra fedeltà di un candidato, qualità di codice generato, velocità
di decode o esecuzione su `engine.c`. Il gate finale di qualità resta il
limite superiore CI95% sul **delta candidato-teacher** ≤+0,02 BPB, con
bootstrap per documento fissato nel protocollo; la baseline da sola non
può passarlo.
