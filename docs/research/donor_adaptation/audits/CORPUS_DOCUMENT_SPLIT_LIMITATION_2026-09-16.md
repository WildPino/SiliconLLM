# Limite del corpus donor: split per chunk, non per documento

**Audit di protocollo, 2026-09-16; nessun nuovo score o training.** Questo
rilievo riguarda la forza degli intervalli di confidenza futuri, non annulla
retroattivamente le misure pubblicate sulle slice storiche.

## Evidenza nel codice e nei file correnti

- [`build_calib.py`](../../../../benchmarks/donor_adaptation/density/build_calib.py)
  legge i documenti sorgente, li taglia in chunk di `8192` byte, esegue uno
  shuffle globale **dei chunk**, poi divide la lista in due metà chiamate
  `calib` e `heldout` (`build()`, righe 103–161). Non salva ID del documento
  originario per ciascun chunk. Lo script verificato ha SHA-256
  `99772ec2b7ac44990386a5a1a5855f17d0d31116b18341e2b1af9e659525dd38`.
- [`common.py`](../../../../benchmarks/donor_adaptation/density/common.py)
  `make_slice()` estrae offset casuali dal file concatenato e legge una
  finestra di `seq_len*8` byte. Una finestra può attraversare il confine fra
  due chunk originariamente non adiacenti; i token sono scelti dal tokenizer
  dell'esperimento. `get_slice()` usa una cache con chiave che non include il
  tokenizer e ne controlla la compatibilità, ma questo non crea parità di
  testo fra tokenizer diversi.
- Il [manifest corrente](../../../../benchmarks/donor_adaptation/density/corpus/manifest.json)
  registra gli SHA-256 dei due file, il mix e il numero di chunk, ma non la
  mappa chunk→documento. SHA-256 del manifest:
  `491d108f2b6c443cc7b24eb1f6a267cafb03745956cf95daed50036d8670fc21`.
  Il corpus heldout è `140,596,742` byte, SHA-256
  `f46b0310c15faec59ca805d5688317d53b7655ae7099008fe5c3439460d58312`.

## Conseguenza, con confine esplicito

Lo split assicura che un *chunk* non sia in entrambe le metà; **non
garantisce** che chunk dello stesso documento non si trovino in calib e
heldout. In assenza di ID non si può verificare la disgiunzione per
documento né costruire un bootstrap clusterizzato per documento da quelle
slice. Il vecchio bootstrap per sequenza, quando usato, descrive la sua
unità di campionamento e non va rietichettato come bootstrap per documento.
Le stime BPB e i confronti paired storici restano descrittivamente validi
per i token/slice effettivamente misurati; questo audit non inventa una
contaminazione avvenuta in un particolare confronto e non trasforma i suoi
gate registrati in `VOID`.

Per un nuovo gate primario `upper-CI95(ΔBPB)≤+0.02`, come proposto nella
[roadmap](../../STRATEGIC_10B_20260916/ROADMAP.md), occorre prima congelare
un corpus con ID documento, split calib/heldout **prima** del chunking,
provenienza/licenza e hash dei documenti, testo UTF-8 identico per teacher e
bracci quantizzati, e bootstrap paired per documento. Il vecchio corpus può
restare un diagnostico di continuità storica, non la prova statistica
primaria. Un cambio di tokenizer richiede di ritokenizzare lo stesso testo,
non riutilizzare gli ID Qwen della vecchia `h1_heldout.npz`.
