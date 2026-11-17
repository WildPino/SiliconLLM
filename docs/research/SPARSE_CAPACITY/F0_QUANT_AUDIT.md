# F0: audit della quantizzazione, addestramento contro motore

2026-11-17. Fase 0 del [piano](PLAN.md), serve alla fase 2 (D-0, D-R) e alla fase 5. Nessun codice e stato modificato:
le correzioni entrano nel protocollo della fase 2.

**Cosa si confronta.**
- **Addestramento:** `benchmarks/phase64/mve/mve_model.py` (`SparseMoEMLP`, `BitLinear158Alpha`, `qat_ternary`,
  `upcycle_moe`), che riusa `BitLinear158` (`benchmarks/phase57/phase57_ternary.py`) e `MoEMLP`
  (`benchmarks/phase57/phase59_moe.py`).
- **Motore:** `benchmarks/phase60/engine.c`, percorso `--mlp lut`, con l'export `benchmarks/phase60/e4_export.py`.

## 1. In breve

- **I pesi coincidono gia.** Scala per riga = media dei valori assoluti, `round` e `clamp(-1, 1)`, la stessa formula
  nell'addestramento e nell'export. Coincidono anche la dReLU, il router e l'ordine "peso dell'esperto, poi down".
- **Le attivazioni no: e la differenza che conta.** L'addestramento e "solo pesi ternari", con attivazioni in virgola
  mobile (`phase57_ternary.py:14`). Il motore quantizza le attivazioni a interi in [-63, 63] in due punti per strato.
- **Oggi l'addestramento corrisponde al percorso `--mlp fp32` del motore,** che usa i pesi ternari dequantizzati e
  attivazioni fp32 (`engine.c:298-301`). Il percorso veloce `--mlp lut` aggiunge la quantizzazione delle attivazioni.
- **Per i micro-esperti (h8, h1) il motore non ha un percorso:** vedi §3.

## 2. Le differenze, e come si risolvono a favore di engine.c

| # | Dove | Addestramento | Motore | Effetto | Correzione per la fase 2 |
|---|---|---|---|---|---|
| A1 | `mve_model.py:105-111`; `engine.c:296` | x all'ingresso degli esperti in fp (o fp16/bf16 sotto autocast) | `quant_i8(xn)`: una scala per token su D, absmax/63, `lrintf`; la stessa per tutti gli esperti del token | Il modello non vede mai il rumore di quantizzazione che incontra nel motore | Fake-quant con STE su x, per token su D, Q = 63, round-half-even. Il router resta fp32 e legge x **non** quantizzato, come `engine.c:289` |
| A2 | `mve_model.py:112-113`; `engine.c:306-307` | h dell'esperto in fp prima della down | `quant_i8(he)` per (token, esperto) su hid_e, **dopo** la moltiplicazione per il peso di instradamento | Come A1, sul secondo ingresso | Fake-quant su h pesato, per (token, esperto) sull'ultima dimensione. Nel percorso compute-all (`phase59_moe.py:95-100`) la stessa operazione va fatta sul blocco (N, E, hid_e), mai sull'hidden intero (N, E·hid_e): una scala condivisa fra esperti sarebbe sbagliata |
| A3 | `phase57_sparse.py:58-75`; `engine.c:259, 270` | MLP densa (fasi C-D): stesse due attivazioni in fp | quant su D all'ingresso e su tutto l'hidden (1024) prima della down | Come A1-A2 per le fasi dense | Stessa fake-quant, scala per token sull'hidden intero |
| W1 | `phase59_moe.py:84-87`, `phase57_ternary.py:41-44` sotto `torch.autocast` (`mve_train.py:828`) | `wq * scale` e convertito in fp16 o bf16 prima del prodotto: la scala perde precisione (2⁻¹¹ in fp16, 2⁻⁸ in bf16, relativo) | interi accumulati in int32, poi `S * sa * sc` in fp32 (`engine.c:304, 309`) | Piccolo, ma sistematico per riga | Prodotto con `wq` in {-1, 0, 1} (esatto in ogni formato), scala applicata dopo, in fp32, come nel motore |
| W2 | `mve_model.py:148-158` | `BitLinear158Alpha` con alpha < 1 non e ternario | il motore legge solo pesi ternari | Un export a meta rampa sarebbe un altro modello | L'export rifiuta un checkpoint con alpha < 1 |
| R1 | `mve_model.py:72`; `engine.c:278-283` | `torch.topk` | a parita, vince l'indice piu basso | A probabilita fp32 una parita esatta e rarissima | Nessuna; la parita dei logit della fase 5 la coprirebbe |

**Verificati uguali, nessuna azione:**
- formula del peso ternario e scala per riga (`phase59_moe.py:84-87` e `e4_export.py:19-23`);
- scala della down per (esperto, uscita d) su hid_e (`e4_export.py:61`, `engine.c:309`);
- dReLU `relu(g) * relu(u)` (`mve_model.py:111`, `engine.c:306`);
- router softmax su tutti gli E in fp32, con bias, top-k rinormalizzato (`mve_model.py:70-73`, `engine.c:289-294`);
- peso di instradamento applicato all'hidden prima della down (`mve_model.py:112`, `engine.c:306`);
- RMSNorm con eps 1e-5 (`phase55_ssm.py:43`, `engine.c:250`);
- arrotondamento al pari piu vicino: `torch.round` e `lrintf`, con il modo di arrotondamento di default.

**Perche Q = 63 e non 127:** la LUT somma due prodotti peso per attivazione in un int8 (`engine.c:158-159`), quindi
|x| ≤ 63 tiene |w0·x0 + w1·x1| ≤ 126. BitNet b1.58 (arXiv 2402.17764) quantizza le attivazioni a 8 bit per token;
qui ne restano circa 7. Il costo in qualita non e misurato: va misurato nella fase 2, prima di scegliere la precisione.

## 3. Fuori dalla quantizzazione, per la fase 5

- **Il motore ha forma fissa a compilazione:**
  - `L = 6` (`engine.c:56`) contro `L = 8` dell'S0 (`mve_model.py:162`);
  - E, HID_E e KTOP fissi (`engine.c:68-70`);
  - niente x_proj a rango basso (`mve_model.py:255-265`): l'export legge `x_proj.weight` (`e4_export.py:50`) e va
    adattato per moltiplicare U·V;
  - niente slot di recall.
- **Micro-esperti:**
  - la LUT lavora a coppie lungo l'ingresso, e la down di un esperto h1 ha un solo ingresso (`TDE = HID_E/2`,
    `engine.c:74`);
  - `matvec_lut_rows` lavora a tessere di 32 righe (`engine.c:131-138`): a h8 legge 4 volte le righe che servono;
  - con la convenzione attuale la down di un esperto h1 ha **una scala per peso** (scala per riga su hid_e = 1), quindi
    non e compressa: circa 4,5 byte per peso invece dei circa 0,5 del formato a coppie. A h8 le scale costano 0,5
    byte per peso.
  - Questa convenzione non si puo risolvere a favore del motore, perche il motore non ha un percorso per h < 32. La
    decide il protocollo della fase 2, se la fase 2 usa esperti sotto h32, e la fase 5 la implementa.

## 4. Cosa entra nel protocollo della fase 2

- A1-A3 e W1 come un'unica opzione `act_quant=engine` del modello, con un selftest: su un batch fisso l'uscita di uno
  strato con fake-quant coincide con una replica in numpy di `quant_i8` e del prodotto intero, a meno dell'arrotondamento
  fp32.
- Il controllo dell'export (W2).
- Se la fase 2 confronta ternario e bf16, la quantizzazione delle attivazioni e un braccio a se: il ternario con
  attivazioni fp e quello con attivazioni a 63 livelli sono due modelli diversi.
