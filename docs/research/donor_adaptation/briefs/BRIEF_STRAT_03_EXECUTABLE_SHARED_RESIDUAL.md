# STRAT-03 — shared nonlineare e router economico, prima di T4

**Stato: preregistrazione, non eseguito.** Questo brief va congelato prima
di scrivere/eseguire il runner o leggere nuovi numeri donor. Solo CPU, nessun
download di pesi nuovo, training GPU, port C o timing. `STRAT-03` è la proposta
della [roadmap](../../STRATEGIC_10B_20260916/ROADMAP.md), non un E68 rerun.

## Domanda e confini

E68 ha ridotto la SSE locale aggregata con uno shared *lineare* rank 64, ma
il beneficio è dominato dal layer 27, il layer 1 peggiora e il selector usa
`z` denso per scegliere i gruppi. Prima di un pilot end-to-end costoso:

1. A **uguale numero di pesi shared**, un piccolo SwiGLU nonlineare predice
   il residuo meglio del linear-rank?
2. Quanta qualità locale si perde quando il top-3 oracle di massa viene
   sostituito da un router lineare `x→256` che non legge il FFN denso?

Un esito favorevole autorizza soltanto un brief BPB/rank/generazione e una
contabilità di byte/formato **della stessa geometria**. Nessun esito qui è
una prova su 10B, `engine.c`, qualità, tok/s o capacità di addestramento
congiunto. Il test non ripete E67: cambia l'operatore sempre attivo, non
solo lo score con cui si scelgono i tre gruppi.

## Oggetto congelato

- Donor `Qwen/Qwen2.5-1.5B`, revisione
  `8faed761d45a263340a0528343f099c05c9a4323`, CPU fp32/eager;
  nessuna modifica ai pesi. Layer `[1,7,14,21,27]`, `D=1536`, FFN
  SwiGLU `F=8960`, etichette D0c `E=256` da 35 neuroni, `k=3`.
- Etichette `density/results/d0c_labels/labels_E256.npz`, SHA-256
  `c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c`.
  Stessa semantica E68: `z=SiLU(W_gate x)⊙W_up x`,
  `y=W_down z`, oracle `a(x)=top3_g Σ_{j∈g}z_j²` in float64, tie a label
  minore; `y_sparse=Σ_{g∈a}W_down[:,g]z_g`.
- Fit: E68 `calib` 8×512, seed 424242, hash IDs
  `92fc41840c8feb0595758e540a048c1661bf3f618b3355d637ecf418b6c00157`.
  Score: esattamente le 512 posizioni E67/E68 (heldout sequenze 0–3,
  posizioni 384–511); hash slice completo
  `a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`,
  hash dei 512 ID selezionati
  `6e4160a27d10288f24d6a60fded1b0537c97fd28df8798d2a65dac235124a4b6`.
  Nessuna scelta di iperparametri/checkpoint da score.
- Target condiviso di fit, per ogni layer: **`R=y-y_sparse_oracle`**.
  Il residuale resta fisso nei confronti; il router economico non viene
  usato per riadattare lo shared. Questo isola il costo di sostituire
  il selector, invece di attribuirgli un nuovo fit.

## Bracci e modello eseguibile

| Braccio | Shared | Selettore | Lettura di `z` denso richiesta per selezione? |
|---|---|---|---|
| `MASS` | zero | mass oracle E68 | sì, controllo |
| `O-L96` | lineare rank 96 | mass oracle | sì |
| `O-N64` | SwiGLU width 64 | mass oracle | sì |
| `X-L96` | stesso lineare rank 96 | router `x→256` | **no** |
| `X-N64` | stesso SwiGLU width 64 | router `x→256` | **no** |

Shared lineare `S_L(x)=(xA)Vᵀ`, `r=96`, fit E68 con ridge
`λ=0.001·tr(XᵀX)/D`, Cholesky+SVD in float64, poi fattori fp32 allo
score. Shared nonlineare `S_N(x)=W_down[SiLU(W_gate x)⊙W_up x]`, senza
bias, width 64. Entrambi hanno **294,912** pesi (`2·1536·96` contro
`3·1536·64`). Non chiamare questo un confronto di byte identici dopo
quantizzazione: è parità di *conteggio* fp32, non di scale/packing.

Il router è `logits=xW`, `W:[1536,256]`, senza bias, top-3 stabile.
I target sono i tre ID oracle come etichette multilabel 0/1; non usa `z`
né `y` a score time. Fit separato per layer su calib, normalizzando ogni
`x` con un RMS per feature `max(sqrt(mean_calib(x_j²)),1e-6)` calcolato
**solo su calib**;
conservare 1536 scale fp32 nel costo. Fit di router e SwiGLU con AdamW
senza weight decay, seed `20260916`, minibatch 256 in ordine deterministico,
`300` aggiornamenti applicati, `lr=1e-3`, nessuna selezione best-step.
Loss router = BCEWithLogits con peso positivo `E/k` e media su tutti gli
elementi; loss shared = MSE media per elemento del residuo. Inizializzazione
PyTorch deterministica standard con seed fissato; dati/operazioni CPU fp32,
`torch.set_num_threads(6)`, `torch.use_deterministic_algorithms(True)`.
Le due reti vengono allenate **indipendentemente**; nessun gradiente del
donor. Salvare curve di loss a step 0/50/100/200/300, non scegliere lo
step in base a queste o all'heldout. Se la configurazione deterministica
non gira, registrare `VOID_APPARATUS` e riparare il runner prima di qualunque
interpretazione scientifica.

All'inferenza il blocco scelto deve poter calcolare solo le 105 righe
gate/up/down. Il runner può catturare `z` completo **per il riferimento e
il controllo**, ma deve verificare su input piantati e reali che il calcolo
selettivo dalle sole righe indicate riproduca `y_sparse` entro relativo
`1e-5`; altrimenti `X-*` non è un proxy eseguibile.

Contabilità minima per layer: selected FFN `3·D·105=483,840`, shared
`294,912`, router `D·256=393,216`, più 1536 scale; totale grandi organi
`1,171,968` pesi contro `3·D·F=41,287,680` del FFN denso. A 0.5
byte/peso sarebbero `585,984` byte/layer **solo payload ipotetico**:
nessuna fedeltà W4, scala, pad, località, throughput o RAM loader è qui
dimostrata. La head e l'attenzione restano fuori da questo conteggio locale.

## Controlli e output immutabile

1. Pin hash di brief, runner, E68 result/runner, labels, codice di caricamento,
   revision e token IDs nel JSON. Rifiutare hash/slice/shape divergenti.
   Il runner non sovrascrive un risultato esistente.
2. Prima del donor: toy per top-k/tie, BCE target, gradienti nonnulli/finiti,
   uguaglianza numero parametri shared e ricostruzione selective-vs-dense.
3. Sul donor: `MASS` per layer deve riprodurre gli SSE E68 mass-only
   `[26876.563427,154032.840704,113665.947309,446756.691130,
   7247181.889910]` a relativo `1e-5`; `O-L96` deve riprodurre il
   rank-96 ricavato dalla **stessa** ridge fit E68 solo se quel valore è
   salvato, altrimenti nessun anchor inventato. Verificare che fit e score
   siano disgiunti. Nessuna ricostruzione può leggere label heldout al fit.
4. Per ciascun layer/braccio: SSE, `SSE/Σ||y||²`, median e p95 dei rapporti
   per-token `||y-yhat||²/(||y||²+1e-12)`, calib vs heldout, e per-sequenza.
   Riportare aggregati come somma SSE, mai media delle percentuali per layer.
   Router: top-3 recall (intersezione/3), exact-set match e frequenze gruppi.
5. JSON write-once con configuration effettiva, loss curves, identificativi,
   tempo/RSS **operativi** e tutte le metriche anche se il gate fallisce.
   Non sono misure di token/s del modello.

## Decisione preregistrata

Le bande sono indipendenti, sullo score heldout, e vanno esposte come
booleani separati:

- `NONLINEAR_BENEFIT`: `O-N64/O-L96 ≤0.80` sull'SSE aggregata, almeno
  4/5 layer con SSE non superiore a `O-L96`, e nessun layer peggiore di
  `1.05×O-L96`.
- `ROUTER_TRANSFER`: `X-N64/O-N64 ≤1.20` sull'SSE aggregata, nessun layer
  peggiore di `1.50×O-N64`, e recall top-3 media ≥`0.50`. Riportare ogni
  condizione, non solo il booleano.
- `EXECUTABLE_LOCAL_SIGNAL`: `X-N64/MASS ≤0.50` sull'SSE aggregata,
  almeno 4/5 layer con `X-N64/MASS ≤0.80`, nessun layer con
  `X-N64/MASS >1.05`, e p95 per-token non superiore a MASS in almeno
  4/5 layer. Questa è una soglia per **promuovere un test end-to-end**,
  non per affermare qualità finale.

Predizione prima del run: `NONLINEAR_BENEFIT` può accendersi, ma
`ROUTER_TRANSFER` è il rischio maggiore; `EXECUTABLE_LOCAL_SIGNAL`
probabilmente non si accende. La scala `k=3/256` è volutamente dura e
non è la shape finale 10B. Se solo l'oracolo va bene, non spendere T4 su
questo router. Se il router va bene ma lo shared non basta, cambiare
capacità/partizione a budget byte fisso, non ritoccare i gate ex post.
Solo se l'ultimo gate si accende si prepara un nuovo brief per BPB,
generazione e costo eseguibile su un checkpoint addestrato.
