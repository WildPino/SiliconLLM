# STRAT-02 Stage 1 — apparato di conversione W4

**Congelato il 17 settembre 2026 prima di convertire pesi reali.** È un
controllo di apparato per il braccio `W4_ALL_LINEAR`, non una nuova ricerca
di quantizer, una misura di qualità, un benchmark di inferenza o un port C.
Il [brief principale](BRIEF_STRAT_02_STDMOE_PRECISION_GATE.md), il
[formato W4 g128](../probes/STRAT_02_STAGE0_WEIGHT_FORMAT.md) e il
[risultato teacher](../probes/STRAT_02_STAGE0_TEACHER_BASELINE_RESULT.md)
restano vincolanti. **Stato: PROPOSED; nessuna conversione donor eseguita.**

## Domanda e invarianti

Possiamo trasformare esattamente i 6.225 pesi `nn.Linear` del checkpoint
F32 pin-nato in blob W4 g128 e ricostruire un riferimento F32 *soltanto dai
blob prodotti*, senza un secondo modello F32 in RAM, perdita di tensori,
modifica dell'operatore o accesso al heldout?

- Checkpoint `allenai/StdMoE_1b14b_1T_Preanneal`, revisione
  `d2a4949c9d4ad6cf47fbac131f7e020077332b21`, 11 shard con hash nel
  [manifest sorgente](../../../../benchmarks/donor_adaptation/density/strat02_weight_sources.json).
  L'inventario esatto contiene 6.259 tensori e 13.568.641.024 parametri
  distinti; i lineari sono 13.363.052.544 pesi, tutti divisibili in gruppi
  contigui di 128 colonne input. I restanti 205.588.480 parametri restano
  F32 byte-identici. Niente transpose, padding, scale aggiuntive o cambio
  di ordine nei gruppi.
- Byte attesi dei **soli** blob lineari: 6.681.526.272 di codici più
  208.797.696 di scale F16 = **6.890.323.968 byte**. Gli altri tensori
  F32 occupano 822.353.920 byte. Il payload candidato totale è quindi
  **7.712.677.888 byte**, esclusi nomi, manifest e overhead filesystem.
  Il riferimento F32 ricostruito completo vale 54.274.564.096 byte
  tensoriali: non è il peso di deployment né un numero di RAM per token.
- La chiave canonica del safetensors index va mappata uno-a-uno a un record
  output write-once (nome, shape, dtype sorgente, tipo W4/F32, dimensione
  attesa/reale, SHA-256, posizione nel contenitore). La lista `nn.Linear`
  deriva dall'esatta classe del modello istanziata su `meta`, non da una
  regex sui nomi. I keyset di modello, indice e output devono coincidere.
  Il contenitore globale può essere per-tensore o per-shard, ma va fissato
  nel manifest dell'apparato **prima** della conversione completa; non può
  cambiare i byte interni scale-then-codes definiti dal formato.

## Pilot senza heldout

Prima dell'export completo, selezionare per nome lessicografico la prima
matrice di ciascun organo `attention`, `router`, `expert` e `lm_head`
(quest'ultima è unica), identificando l'organo dall'inventario `nn.Linear`
e registrando i quattro nomi prima di leggerne i valori. Per ciascuna,
usare le prime `min(256, out_features)` righe intere. Il controllo non
produce un candidato valutabile: è solo un test limitato di I/O e codec.
Sull'indice pin-nato i nomi scelti sono, rispettivamente,
`model.layers.0.self_attn.k_proj.weight`,
`model.layers.0.mlp.gate.weight`,
`model.layers.0.mlp.experts.0.down_proj.weight` e `lm_head.weight`;
un inventario che non li conferma deve fermare il pilot.
Cap del pilot, inclusa la lettura hash degli shard: 30 minuti wall;
all'avvio almeno 8 GiB di RAM fisica disponibile e 2 GiB liberi sul
volume output; durante il run fermarsi se la RAM disponibile scende sotto
4 GiB oppure il private commit del worker supera 8 GiB. Il supervisor
può terminare soltanto il processo avviato dal pilot; ogni cap superato è
`VOID_RESOURCE`, non un verdetto sul W4.

1. Verificare tutti gli hash shard una sola volta oppure riutilizzare una
   verifica di integrità *dello stesso run*; aprire soltanto gli shard
   locali pin-nati, senza download.
2. Confrontare i byte W4 dell'encoder a tile con il riferimento row-wise
   sugli stessi input F32; verificare lunghezza e SHA del blob.
3. Decodificare **dal blob**, non dal sorgente, con la reference scalar;
   confrontare con la decodifica vettoriale e con un `nn.Linear` fittizio
   che usa il F32 ricostruito. Un mismatch è `VOID_APPARATUS`.
4. Registrare versione NumPy, tempo diagnostico, RAM/disco e tutti gli
   hash. Nessuna perplexity, score o generazione del donor in questo pilot.

Solo un pilot completo autorizza a proporre l'export dell'intero checkpoint.
Il run completo deve avere directory di output nuova, controllo dello spazio
per packed **e** ricostruzione, cap wall/RAM dichiarati al lancio, output
persistito per tensor/shard e manifest finale write-once. Un'interruzione
conserva il parziale etichettato `INCOMPLETE`, senza selezione di tensori o
resume implicito. La ricostruzione F32 destinata allo scoring deve leggere
solo packed/F32 passthrough verificati, mai i lineari originali per
correggere errori. Poi servono parità dell'operatore, controllo teacher
repeatability ancora aperto, calibrazione operativa e **una sola** misura
heldout W4 secondo il brief principale. Nessuna T4 è prevista; qualsiasi
proposta T4×2 va prima comunicata all'utente.
