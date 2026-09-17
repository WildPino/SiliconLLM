# STRAT-02 W4 v1 — censimento completo dei gruppi non codificabili

**Preregistrato il 17 settembre 2026 dopo il `VOID_FORMAT` del pilot v1 e
prima di scandire gli altri lineari.** Il
[risultato v1](../probes/STRAT_02_W4_PILOT_V1_RESULT.md) ha identificato
256 gruppi non nulli, tutti nella riga shared del router (16 per layer),
ma non ha controllato gli altri 6.209 tensori lineari. Questo censimento
è sola diagnosi del *formato già congelato*: non quantizza un candidato,
non misura BPB/tok/s, non usa heldout o T4 e non modifica pesi.

## Input, algoritmo e cap

- Stesso checkpoint/revisione e 11 hash dello STRAT-02 v1; verificare il
  manifest e gli shard una volta prima di leggere i tensori. Inventario
  dalla classe pin-nata su `meta`: 6.225 `nn.Linear`, tutti e soli
  13.363.052.544 pesi lineari, 104.398.848 gruppi contigui g128.
- Visitare **tutti** i tensori lineari nell'ordine lessicografico delle
  chiavi index, con slice di righe limitate. Per ogni gruppo F32 finito,
  calcolare `m=max(abs(w))`. Un gruppo con `m==0` è valido. Altrimenti
  applicare esattamente l'aritmetica del codec v1 per i fattori estremi
  `c=0.50` e `c=1.00` prima del cast F16. Se la scala a `0.50` è zero,
  almeno un candidato v1 è invalido; se la scala a `1.00` è non finita,
  almeno un candidato è invalido. La monotonicità per `m>0` rende questi
  estremi sufficienti per gli undici candidati. Controllare comunque su
  gruppi piantati che il risultato coincida col loop completo v1.
- Salvare per tensore conteggio dei gruppi, zeri veri, underflow,
  overflow/nonfinite e coordinate `(riga, gruppo)` dei gruppi invalidi,
  con conteggio globale e SHA del risultato. Nessun valore grezzo del
  peso e nessun testo del corpus negli output. I primi 256 gruppi noti
  della riga router 127 sono un controllo positivo, **non** una ragione per
  saltare quei layer.
- Directory risultato nuova/write-once; fallimento o interruzione produce
  `INCOMPLETE` e conserva il parziale. Preflight: almeno 8 GiB di RAM
  disponibile e 1 GiB di spazio output. Supervisione ogni 5 s, solo PID
  worker creato: stop a RAM disponibile <4 GiB, private commit >8 GiB o
  wall >30 minuti, inclusa la passata SHA. Nessuna cancellazione o resume
  implicito.

## Decisione fissata prima del censimento

Se e solo se **tutti** i gruppi invalidi sono i 256 gruppi della riga
`model.layers.{0..15}.mlp.gate.weight[127,:]`, e il controllo funzionale
indipendente conferma che quella riga non influenza hidden/logits di
inferenza con `num_shared_experts=1` e `always_active_experts=null`, si può
preregistrare un nuovo braccio con canonicalizzazione esplicita di quelle
16 righe a zero *prima* del W4 g128 invariato. Non chiamarlo v1: il
checkpoint serializzato cambia e i logits diagnostici router possono
differire. Il nuovo arm richiede ancora parità packed→F32, gate BPB
paired e task, poi il bridge nativo/rate.

Se compaiono invalidi in altri organi/righe, fermare la semplice
canonicalizzazione: proporre un formato/precision map nuovo che li
rappresenti, con budget byte e gate propri. Nessun underflow autorizza
automaticamente W2 o un fallback silenzioso.
