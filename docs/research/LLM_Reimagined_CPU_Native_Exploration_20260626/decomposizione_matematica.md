# Decomposizione matematica del fenomeno LLM + albero delle innovazioni

> **Mandato (utente, 2026-06-26):** sviluppa innovazione; dividi tutti i problemi in sotto-problemi, ricorsivamente; pensa fuori dagli schemi (es. *fondi 4 valori in 1, tipo hash — sacrifichi un po' di precisione per 4× dati*); approccio **matematico** al fenomeno LLM. Documento tecnico denso. Versione spiegata: vedi `spiegazione/`.

---

# Parte I — Formalizzazione (M1)

## I.1 L'oggetto

Un LLM è una mappa parametrica
$$ f_\theta : \Sigma^* \to \Delta(\Sigma), \qquad \theta \in \mathbb{R}^N $$
dove $\Sigma$ è l'alfabeto di token ($|\Sigma|=V$), $\Sigma^*$ le sequenze finite, $\Delta(\Sigma)$ il simplesso di probabilità su $\Sigma$. Dato un contesto $c=(x_1,\dots,x_{t-1})$ produce $p_\theta(\cdot\mid c)$. Il decode è il punto fisso stocastico
$$ x_t \sim p_\theta(\cdot \mid x_{<t}), \quad \text{poi } c \leftarrow c\,\Vert\,x_t,\ \text{ripeti.} $$

Strutturalmente $f_\theta = U \circ (\bigcirc_{l=1}^{L} \mathrm{Block}_l) \circ E$, alternanza di:
- **token-mixing** (tra posizioni): ricorrenza SSM $h_t = A h_{t-1} + B x_t,\ y_t = C h_t$ — **stato $O(1)$**, niente KV che cresce;
- **channel-mixing** (dentro la posizione): $\mathrm{FFN}(x)=W_2\,\sigma(W_1 x)$ — **~2/3 di $N$**;
- più embedding $E\in\mathbb{R}^{V\times d}$, head $U\in\mathbb{R}^{d\times V}$, norme.

## I.2 Il funzionale di costo (la fisica della CPU)

A decode batch-1 il tempo-per-token è dominato dal traffico DRAM:
$$ T_{\text{tok}} \approx \frac{B_{\text{tok}}}{\beta_{\text{eff}}}, \qquad B_{\text{tok}} = \sum_{i=1}^{N} b_i \, a_i(c) $$
- $b_i$ = byte per memorizzare il peso $i$;
- $a_i(c)\in\{0,1\}$ = il peso $i$ è letto per questo token nel contesto $c$;
- $\beta_{\text{eff}}$ = banda effettiva (dipende dalla cache-residency e dal pattern d'accesso).

## I.3 Il problema-master (fattorizzazione a 5 leve)

Scrivo i byte attivi attesi come prodotto di fattori **quasi-indipendenti**:
$$ \boxed{\ \text{tok/s} \;\approx\; \frac{\beta_{\text{raw}}\cdot \rho \cdot m}{\bar b \cdot \bar a \cdot N}\ } $$
con
| simbolo | nome | direzione | leva |
|---|---|---|---|
| $\beta_{\text{raw}}$ | banda DRAM grezza | fisso (~45 GB/s) | — |
| $\rho \ge 1$ | fattore cache-residency $=\beta_{\text{eff}}/\beta_{\text{raw}}$ | ↑ | **P1** |
| $m \ge 1$ | token emessi per passata-pesi | ↑ | **P2** |
| $\bar b$ | byte/peso medi | ↓ | **P3** |
| $\bar a$ | frazione di pesi attivi/token | ↓ | **P4** |
| $N$ | capacità (parametri) | **↑ desiderato** | — |

**Il vincolo (la quinta leva, trasversale):**
$$ \mathrm{BPB}(\theta;\ \text{compressione},\ \text{sparsità}) \le \tau \qquad \textbf{(P5)} $$
+ **penalità d'accesso**: l'accesso random abbatte $\rho$ (latenza ~100ns/miss) → ogni leva che introduce gather casuale paga $\rho$.

## I.4 La tesi che il problema-master rende OVVIA

$N$ sta a **denominatore** ma lo vogliamo **grande** (capacità = intelligenza). L'unico modo di crescere $N$ senza pagare in tok/s è **far crollare $\bar a$** (e $\bar b$) così che $\bar a \cdot N$ resti piccolo:
$$ \text{disaccoppia la CAPACITÀ } N \text{ dai BYTE-ATTIVI } \bar a\, \bar b\, N. $$
Questa è, in una riga, l'intera strategia "memoria grande / set-attivo piccolo" (= product-key memory K1). **Il problema-master non è una metafora: dice esattamente dove spingere.**

---

# Parte II — Albero dei problemi (M2 + M3)

Notazione: **P** = leva, **P.x** = sotto-problema, **P.x.y** = foglia azionabile. Ogni foglia ha un *trade* $(\Delta\text{costo}, \Delta\text{qualità})$.

## P1 — Alzare $\rho$ (cache-residency)
- **P1.1 — far stare il working-set attivo in cache.**
  - P1.1.1 — rimpicciolire il nucleo-compute → **weight-sharing/profondità-ricorsiva** (K2). *Trade: $-$qualità(tying) per blocco $\le$ L3.*
  - P1.1.2 — rimpicciolire lo slice-memoria attivo → rimanda a **P4**.
  - P1.1.3 — **layout dei dati**: rendere l'accesso sequenziale (codici per-bucket contigui, righe allineate a cache-line 64B). *Trade: nessuno di qualità, solo ingegneria.*
- **P1.2 — predire & prefetchare il prossimo working-set** (K3) → converte latenza-random in streaming.
  - P1.2.1 — **predittore** $g_\phi(h_t)\to \hat s_{t+1}$ (set attivo successivo).
  - P1.2.2 — **rendere il modello predicibile** (regolarizzatore $L_{\text{pred}}$, gradiente nel backbone). *Trade: $-$qualità(reg) per hit-rate prefetch.*

## P2 — Alzare $m$ (token per passata)
- **P2.1 — multi-token / speculative** (MTP self-draft). *Trade: complessità per ~1.5-2.5×.*
- **P2.2 — tasso di accettazione** dei token bozza (verifica).
- **P2.3 — (invertito) non-autoregressivo / diffusion** → **SCARTATO** su CPU batch-1 (vedi P-traps).

## P3 — Abbassare $\bar b$ (byte/peso) — **QUI VIVE L'ESEMPIO HASH**
Pattern generale = mappa di **fusione lossy** $\varphi:\mathbb{R}^k\to \mathcal{C}$ che sostituisce $k$ pesi reali con **un codice** di $\lceil\log_2|\mathcal{C}|\rceil$ bit:
$$ \text{compressione} = \frac{k\cdot 32}{\log_2|\mathcal{C}|}, \qquad \varepsilon = \mathbb{E}\big\|w-\varphi^{-1}(\varphi(w))\big\|. $$
- **P3.1 — scalare** ($k{=}1$): ternario $|\mathcal{C}|{=}3\Rightarrow 1.58$ bit, ~20×, $\varepsilon$ alto.
- **P3.2 — vettoriale/product-quant** ($k{=}4,|\mathcal{C}|{=}256$): $128\,\text{bit}\to 8\,\text{bit}$, **16×**, $\varepsilon$ basso (codebook appreso). **= L'ESEMPIO DELL'UTENTE.**
- **P3.3 — hashing** (HashedNet): $k$ pesi → **1 valore condiviso** via $h(i)$; compressione $=k$, **nessun indice da memorizzare** ($h$ lo calcola), $\varepsilon$ = rumore di collisione.
- **P3.4 — strutturato** (Monarch/butterfly): meno parametri liberi $+$ accesso sequenziale SIMD.
- **P3.5 — entropico** (coding aritmetico dei pesi): $\bar b\to H(\theta)$, ma **decode seriale sul critical-path** → trappola.

## P4 — Abbassare $\bar a$ (sparsità)
- **P4.1 — conditional compute**: MoE / **product-key memory** (K1). $\bar a = m/N \to 0$ crescendo $N$. *Trade: gather-random per $\bar a$ minuscola.*
- **P4.2 — sparsità d'attivazione allenata** (Q-Sparse top-K, dReLU). $\bar a \approx$ frazione-top-K.
- **P4.3 — profondità adattiva / early-exit** (DEQ-bounded, MoR-halt). $\bar a$ scende sui token facili.
- **P4.4 — tasso-token adattivo** (patch a entropia, K4): salta compute sui tratti prevedibili. = **fusione su asse TEMPO** (vedi P4.4* sotto).

## P5 — Preservare la qualità (vincolo $\tau$) — dove il "sacrificio di precisione" va **bounded**
- **P5.1 — sensitività**: *dove* $\varepsilon$ costa poco. $\partial \mathrm{BPB}/\partial \varepsilon_i$ per componente → comprimi aggressivo dove la derivata è bassa (FFN bulk), conservativo dove è alta (head, recall-head, token rari).
- **P5.2 — residuo/correzione**: $W = W_{\text{lossy}} + \Delta W_{\text{sparse}}$, correggi solo i top-$\varepsilon$ errori. *Trade: pochi % di byte per recuperare la coda di qualità.*
- **P5.3 — train-aware-dell'approssimazione**: metti l'operatore lossy NEL forward di training (QAT, STE) → il modello si adatta all'errore. *La compressione "gratis" perché il modello la compensa.*

---

# Parte III — Innovazioni fuori-schema alle foglie (M4)

Ogni innovazione = una mossa "**fondi $k\to1$, sacrifica $\varepsilon$ per $k\times$**" applicata a un asse diverso. Le tratto matematicamente e quantifico il trade. **★ = candidato originale/forte; ◇ = noto, da adottare; ⚠ = trappola.**

## III.1 — Hash sui PESI (P3.2/P3.3) — l'esempio dell'utente, generalizzato
**(a) ◇ Product-quant dei pesi (LUT-GEMM).** Raggruppa 4 pesi → vettore $\in\mathbb{R}^4$, memorizza l'indice del centroide più vicino in un codebook di 256. $8\times$ compressione; il matvec diventa **lookup in tabella** (pshufb-AVX2, niente VNNI). $\varepsilon$ minimizzabile con codebook appreso **per-sottospazio** e — twist nostro — **anisotropico** (ScaNN): penalizza l'errore nella direzione che conta per l'output, non la norma.
**(b) ★ Hashing dei pesi + ternario (HashedNet ⊗ BitNet).** $h:\{1..N\}\to\{1..M\}$, $M\ll N$: i pesi che collidono **condividono un valore ternario**. Compressione $=N/M$ **senza memorizzare nessun indice** (l'hash lo calcola al volo). Combinato col ternario: $\bar b \to \frac{1.58}{N/M}$ bit/peso-logico. **Novità nostra:** nessuno compone hashing-condiviso con ternario su CPU; e l'hash dà accesso **deterministico** (calcolabile = prefetchabile, lega a K3). $\varepsilon$ = rumore di collisione, controllabile con $M$ e con **segno-hash** (Count-Sketch: $w_i \approx \text{sign}(i)\cdot v_{h(i)}$, collisioni a media-zero).
**(c) ★ Codebook condiviso lungo la profondità (P3.2 ⊗ K2).** Se il blocco è riusato (K2), **condividi anche il codebook** tra le ricorsioni → i byte unici crollano ancora. Compressione composta $=k \times K_{\text{reuse}}$.

## III.2 — "Sacrifica precisione, poi correggi dove conta" (P5.2)
**(d) ★ Low-rank + residuo sparso ternario.** $W \approx \underbrace{U V^\top}_{\text{rank-}r,\ \text{cheap}} + \underbrace{\Delta}_{\text{sparso, top-}\varepsilon}$. Il grosso di $W$ in $r(d_1{+}d_2)$ numeri (lossy, economico), più una manciata di correzioni dove l'errore morde. *Trade: $\varepsilon$ concentrato → correzione mirata invece di bit uniformi.* Compone con P5.1 (la sensitività dice *dove* mettere $\Delta$).

## III.3 — Sacrificare precisione **per-token** ma recuperarla **sulla traiettoria** (P5.3) — la più fuori-schema
**(e) ★★ Stochastic rounding self-averaging.** Arrotonda i pesi/attivazioni in modo **stocastico e non-distorto**: $\mathbb{E}[\hat w]=w$. Per-token l'errore è alto (varianza), MA il decode è una **traiettoria di $T$ passi**: sotto indipendenza, l'errore aggregato sull'output scala come $\varepsilon/\sqrt{T_{\text{eff}}}$ — **il modello si auto-media**. → permette $\bar b$ **più aggressivo** del rounding deterministico a parità di BPB. **Genuinamente fuori-schema e cheap da testare** (cambi solo l'operatore di quantizzazione). Rischio: la correlazione temporale riduce $T_{\text{eff}}$; la varianza può destabilizzare il sampling a bassa-temp (lega ai loop di Phase 54 → la rep-penalty lockata aiuta).

## III.4 — Matmul approssimato per campionamento (confine P3/P4)
**(f) ★ Dot-product campionato.** $y=\sum_j w_j x_j$: campiona i termini con $|w_j x_j|$ grande (importance sampling), salta il resto. Su CPU **= leggere meno pesi** = meno byte. $\hat y$ non-distorto, varianza $\propto 1/(\text{#campioni})$. *Trade: varianza per $\bar a$.* Twist nostro: il pattern di campionamento, se **fissato dal contesto** (lo stato SSM lo predice), diventa prefetchabile (K3) invece che dinamico-casuale.

## III.5 — Hash sull'asse TEMPO, non sui pesi (P4.4)
**(g) ★ Token-merge gerarchico (hash sul tempo).** Fondi 4 stati-token consecutivi → 1 riassunto (pooling appreso), fai girare i **layer costosi a 1/4 del tasso**, poi de-pooling. = "fondi 4→1" applicato al TEMPO: $4\times$ meno passate costose, sacrifichi risoluzione temporale. È la versione hard della patch-a-entropia (K4): hourglass/MegaByte-style. *Trade: risoluzione per $m_{\text{eff}}\times 4$.* Compone con tutto il resto (ortogonale agli assi pesi).

## III.6 — Memoria probabilistica con collisioni accettate (P4.1, con cautela)
**(h) ⚠→◇ Sketch di frequenza per il recall caldo.** NON la superposizione VSA (morta: capacità $\sim D/16$, Phase 51). Un **Count-Min sketch** dei pattern/n-grammi **frequenti**: $k$ contatori, collisioni a sovrastima-bounded. Serve solo i token caldi (coda corta), residente in L2. *Trade: precisione del recall raro (che sta altrove) per un hot-cache cheap.* Nicchia minore, non baricentro.

---

# Parte IV — Triage (M5)

## IV.1 — Ranking (novità × leva × CPU-fit × cheap-to-test)
| Id | Innovazione | Leva (asse) | Novità | Probe |
|----|-------------|-------------|--------|-------|
| **(e)** | Stochastic-rounding self-averaging | $\bar b$ ↓ (P5.3) | ★★ alta | **cheap, ORA** |
| **(b)** | Hash-weights ⊗ ternario (+Count-Sketch) | $\bar b$ ↓ (P3.3) | ★ alta | cheap |
| **(g)** | Token-merge gerarchico (hash sul tempo) | $m,\bar a$ (P4.4) | ★ media | medio (train) |
| **(d)** | Low-rank + residuo sparso | $\bar b$ ↓ (P5.2) | ★ media | cheap (post-hoc) |
| **(c)** | Codebook condiviso in profondità | $\bar b$ ↓ ⊗ K2 | ★ media | con Probe-1(K2) |
| **(f)** | Dot-product campionato + prefetch | $\bar a$ ↓ (P4) | ★ media | microbench |
| **(a)** | Product-quant pesi (LUT-GEMM, anisotropico) | $\bar b$ ↓ (P3.2) | ◇ noto+twist | microbench |
| **(h)** | Count-Min hot-cache | $\bar a$ (P4.1) | ◇ minore | rimandato |

## IV.2 — Le due candidate più fuori-schema e cheap (le porto avanti)
1. **(e) Stochastic-rounding self-averaging** — la più originale come *principio* (sfrutta che il decode è una traiettoria, non un colpo singolo). Probe: prendi il checkpoint Phase-55 esistente, quantizza i pesi con rounding **stocastico** vs **deterministico** a pari $\bar b$ aggressivo, misura BPB e leggi 8 sample (con la rep-penalty lockata). *Una variabile, zero training, ore.* Domanda secca: *a pari byte, lo stocastico tiene la qualità meglio del deterministico?*
2. **(b) Hash-weights ⊗ ternario** — la traduzione diretta del tuo esempio, spinta. Probe: microbench C — matvec con pesi hash-condivisi-ternari (compressione $N/M$) vs ternario pieno; misura tok/s + $\varepsilon$ sul matvec; poi un mini-train per la BPB. *Cheap.*

## IV.3 — Mappa innovazioni ↔ keeper esistenti (tutto si incastra)
- P3 (byte/peso) → era già l'asse **ternario-LUT** del bandwidth-report; (a)(b)(c)(d)(e) lo **arricchiscono**.
- P4 (sparsità) → **K1** (product-key) + **K4** (token-rate); (f)(g)(h) lo arricchiscono.
- P1 (cache) → **K2** (sharing) + **K3** (predicibilità); il layout P1.1.3 è ingegneria.
- P2 ($m$) → **K5** (MTP). P5 (qualità) → il guardrail trasversale, dove vive la disciplina anti-Goodhart.
- **La decomposizione NON sostituisce i keeper: dà loro un sistema di coordinate** e svela le foglie nuove (e=★★, b=★) che l'analisi a-stazioni (exploration.md) non aveva isolato.

## IV.4 — Trappole confermate (P-traps)
- ⚠ **P3.5 entropic-weights**: decode seriale sul critical-path perde vs "meno bit".
- ⚠ **P2.3 diffusion/NAR**: su CPU batch-1 perde (verificato, arXiv:2510.04146).
- ⚠ **VSA-superposition** (≠ III.6 sketch): morta, capacità $\sim D/16$ (Phase 51).
- ⚠ **sparsità non-strutturata** generica: accesso random = muro latenza.
- ⚠ **regola d'oro**: ogni leva che introduce gather-random paga $\rho$ → va sempre accoppiata a P1.1.3 (layout) o P1.2 (prefetch). Mai una "compressione" che diventa accesso casuale non prefetchato.

---

# Sintesi

Il fenomeno LLM, ridotto a matematica, è **un problema di ottimizzazione con un funzionale di costo a 5 leve** ($\rho, m, \bar b, \bar a$ contro il vincolo $\tau$), con $N$ da far crescere disaccoppiata dai byte-attivi. Decomporlo ricorsivamente dà un **albero di foglie azionabili**; a ogni foglia la mossa fuori-schema è la stessa famiglia — **fusione lossy $k\to1$** — applicata ad assi diversi (pesi, tempo, traiettoria). Le due foglie nuove e cheap che ne escono: **(e) stochastic-rounding self-averaging** (sfrutta che generare è una traiettoria) e **(b) hash-weights ⊗ ternario** (il tuo esempio, spinto). Entrambe testabili presto, una variabile, senza toccare la rampa run-6.
