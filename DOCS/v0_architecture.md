# Silicon Sequence Compressor V0
**Architecture Specification & Rationale**

Questo documento cristallizza le specifiche del nucleo estrattivo V0 di SiliconLLM, misurato e validato attraverso test empirici intensivi (Phase 8). 

## 1. Filosofia: Compressore Predittivo Locale
Il Silicon Sequence Compressor V0 non è (ancora) un Large Language Model nel senso classico. È un formidabile **compressore predittivo locale** guidato dalla termodinamica della CPU. 
Non possiede layer di attenzione quadratici, ma si affida a un'architettura completamente cache-native, discreta, basata sull'integrazione di percorso di simboli mappati nello spazio.

## 2. Architettura V0: Specifica Tecnica

Il core implementato in `src/silicon_v0.c` definisce quattro componenti fondamentali:

1. **Topologia Spaziale (G128)**: 
   - Una griglia circolare di `128` blocchi AVX2 (`4096` celle fisiche). 
   - Espandere oltre 128 blocchi causa un rallentamento lineare senza alcun guadagno nell'estrazione di informazioni. G128 è lo *sweet spot* L1.

2. **Encoding Fisico (Single Random Binary Codebook)**:
   - Ogni byte (simbolo) in input viene decodificato in un vettore randomico binario di `32` byte (`0` o `255`).
   - Il vettore viene iniettato *sempre interamente* in un singolo blocco della griglia per mantenere compatta la geometria del segnale.
   - *Nota:* Iniettarlo in blocchi distribuiti causava sovrapposizioni premature distruttive (-4% di accuracy empirica).

3. **Iniezione Temporale (T3 Shift-Window)**:
   - Ad ogni tick, una history buffer (M4) di `16` token storici (finestra T3) viene re-iniettata nella griglia.
   - Ogni token storico viene posizionato (shiftato) nello spazio: il token più recente all'indice spaziale `0`, il precedente a `8`, poi `16`, ecc.

4. **Dinamica: Temporal Path Integration con Dissipazione**:
   - Una volta iniettata, la griglia subisce *4 step* di diffusione spaziale con saturazione (`sat_add`).
   - Prima della diffusione, un *damping termodinamico* dimezza l'energia del campo (`avg` + shift). 
   - Poiché ogni tick reinietta l'intera finestra T3 sfalsandola nel tempo, *lo stesso simbolo attraversa coordinate spaziali diverse nel tempo*. La memoria persistente della Wave non fa solo decadere i vecchi token, ma accumula **i residui e le traiettorie spaziali** di queste iniezioni successive, creando un *temporal discounting fisico*.

5. **Readout: Lane-Aware Pooled 32D**:
   - Al momento dell'estrazione, la griglia spaziale viene *poolata* (sommata) preservando l'identità delle 32 "lane" del codebook.
   - Non restituisce mai somme energetiche scalari cieche. L'output è sempre un vettore a 32 dimensioni `float`/`double`. 

## 3. Fallimenti Storici (Cosa il Silicio ha Rifiutato)

La V0 è nata escludendo rigorosamente i design fallimentari durante i tribunali sperimentali:

- **Rifiuto dell'Intensità ASCII**: Inizialmente mappavamo i byte (es. 'a' = 97) come energia fisica (97 saturato). La griglia produceva pattern indistinguibili (-40% sotto il Bigramma). Il Silicio esige collisioni ortogonali categoriali, non scale d'intensità spaziali.
- **Rifiuto del Readout R2-Sum Cieco**: Estrarre una singola "somma di inchiostro" da ogni canale distruggeva l'identità del Codebook (-3% accuracy). Le lane del codebook vanno preservate nel readout.
- **Rifiuto del Codebook Distribuito**: Accendere 4 blocchi sparsi per simbolo (-4% accuracy). La wave locale mischia il segnale in un raggio stretto; una distribuzione iniziale sparpagliata distrugge i pattern per *crosstalk* precoce prima che il layer lineare possa valutarli.
- **Linear Layer Congelato**: La V0 estrae feature 32D, ma non si allena internamente. Non imponiamo una forma "Machine Learning" (Ridge, NLMS) al nucleo; la lettura entropica reale viene delegata all'harness esterno.
