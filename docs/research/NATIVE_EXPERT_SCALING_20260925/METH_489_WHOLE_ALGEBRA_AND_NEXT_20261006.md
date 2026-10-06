# Dopo489: costo completo, margini, massa e informazione conservata

6 ottobre 2026. Goal ATTIVO/INCOMPLETO. Questo è un riesame del progetto,
con deduzioni sulle misure ammesse e un prossimo lavoro proposto. Non contiene
nuova inferenza, un nuovo fit o una misura di prestazione.
[Risultato completo489](METH_489_SHARED_INTEGER_BACKEND_RESULT_20261006.md).

## 1. Cosa abbiamo e cosa manca

La tesi rimane un nucleo compatto e riutilizzabile, con molte funzioni distinte
utili in RAM e poche funzioni consultate per token. Occorre costruire il percorso
dal pretrained a questa geometria, conservando informazione e qualità.

| Obbligo | Evidenza disponibile | Vincolo ancora aperto |
| --- | --- | --- |
| Capacità condizionale utile |123/183: figli Qwen piccoli utili;369: identità Switch utili |Molti n distinti dopo trasferimento conveniente;372 ha risultati misti |
| Esecuzione fedele del pretrained |Due scale Switch originali in C;489 output esatti CPU/GPU |Artefatto trasformato conveniente e generalità tra famiglie |
| Qualità e velocità insieme |489 CPU n128 conserva la qualità del source sul campione e passa warm50 prose |Non prova ancora la tesi del core compatto, scaling utile o coldDRAM |
| Calcolo condiviso esatto |489:169 operatori GPU qualificati |Layout per operatore sincronizzato FAIL su entrambe le scale |
| Scelta CPU economica |Supervisione479; varianti478/480 ammesse e respinte |Geometria economica che distingua le funzioni utili sul dominio reale |
| Peso della scelta |Massa nativa479;481 esclude la formula fissa al secondo momento |Rappresentazione economica della massa con errore controllato |
| n grande / LUT / memoria |373: costo FULL limitato64->256; legge fisica del formato484 |Vincitore E massa, accessi DRAM effettivi, nuove funzioni utili, altre famiglie/100B |

I due Switch hanno pesi e affinità CPU diversi: le differenze di qualità o tempo
non sono una curva causale del solo n. Il successo BF16 Qwen123/183 non si combina
con la velocità FP32 di127. Gli header QwenNext397 non sono pesi/qualità a100B.
Le precedenti evidenze Giga restano riutilizzabili, senza riavviare automaticamente
il percorso donor-adaptation congelato.

## 2. Il budget delle fasi elimina due scorciatoie

Per ogni source, K è il numero di ID prose accettati e

```
P = T_encoder + T_crossKV
D = T_decoding
R = K/(P+D)
50 ID/s richiede P'+D' <= K/50.
```

Sommare le medie dei primi tre tempi di ogni caso, includendo quelli rifiutati.
[Derivazione numerica](meth489_primary_phase_algebra.json), SHA256
`1a368ee25e650ff0f8d47bfcc026ad9e01a4d6932d7d3d0c6a526c8f45c02202`,
usa somme esatte di razionali `Fraction.from_float` sui tempi già ammessi,
verifica la somma delle fasi e conserva tutte le ipotesi. Non è profile1.

| Source/backend | P (s) | D (s) | K/50 (s) | R con P'=0,D'=D | R con D'=0,P'=P |
| --- | ---: | ---: | ---: | ---: | ---: |
| n128 CPU |6.89470617 |5.16856960 |13.24 |128.0819 |96.0157 |
| n128 GPU |16.00081237 |16.40157437 |13.24 |**40.3620** |**41.3729** |
| n256 CPU |7.69659557 |5.92343550 |9.80 |82.7223 |63.6645 |
| n256 GPU |18.56060723 |18.52458660 |9.80 |**26.4513** |**26.4000** |

Per la GPU, neppure una delle due fasi resa gratuita basta a50 se l'altra fase
e K restano invariati. Sul256, anche con prefill gratuito il decoding deve
scendere almeno al52.9027% del costo osservato; con decoding gratuito il prefill
deve scendere al52.8000%. Sul128 i due limiti sono80.7240% e82.7458%.

Sono limiti condizionati alle altre quantità invariate. Una trasformazione può
cambiare entrambe le fasi, cache, clock o K; questi calcoli non escludono quella
possibilità. Escludono la promessa di raggiungere50 cambiando solo una fase
alle condizioni indicate. Non attribuiscono causalmente il ritardo a cuBLAS,
sincronizzazione, CPU o interferenza: profile1 non è ammesso.

Il backend impiega169646/163101 chiamate GPU per un giro dei96casi sulle due
coorti, coerentemente con il denominatore SUM delle medie per caso; non è il
totale di tutte le ripetizioni/warmup dell'apparato. Il numero esatto di chiamate
è60+24S+85T per caso; la formula è stata auditata. Una futura fusione
potrebbe cambiarlo, ma un rapporto tra chiamate non è un rapporto tra tempi.
Non avviare uno sweep di batch/tile/fusioni di questa variante chiusa.

Sulla CPU, il128 ha1.17672423s di margine nel budget puntuale13.24s; il256
richiede invece almeno3.82003107s di riduzione, circa28.05% del costo completo,
se K resta490. Non sono nuovi limiti bootstrap. I diversi K (662contro490)
contribuiscono alla differenza di rate quanto i tempi: checkpoint e filtri
generativi producono output diversi. Non attribuire questa differenza al solo n.

## 3. Margine e massa: tre limiti elementari, indipendenti da una ricetta

Per punteggi reali finiti s, definire

```
L(s) = log(sum_j exp(s_j))
e = argmax_j s_j
Delta = s_(1)-s_(2)
p_e = exp(s_e-L(s)).
```

Se un'approssimazione t soddisfa `max_j |t_j-s_j| <= eps`:

1. `|L(t)-L(s)| <= eps`. Infatti ogni esponenziale cambia entro fattori
   exp(-eps),exp(eps); lo stesso vale per la somma e poi il logaritmo.
2. Il vincitore unico resta invariato se `Delta > 2*eps`: il primo può perdere
   eps e il secondo guadagnare eps. Pareggi e casi sul confine richiedono la
   tie key nativa, non questa condizione sufficiente stretta.
3. Per lo stesso ID, `|log(p'_e/p_e)| <= 2*eps`, quindi l'errore relativo è
   al massimo `exp(2*eps)-1`. Per garantire1% basta
   `eps <= log(1.01)/2`, circa0.00497517.

Il limite del normalizzatore non contiene n. Aumentare n non obbliga la qualità
a scendere. Tuttavia non garantisce margini grandi né un'approssimazione uniforme
di tale precisione a costo piccolo. È proprio questa la difficoltà geometrica.
Un errore medio piccolo non soddisfa questi limiti uniformi.

Queste sono nostre deduzioni in aritmetica reale. Il router effettivo usa pesi
F32, dot F64 e cast F32, poi operatori finiti per exp/somma/probabilità. Occorre
aggiungere il budget di arrotondamento e i confronti col programma reale.
Il router sorgente NON quantizza direttamente x in A16.489 lo lascia invariato,
quindi non dimostra ancora una LUT approssimata corretta.

## 4. Separare la scelta dalla massa e controllare l'informazione persa

Per un gruppo g, `h_g=max_{e in g}(s_e)` decide se contiene il vincitore;
`L_g=log(sum_{e in g} exp(s_e))` descrive la massa. Il gruppo con più massa
può non contenere il massimo. Una gerarchia deve rappresentare entrambe le
quantità. Una scelta greedily basata solo sulla massa non conserva automaticamente
l'argmax piatto. Calcolare tutte le masse esattamente può ancora costare O(n).

Per il vincitore ideale definire `g=L(s)-max(s)>=0`, allora `p_e=exp(-g)`.
Predire un gap nonnegativo evita p>1, ma NON dimostra l'accuratezza della massa
né la normalizzazione di una distribuzione completa sulle foglie. Analogamente,
sigmoid e prodotto delle masse condizionali garantiscono una distribuzione in
aritmetica reale, ma non la fedeltà al source.480/481 lo rendono concreto.

Con probabilità condizionali di un percorso di profondità d:

```
log(p_candidate/p_source) = sum_l log(q_candidate,l/q_source,l).
eta = min(log(1.01), -log(0.99)) = log(1.01)
sum_l |log(q_candidate,l/q_source,l)| <= eta
```

è sufficiente per1% relativo. Un budget uniforme eta/d è conservativo: circa
0.00142148 per sette livelli,0.00124379 per otto. Con n moltiplicato per10,
la profondità bilanciata aumenta di circa log2(10), non di10volte; storage e
apprendimento dei nodi comunque crescono. Non si rinnova l'intero budget a ogni
livello. Le masse0/1 dei rami vanno conservate e gestite esplicitamente.

Per l'uscita selezionata e probabilità nonnegative:

```
||p_hat*f_ehat(x) - p*f_e(x)||
 <= |p_hat-p|*||f_e(x)|| + p_hat*||f_ehat(x)-f_e(x)||.
```

Scelta e peso perdono informazione in modi diversi. Un ID diverso può essere
innocuo solo se le funzioni coincidono su x; lo stesso ID con peso errato può
cambiare lo stato.477-R1 mostra190/3065 argmax cambiati con2.83% RMS FFN: nessuna
metrica locale sostituisce predizione, generazione propria e task del risultato.

La composizione su stati propri introduce inoltre un secondo termine esatto:
`||F_candidate(x')-F_source(x)|| <=
||F_candidate(x')-F_source(x')|| + ||F_source(x')-F_source(x)||`.
Un controllo locale su x del source non controlla automaticamente nessuno dei
due termini per tutti gli x' raggiunti dal candidato. Quantizzatori, tie e
accettazione impediscono di presumere una costante globale di Lipschitz utile.
Servono margini/moduli di perturbazione qualificati o la verifica end-to-end
su stati propri; questo è il motivo matematico per cui i positivi locali non
si compongono automaticamente in qualità conservata.

## 5. n cresce con RAM solo sotto prerequisiti espliciti

Per questo formato originale,484 verifica

```
B_file(n) = 265878016 + 56844288*n byte
B_machine = B_file + B_OS + B_runtime + B_state + B_workspace.
```

n conta gli ID per banca, su12banche.80GiB consentono al massimo1506ID per banca
nel solo file; il limite utilizzabile è inferiore. Non sono1506funzioni addestrate
utili né una misura di memoria residente o una dimostrazione100B.

100B contro10B comporta circa10volte n soltanto se core, larghezza, formato e
dimensione per esperto restano comparabili. H(ID)<=log2(n) limita i bit di
identità, non il costo di calcolarli o la capacità informativa delle funzioni.
La media di esposizione con Nquery/k scelte è Nk/n: nuovi residui da apprendere
richiedono dati sufficienti; copiare pesi esatti ha un costo informativo diverso.

LUT CPU, normalizzazione, bytes attivi e accessi DRAM vanno misurati sullo stesso
artefatto a n grande. Payload mappato, cache calda e massa di coefficienti letti
logicamente non sono misure di traffico DRAM fisico. Gli interventi di utilità
devono distinguere nuove funzioni utili da duplicazioni o alias.

## 6. Prossima incertezza: teste della massa separate dai supporti del massimo

**Decisione attuale:** conservare CPU n128 come riferimento qualificato; chiudere
il layout GPU per operatore; tornare alla rappresentazione condizionale. Nessun
altro test C/GPU è giustificato dai risultati489 prima di un nuovo budget completo.

Prossima linea proposta, NON implementata né osservata: una testa della massa
condizionale normalizzata separata dalla testa del vincitore.480 usa supporti
fissi2/root16 e fallisce;481 usa una formula al secondo momento e fallisce.
La nuova variabile è imparare direttamente q_R=Z_R/(Z_L+Z_R), senza riusare
supporti del massimo o quella formula come modello della massa.

Prima inquiry limitata ai12root dell'originale n128, con TUTTO lo sviluppo479,
ALLconsumedval e tutti i libri/ruoli/occorrenze originali. È un filtro necessario
per UNA proposta di teste affini sigmoid con budget uniforme a sette livelli;
non risolve la scelta del vincitore e non misura ancora il costo di una LUT.
La sigmoid rende validi i due rami in aritmetica reale; la qualifica fisica resta
necessaria. Una buona root non dimostra i restanti126nodi per banca o il percorso composto.

Il prossimo atto è preparare e congelare UN protocollo/controller/audit, prima
di qualsiasi fit. Riutilizzare479 senza replay di capture/source e senza nuovi
split. Usare solo sviluppo per i coefficienti; consumedval è già consumato,
mai una nuova prova held-out. Conservare target0/1 senza epsilon o esclusioni.
Specificare deterministicamente loss/ottimizzatore/iterazioni/rounding, tutti
i12root e la verifica del budget `eta/7` sul ramo del vero vincitore; nessuna
selezione di root, restart o griglia dopo i risultati. L'audit deve rederivare
target/probabilità/errore fisico dai dati e coefficienti, senza importare il fit.

Budget prospettico massimo10min main/1GiB parent,5min audit/512MiB,64MiB output,
CPU con BLAS1, nessuna T4 o nuova risorsa; controllare l'eligibilità della memoria
e i dati effettivi durante la preparazione. Se non compatibile, arrestarsi prima
del fit e documentare il limite. Ogni prima interruzione resta registrata.

**Decisione che può cambiare:** se questa ricetta passa le root, preparare la
composizione di tutte le masse e la geometria del vincitore, poi round-trip C e
qualità fresca/SAMErate. Se fallisce, chiudere il requisito uniforme di QUESTA
ricetta senza estenderlo a ogni geometria o budget nonuniforme. Un timeout resta
inconclusivo e non autorizza un aumento equivalente del tempo. Non promuovere
sigmoid valida a massa fedele, né massa fedele a utile scaling-n.

## 7. Ripresa e criteri del prossimo riesame

489 è completo:107 nuovi blocchi,192casi/3456file, ALL6audit/ALL5admission;
nessun processo scientifico proprio attivo. Tutte le precedenti esecuzioni,
interruzioni e namespace restano immutabili. Engine e lavoro estraneo preservati.

Riprendere preparando la sola eligibility/protocollo delle teste della massa
separate. Il prossimo riesame deve avvenire appena tale filtro cambia una
decisione, prima di un'estensione a tutti i nodi. Restano insieme: trasferimento
pretrained conveniente, funzioni distinte utili, winner E massa CPU/LUT, DRAM,
qualità fresca e50/s sullo stesso artefatto, generalità tra famiglie/scale.
