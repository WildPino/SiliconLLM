# Dopo481: capacità condizionale, geometria delle decisioni e costo completo

5 ottobre2026. Goal originale ATTIVO/INCOMPLETO. Questo documento separa
misure ammesse, deduzioni algebriche e lavoro proposto. Non è un nuovo test.
[Risultato481](METH_481_GEOMETRY_RESULT_20261005.md).

## 1. Stato visto nel complesso

La tesi resta: grande capacità conservata in RAM, nucleo riutilizzabile piccolo,
poche funzioni consultate per token. Il numero di esperti deve aggiungere
capacità utile; duplicare pesi o ridurre solo il router non completa il metodo.

| Problema | Evidenza disponibile | Parte mancante |
| --- | --- | --- |
| Trasferire capacità utile a funzioni condizionali |123/183: Qwen piccolo,128->1280,scelta dei figli causalmente utile;369: identità Switch utile |Trasferimento conveniente completo a scala grande,nuovi n distinti utili |
| Eseguire il pretrained originale |Due scale Switch originali qualificate in C |Artefatto trasformato con qualità fresca e velocità sullo stesso oggetto |
| Scegliere esperti con poco lavoro |Supervisione479 completa;478 e480 varianti implementate e respinte |Geometria economica con decisioni sufficienti sul dominio reale |
| Conservare il peso dell'esperto |479: prodotto delle masse condizionali riproduce pF32;480/481 mostrano errori separati |Rappresentazione della massa economica,qualificata e calibrata |
| Abbassare il costo del modello completo |458: costi misurati per componente;453: qualità locale della precisione |Core/head/FFN convenienti che si compongano senza perdere qualità |
| Scalare n con RAM/LUT/DRAM |373: costo FULL limitato64->256; positivi piccoli precedenti |Molti esperti distinti,qualità e traffico fisico a n grande/altre famiglie |

I positivi non si sommano tra artefatti:123/183 qualità BF16 e127 velocità
FP32 appartengono a oggetti diversi.477-R1 mostra cambi di predizione nonostante
piccolo errore FFN: RMS locale non sostituisce generazione e task freschi.
Nessuna nuova prova su100B effettivi deriva dai soli header disponibili.

## 2. Che cosa481 ha chiarito

Il secondo momento fallisce già con tutti i punteggi originali:238465/238872
probabilità oltre1%,232166 troppo alte,80993 sopra1. Il drift nativo osservato
è al massimo3.86e-8 log-unità, contro errori del riferimento fino a20.24.
Il difetto del riferimento non è spiegato da quel drift.

Per A2=mu+log(n)+log(1+c1+v/2),la derivata rispetto a v è positiva. Abbassare
v peggiora ogni witness di sovraprobabilità mantenendo gli altri termini fissi.
Una troncatura PSD di questo riferimento ha dunque la direzione sbagliata su
quegli input. Chiusa questa specifica strada; nessun fit del rango su quel
presupposto. Non è una confutazione di ogni modello della massa o del MoE.

Anche le decisioni sono ora ricondotte a quantità precise:

```
D = max_R(s) - max_L(s)
D_candidate = D + epsilon_R - epsilon_L.
```

Gli errori comuni cancellano; conta l'errore differenziale confrontato col
margine.1,129,438nodi visitati soddisfano la condizione sufficiente stretta e
nessuno sbaglia. Pareggi e margini piccoli restano presenti.480 sbaglia spesso
già ai primi livelli; questa è evidenza di insufficienza della ricetta specifica,
non prova che l'ottimizzatore abbia trovato l'ottimo o che ogni gerarchia fallisca.

## 3. Geometria: un gruppo non è automaticamente una semiretta separabile

Nel modello ideale con s_e=w_e^T x,la regione dell'esperto e è

```
C_e = intersection_f { x : (w_e-w_f)^T x >= 0 }
```

con le disuguaglianze/tie key appropriate sui confini. Ogni C_e è poliedrale
convessa; la regione di un gruppo è l'unione dei C_e del gruppo e può essere
non convessa. D è lineare a tratti: una sola testa affine non rappresenta
automaticamente la frontiera tra due gruppi. L'ordinamento/partizionamento
delle funzioni è una variabile geometrica sostanziale.

Le formule sono in aritmetica reale. Il routing originale usa dot eF32 con
arrotondamenti qualificati; non si sostituisce quel programma con una presunta
uguaglianza BYTE delle forme ideali. Sul dominio finito possiamo verificare
etichette/margini del programma reale, poi qualificare l'export fisico.

Una gerarchia bilanciata legge O(log n)teste ma ne conserva O(n). Questa è
contabilità della rappresentazione, non un teorema di qualità o di ricerca
sublineare per pesi/dati arbitrari. Il lavoro economico dipende dalla geometria
effettiva delle regioni e dall'informazione necessaria per distinguerle.

## 4. Massa: obiettivo distinto, con normalizzazione strutturale

Per un nodo g con figli L/R,definire masse originali Z_L/Z_R e

```
q_R = Z_R/(Z_L+Z_R), q_L = 1-q_R
p(e) = product_along_path(q_selected_child).
```

Questa identità è già verificata in479 contro il pF32 originale. Un modello
condizionale normalizzato a ogni nodo genera una distribuzione valida sulle
foglie in aritmetica reale. Una testa della decisione e una testa della massa
possono essere diverse: il figlio con più massa non è necessariamente quello
che contiene il punteggio massimo. La scelta del percorso e il suo peso hanno
quindi due target distinti; non assumere che la scelta massimizzi il p foglia
del modello approssimato.

Con log-errori delta_l sulle probabilità del percorso,

```
log(p_candidate/p_source) = sum_l delta_l.
```

Il budget non si rinnova a ogni livello. Per il riferimento ideale,garantire
abs(sum delta)<=min(log1.01,-log.99) è sufficiente per1% relativo. Ripartirlo
uniformemente su7livelli richiede circa0.00142log-unità per livello; è una
condizione sufficiente prudente,non un'accuratezza già ottenuta. La verifica
fisica deve includere target pF32,coefficienti arrotondati e operatori C.

Le759masse figlio nulle della supervisione non vanno tolte o sostituite con
epsilon. Target di probabilità/BCE consentono0 e1; le formule devono gestire
i limiti esplicitamente. Il percorso del vero massimo contiene massa positiva.
Sigmoid e prodotti finiti richiedono comunque qualifica di overflow/underflow/
arrotondamento; validità della probabilità non equivale a fedeltà o qualità.

Il riferimento quadratico usa covarianza uniforme dei punteggi. Invece,per
forme lineari,la curvatura del vero log-partition è la covarianza dei pesi
ponderata dal softmax e dipende da x. Non si trasferisce automaticamente una
buona approssimazione dell'una nell'altra. Questa deduzione non autorizza un
nuovo sweep di rango o una promessa di piccolo rango.

## 5. Informazione, RAM e tempo: tenere tutti i termini

```
RAM_used = B_core + n*B_expert + B_router(n) + B_state + B_workspace
T_token = T_core + T_head + T_router(n) + T_selected_functions + T_other
```

Le durate includono i rispettivi accessi reali alla memoria; un eventuale
termine DRAM separato deve evitare di contarli due volte. Il vincolo finale
resta20ms/token accettato per50/s,nel contesto e hardware dichiarati. Un dato
warm non dimostra un accesso DRAM cold conveniente.

n è circa lineare nella RAM disponibile soltanto tenendo fissi formato,
dimensione per esperto,core e overhead.100B contro10B significa circa10volte n
solo se questi termini restano comparabili; lo scaling della larghezza o delle
precisioni cambia il rapporto. Capacità del donor,parametri distinti memorizzati,
parametri attivi e capacità causalmente utile sono quantità diverse.

Identificare e richiede al massimo log2(n)bit di identità; questo non limita il
costo del calcolo che produce quei bit. Se la profondità cresce,errori di scelta
condizionali q_l hanno union bound sum_l q_l,sui rispettivi prefissi corretti.
Più scelta richiede esposizione dei nuovi esperti e controllo degli errori dei
livelli; entropia/uso bilanciato da soli non provano funzione utile.

Due forme per7nodi a larghezza768 leggono10752coefficienti/43008B vettoriali,
prima della memoria di controllo. Se sono realmente affini,i14bias costano
altri56B;224B di header erano solo un'ipotesi. Conteggiare inoltre indici,
allineamento,operatori sigmoid e stato. Memoria totale delle due teste circa
2*(n-1)*(d+1)coefficienti: il traffico attivo piccolo non significa archivio
router piccolo. Nessun modello di questo tipo è ancora fitted/timed/qualificato.

Precisione importante rispetto al margine,non al solo errore medio. Se la
perturbazione fisica della decisione è inferiore al suo margine,il segno è
stabile; gli input vicini alla frontiera e i pareggi richiedono un trattamento
esplicito. La LUT va qualificata insieme alla geometria concreta e ai suoi
pesi; non basta la correttezza del kernel su dati sintetici.

## 6. Correzione di una scorciatoia nel quadro precedente

458 misura1.445771%/2.278304% del tempo per le sole MATRICI di scoring router.
Massa/softmax/controllo stanno anche nel residuo. Il precedente richiamo480 a
'router removal' era troppo ampio: il limite Amdahl~1.015/~1.023 vale eliminando
solo quel lavoro misurato,non tutto il routing. Non abbiamo qui la frazione
isolata di tutto il routing. Questa distinzione corregge la deduzione futura,
senza modificare i record sperimentali originali.

Resta vero che dense/control40..44%,expert matrices24.86..27.09% e head~6%
occupano parti rilevanti dei profili458. Dense/control aggrega organi diversi;
non è una misura del solo 'core target'. Il profilo completo limita quanto
una vittoria locale possa migliorare il tempo finale. Routing serve allo
scaling di n; il percorso end-to-end deve risolvere anche gli altri termini.

## 7. Strade aperte e scelta del prossimo lavoro

| Strada | Nuova variabile sostanziale | Rischio da risolvere |
| --- | --- | --- |
| Decisione diretta + massa condizionale normalizzata |Separare sign(D) dal target q;percorso probabilistico valido |Separabilità/curvatura,errore cumulato e quantizzazione |
| Gerarchia/feature comuni apprese per la funzione finale |Distillare scelta,funzione e core insieme;nuovi residui condizionali |Costo delle feature/core,conservazione della conoscenza e utilità dei figli |
| Partizionamento orientato alle regioni effettive |Cambiare geometria dei gruppi con vincoli di costo/esposizione |Costo di costruzione,facce esposte,massa non locale e generalizzazione |

La prima strada offre una domanda iniziale verificabile senza un altro fit
dei prototipi: i12nodi radice,necessari per qualsiasi percorso del tree attuale,
ammettono una decisione affine con margine fisicamente utilizzabile sul dominio
di sviluppo completo? Iniziare da tutti i root conserva ogni banco e classe;
è un prerequisito dichiarato,non una selezione dei nodi facili. Nessuna massa
appresa prima di aver giustificato questa prima classe di decisioni.

Prossimo passo operativo: preparare UN protocollo/code/runtime binding completo
di fattibilità affine dei root,non una griglia di seed/step/K/ranghi. Usare
solo sviluppo per costruzione,tenere la consumedval per valutazione dichiarata,
con ogni tie/ID/esposizione. Un possibile problema è massimizzare il margine
minimo y_i*theta^T[x_i,1] con norm1(theta)<=1,per ciascun root. Questa norma
fissa la scala del problema; il margine utile deve poi dominare l'errore del
formato fisico dichiarato. Coefficienti/ottimo/dual/condizionamento/costo pagati.

Un solver che dichiara INFEASIBLE o produce un residuo piccolo non fornisce
da solo una prova algebrica. Per parametri illimitati,un residuo dual nonzero
non certifica Farkas. Con norm1 limitata,un residuo r dà un limite tramite
norm_inf(r),incluse incertezze numeriche;si può certificare un limite sul margine
utile. PASS richiede verificare le disuguaglianze su TUTTI gli input previsti
e la perturbazione di export. Esiti entro tolleranza/budget sono INCONCLUSIVI,
non impossibilità di ogni geometria. Un witness esatto,quando disponibile,
va verificato separatamente. Non promettere un certificato prima di costruirlo.

Tetto prospettico600s/4GiB/64MiB,da prezzare prima del freeze;dipendenze del
solver e controllo indipendente devono essere completamente dichiarati prima
di qualsiasi import numerico. Nessun482programma/protocollo scientifico è
ancora congelato o registrato. Non lanciare un LP tramite un vecchio runner.

Se il margine affine non è utilizzabile,fermare questa variante e progettare
la feature/gerarchia con un budget completo prima di fit. Se è utilizzabile,
valutare massa condizionale e round-trip C per quella geometria. Un risultato
locale favorevole resta un prerequisito: integrare il candidato in UN artefatto
completo e misurare prediction,generazione e task su traiettorie proprie fresche,
qualità donor-relativa e SAME>=50 prima di espandere n. Il prossimo piano deve
includere core/head/FFN e utilità causale dei nuovi esperti,oltre al router.

Non è autorizzato da questo documento un fit lungo,uno sweep o una nuova cattura
del donor. Le attività entro il goal e le risorse esistenti non richiedono una
nuova conferma dell'utente;serve completarne la specifica prima dell'esecuzione.
