# Dopo491: trasferire la funzione, misurare il costo completo

6 ottobre 2026. Goal ATTIVO/INCOMPLETO. Questa rivalutazione separa risultati
ammessi, deduzioni matematiche e proposte. [491](METH_491_MASS_INTERVAL_RESULT_20261006.md)
è completo; il successivo filtro492 è solo un'eligibility da implementare e
congelare prima di osservarne gli esiti. Nessun nuovo modello deriva dal piano.

## 1. Il traguardo e lo stato reale

Vogliamo trasformare un pretrained in un nucleo conveniente più molte funzioni
condizionali utili, con qualità rispetto al donor e almeno50 ID accettati/s
sullo STESSO artefatto. n deve poter crescere con la RAM senza far crescere
proporzionalmente lavoro e traffico per token. La conoscenza trasferita, la
scelta delle funzioni e la loro esecuzione costituiscono un unico procedimento.

| Oggetto | Evidenza disponibile | Parte ancora mancante |
| --- | --- | --- |
| Esperti con scelta utile |123: Qwen piccolo128->1280, qualità esterna;183: allineamento dei child causalmente utile;369: identità Switch utile |Stesso risultato dopo trasferimento conveniente, grande n e altre famiglie |
| Geometria di routing conveniente |54: product-key sintetico rank64, .4405ms/24layer a273408;56: product-key appreso128 con ritenzione BF16 |Trasferire un donor arbitrario in quella struttura; CPU LUT e qualità/costo integrati |
| Esecuzione del donor |Due scale Switch C qualificate;489: CPU128 warm prose54.8773/s, bootstrap quinto percentile53.4759 |Artefatto trasformato e scalabile; prestazione nei contesti dichiarati |
| Trasferimento conveniente di funzioni |453: qualità locale full-width;477-R1: calibrazione dell'effetto sulla head |Costo conveniente con qualità sul proprio percorso; ricette di riduzione già fallite |
| Fedeltà della massa |490: errore della ricetta, prima dell'export;491: quattro limiti necessari sulla norma affine |Rappresentazione o trasferimento funzionale che superi i gate completi |
| RAM e costo fisico |373: costo FULL limitato64->256; conteggi dei byte originali |Molti esperti distinti utili, residenza/accessi reali/DRAM, dati freschi a n crescente |
| Generalità |Giga/Ling/Granite: varianti e fallimenti specifici;397: header79.674B |Un percorso conveniente valido su più famiglie e valori realmente elaborati a100B |

54 misura keys sintetiche;56 conserva il core dense del donor BF16. Non
possiamo sommare il timing54 alla qualità56/123 come se fossero lo stesso
artefatto. CPU128 e CPU256 hanno pesi e worker diversi: non isolano l'effetto
di n. Gli header397 non sono pesi100B trasferiti. I positivi rimangono utili
nel loro perimetro, ma nessuno conclude da solo il goal.

## 2. Che cosa cambia491

490 non raggiunge il budget su236613/238872 stati; questi fallimenti restano
nell'ideale della funzione appresa. Il drift numerico massimo osservato
2.83271e-7 è molto inferiore all'errore mediano .109139. 32 passi BCE non
costituiscono una prova sull'ottimo della classe.

491 aggiunge un vincolo indipendente da quella traiettoria. Sul dominio di
sviluppo fissato, quattro root affini che rispettassero eta richiederebbero
norme L1 delle pendenze di almeno98,901,154;63,828,375;19,233,036;47,917,737.
Il bias è eliminato mediante un vincolo già verificato. Sono limiti necessari
conservativi, rispetto alle norme osservate490 di circa23.40,23.62,12.18,10.09.
Le altre otto inquiry rimangono inconclusive; tutti i residui sono nonzero.

Quindi prolungare lo stesso fit nella regione di norme ordinarie non può
risolvere quel gate sulle quattro root. Coefficienti illimitati rimangono
matematicamente aperti. La fattibilità fisica F32 non è stata confutata.

Per z=a^T x+b, in tutto R768:

```
sup_{||delta||infinity<=h} |z(x+delta)-z(x)| = h ||a||1.
```

Questa identità spiega la geometria della norma, ma la direzione peggiore può
uscire dal dominio degli stati del modello. Non prova instabilità sulla sua
traiettoria, né perdita di qualità. Un cambio affine invertibile di coordinate
non cambia la classe delle funzioni affini in x; può cambiare norme coordinate
e implementazione. Feature nonlineari, partizione diversa e trasferimento
della funzione pesata cambiano invece il problema rappresentato.

## 3. Il vero oggetto da trasferire

Nel riferimento matematico ideale, una layer top1 seleziona:

```
e(x) = argmax_e s_e(x)
p_e(x) = exp(s_e(x))/sum_j exp(s_j(x))
Y(x) = p_e(x) F_e(x).
```

Il programma originale ha un ordine e precisioni specifici; questi oggetti
servono a progettare, mentre la fedeltà si verifica sul programma fisico.
Un target con decisione d(x) e funzioni G produce Yhat=G_{d(x)}(x). Per ogni norma:

```
||G_{d(x)}(x)-Y(x)||
 <= ||G_{d(x)}(x)-G_{e(x)}(x)|| + ||G_{e(x)}(x)-p_e(x)F_e(x)||.
```

Il primo termine riguarda l'effetto funzionale di una scelta diversa; il
secondo riguarda il trasferimento sull'ID corretto. Una differenza di ID
non implica sempre una differenza funzionale grande; un ID identico non
garantisce una massa o funzione corretta. Per una rappresentazione separata
G_e=phat_e Fhat_e, phat_e>=0:

```
||phat_e Fhat_e-p_e F_e||
 <= |phat_e-p_e| ||F_e|| + phat_e ||Fhat_e-F_e||.
```

Un modello della massa root entro log(1.01)/7 è una condizione SUFFICIENTE
scelta per una particolare composizione gerarchica. Non è un requisito
fondamentale del goal: il donor può essere trasferito direttamente attraverso
Y. Cambiare questa via richiede un nuovo protocollo, criteri funzionali e
qualità completa; non rende retroattivamente positivi480/490/491.

Per logits u e perturbazione delta, la relazione informativa esatta è:

```
p = softmax(u)
KL(p || softmax(u+delta)) = log(sum_j p_j exp(delta_j))-sum_j p_j delta_j.
```

Una costante comune nei logits non cambia la distribuzione. Per delta piccolo
il termine quadratico è .5 delta^T(diag(p)-p p^T)delta; resta un'approssimazione
locale, non un gate. 477-R1 ha verificato190/3065 argmax cambiati con2.83%RMS
FFN: RMS locale e qualità finale non sono intercambiabili. Stati propri,
generazione e task sul candidato completo rimangono indispensabili.

## 4. Vie aperte e decisione

| Via | Nuova variabile | Condizione per investirci |
| --- | --- | --- |
| Decisione+normalizzazione strutturate |Partizione/features/supporto che rappresentano winner e massa economicamente |Costo completo e fedeltà dimostrati; nessun nuovo sweep della stessa root affine |
| Product-key o gerarchia appresa congiuntamente |Score/student ed esperti acquisiscono la struttura del target |Trasferimento reale e core conveniente;54/56/123 sono precedenti separati |
| Trasferimento della funzione pesata su celle dell'ID |G_e approssima p_e F_e sul dominio dove e è consultato |Separare prima i casi di assorbimento esatto/approssimato dalla dipendenza residua dall'input |

Per score cartesiani s_(i,j)=u_i+v_j, winner e partizione fattorizzano:

```
argmax_(i,j) s_(i,j) = (argmax_i u_i,argmax_j v_j)
sum_(i,j) exp(u_i+v_j) = (sum_i exp(u_i))(sum_j exp(v_j)).
```

L'identità vale nella geometria fattorizzata; non dice che un router pretrained
arbitrario possieda quella struttura. Questa strada è già studiata53..56,
non è una nuova scoperta491. La tesi di una sola massa affine globale non
deve assorbire indefinitamente le risorse della ricerca.

**Scelta successiva:** un solo filtro senza fit per il trasferimento della
funzione pesata. Chiedere se p selezionata possa diventare una costante PER
ESPERTO sul dominio dove esso vince. È un dominio diverso dalla massa root
aggregata su molti esperti. L'eligibility492 usa extrema e rapporti razionali
esatti dei valori F32 originali, quindi può decidere la classe costante senza
Adam, LP o SVD. Il suo costo e arresto sono fissati nella
[eligibility492](METH_492_SELECTED_AMPLITUDE_ELIGIBILITY_20261006.md).

Questo filtro è giustificato solo come scelta del percorso di trasferimento:
una costante può essere assorbita in una mappa di uscita lineare in aritmetica
reale; un fallimento richiede dipendenza dall'input per preservare quel budget
di ampiezza. Non confuta una distillazione funzionale che cambi F, né decide
gli stati dove F è zero. Non riapre le vecchie ricette di riduzione FFN.

Il filtro non riduce la larghezza FFN o il core, e la decisione originale
resta una scansione dei punteggi finché non viene trasferita separatamente.
Un pass non conclude il trasferimento: occorrono export, winner economico,
funzioni convenienti e qualità completa. Un fail chiude questa classe senza
un sweep; il passo seguente deve essere progettare G_e(x) dipendente dall'input
con un budget complessivo, oppure cambiare score/partizione con apprendimento
congiunto. Nessun lungo addestramento è avviato prima di questa scelta.

## 5. Scaling: algebra dei byte, del lavoro e dell'esposizione

```
B_total = B_core + n B_expert + B_route(n) + B_state + B_workspace
T_accepted = T_core + T_head + T_route(n) + T_selected(k) + T_other <=20ms.
```

Per il formato originale B_file=265878016+56844288*n byte; n1280 richiede
73,026,566,656B per il SOLO file. Su80GiB il limite file-only è n1506, prima
di stati, workspace e sistema. Questo non è un limite del progetto: cambiare
la rappresentazione cambia B_expert. Non è una prova di residenza o banda.

Una gerarchia bilanciata1280 ha profondità massima11; l'incremento da128 non
è10x nella profondità. Le teste devono però rappresentare le decisioni e
masse necessarie. Per errori logaritmici per livello delta_l:

```
log(phat_e/p_e) = sum_l delta_l.
```

Gli errori possono cancellare, ma la somma dei valori assoluti dà una garanzia
sufficiente. Un budget uniforme su11 livelli è più stretto di uno su7.
Inoltre un indice e appartiene a n scelte, con entropia marginale H(E)<=log2(n);
questo conta informazione del selettore, non conoscenza nei pesi. La crescita
della RAM non crea da sola funzioni utili. Con esposizione uniforme l'esposizione
media per esperto è Nk/n; a n10x, stesso N e k, cala10x. La distribuzione reale,
il costo di trasferimento e gli esperti mai osservati devono essere espliciti.

## 6. Ripresa esatta e parti mancanti

491 main/verifier/finalizer e corollario sono completi; nessun nostro processo
scientifico attivo. Tutti i vecchi namespace, primi fault e byte preservati
restano immutabili. La prossima azione è implementare e congelare il filtro492
e il suo verificatore indipendente, preparare la binding fresca, poi eseguire
le sole invocazioni entro il preventivo.492 non è ancora implementato/osservato.

Ogni futura promozione richiede congiuntamente: funzioni pretrained trasferite
in un core conveniente; winner+ampiezza o funzione pesata coerenti; CPU LUT
fisicamente qualificata; qualità fresca sul proprio percorso;>=50 sullo stesso
artefatto; n distinto causalmente utile; costo DRAM reale; varianti applicabili
ad altre famiglie e scale. Goal ancora ATTIVO/INCOMPLETO.
