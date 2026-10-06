# Dopo490: separare ottimizzazione, geometria e informazione

6 ottobre2026. Goal ATTIVO/INCOMPLETO. [490](METH_490_ROOT_MASS_RESULT_20261006.md)
è completo e ammesso indipendentemente. Qui distinguiamo misure, deduzioni
algebriche e un prossimo filtro proposto. Il filtro non è ancora implementato
o osservato; nessuna nuova inferenza o prova di qualità deriva da questo testo.

## 1. Il progetto nel complesso

La capacità condizionale utile dipende da tre oggetti collegati: funzione
trasferita, decisione che la consulta e peso con cui entra nel modello. La
convenienza richiede anche un nucleo piccolo e una verifica sullo stesso artefatto.

| Problema | Evidenza ammessa | Vincolo ancora aperto |
| --- | --- | --- |
| Capacità utile negli esperti |123/183,Qwen piccolo128->1280;369,identità Switch utile |Trasferimento conveniente e utile a grande n,anche altre famiglie |
| Esecuzione del pretrained |Due scale Switch C qualificate;489CPU n128 warm prose54.8773/s |Artefatto trasformato,qualità fresca e costo misurato insieme |
| Geometria delle decisioni |Supervisione479;478/480 ricette respinte;482/483 margini inconclusivi |Decisioni economiche e fisicamente fedeli sul dominio reale |
| Massa condizionale |479 identità di composizione;481 formula respinta;490 ricetta affine respinta |Modello conveniente della massa con errore composto controllato |
| Costo completo |489 GPU per operatore qualificata ma lenta;CPU128 positivo |Costo core/head/esperti/memoria del risultato trasferito |
| Scaling con RAM |373 costo FULL limitato64->256;legge dei byte |Molti esperti distinti utili,CPU LUT e DRAM fisica,qualità a n crescente |

I positivi appartengono ai loro artefatti e contesti. CPU128/256 differiscono
per pesi e worker: le velocità non isolano l'effetto causale di n. Una qualità
locale o una distribuzione normalizzata non dimostrano traiettorie proprie utili.
I soli header397 non costituiscono un modello100B trasferito o valutato.

## 2. Il risultato490 individua la scala dell'errore

Budget root.00142148log-unità; errore mediano.109139; massimo3.96083.
L'export introduce al massimo2.83271e-7log-unità sul ramo scelto. Tutti236613
fallimenti persistono nell'ideale della funzione appresa. Questo identifica
l'errore della ricetta rappresentazione+ottimizzazione osservata; l'ottimo della
classe affine rimane ignoto. Le loss oscillanti e il gradiente prima dell'ultimo
update impediscono di attribuire a una classe intera il risultato di32 passi.

Anche l'errore medio sullo sviluppo è.189744: il difetto appare sui dati usati
per il fit e sulla consumedval. Questi dati consentono una diagnosi della ricetta;
una stima di generalizzazione richiede dati realmente freschi.

## 3. Geometria esatta della variabile da rappresentare

In aritmetica reale,per punteggi lineari s_e=w_e^T x e una partizione L/R:

```
A_G(x) = log(sum_{e in G} exp(w_e^T x))
g(x) = A_R(x)-A_L(x)
q_R(x) = sigmoid(g(x))
grad A_G = sum_e pi_{e|G} w_e
Hessian A_G = Cov_{pi_{e|G}}(w_e)
Hessian g = Cov_R(w)-Cov_L(w).
```

La differenza delle covarianze può avere segni diversi; una singola forma
affine ha Hessiana nulla. Una cancellazione o un dominio di stati particolare
può rendere l'approssimazione affine efficace. Queste identità non provano
che il dominio finito479 richieda una funzione non affine. Inoltre i target479
appartengono al programma F32/F64 originale: il modello reale è una lente
geometrica,la fedeltà fisica si verifica sul programma effettivo.

Il massimo di gruppo riguarda l'unione delle regioni poliedrali dei suoi esperti;
la massa riguarda la somma degli esponenziali di tutti i membri. La partizione
degli esperti influenza entrambe. Una gerarchia bilanciata fornisce una legge
O(log n) delle teste visitate quando queste teste sono sufficienti: il costo
algebrico della rappresentazione deve essere dimostrato insieme alla fedeltà.

## 4. L'informazione media e il vincolo uniforme sono problemi differenti

Per y=qR_original e r=sigmoid(z),con target0/1 mantenuti:

```
BCE(y,r) = H_Bernoulli(y)+KL(Bernoulli(y)||Bernoulli(r))
loss_weighted-H_weighted = sum_i omega_i KL_i.
```

490 ha ridotto la BCE in ogni banca; l'eccesso KL medio per banca resta positivo,
con media fra banche.05447155nats. Questa quantità non è KL della distribuzione
dei token del modello. La condizione sul percorso richiede invece ogni errore
di log-probabilità selezionata entro eta. Un errore grande su un peso omega_i
piccolo contribuisce poco alla loss media. La minimizzazione della BCE conserva
quindi un significato informativo,ma la scelta della loss e il gate uniforme
richiedono una giustificazione di composizione ulteriore.

Con livelli l lungo il percorso:

```
log(p_hat(e)/p(e)) = sum_l log(q_hat_l/q_l).
```

Gli errori firmati possono cancellare; il limite sulla somma dei valori assoluti
è sufficiente e conservativo. Un budget nonuniforme deve essere fissato e
verificato sul percorso completo. La root490 eccede perfino log(1.01) in221967
stati: questa è la scala osservata dell'errore,non una prova sull'assenza di
cancellazioni in teste future. La root da sola non crea il candidato completo.

Per la funzione del donor selezionata,qualsiasi norma:

```
||p_hat F_ehat(x)-p F_e(x)||
 <= |p_hat-p| ||F_e(x)|| + p_hat ||F_ehat(x)-F_e(x)||, p_hat>=0.
```

La qualità finale dipende da entrambi i termini e dagli stati successivi.
477-R1 mostra190/3065argmax cambiati pur con2.83%RMS FFN. Un futuro obiettivo
di distillazione funzionale può accoppiare scelta,massa e funzione con un
budget di byte/lavoro; il suo successo richiede qualità fresca sul proprio
percorso. Questa proposta resta distinta dal trasferimento già validato.

## 5. Prossimo filtro algebrico: intervalli della massa,prima di un altro fit

La prossima incertezza è la fattibilità utile della rappresentazione affine
rispetto al budget uniforme. Il nuovo target è il vincolo della MASSA;
482/483 trattavano invece etichette/margini della decisione. L'esecuzione490
misura una traiettoria BCE finita. Nessuno dei due risolve questo nuovo problema.

Per ogni stato di sviluppo i,sia q_i>0 la massa del ramo del vero vincitore,
t_i=+1 se il ramo è R e -1 se è L,phi_i=[x_i,1],a_i=q_i exp(-eta),
b_i=min(1,q_i exp(eta)). La condizione ideale è equivalente a:

```
logit(a_i) <= t_i phi_i^T theta <= logit(b_i)
```

con estremo superiore+infinito quando b_i=1. L'estremo inferiore è finito anche
per q_i=1. Ogni target e ruolo rimane presente; i coefficienti usano solo
sviluppo e la consumedval conserva il proprio denominatore. La sigmoid è
monotona,quindi queste equivalenze sono derivate direttamente dal gate,
senza surrogate di loss o esclusioni dei casi difficili.

Con matrice delle disuguaglianze Htheta>=c e limite dichiarato ||theta||1<=R,
qualsiasi lambda>=0 con somma1 soddisfa la condizione necessaria:

```
lambda^T c <= R ||H^T lambda||infinity.
```

Segue moltiplicando le disuguaglianze e applicando la dualità delle norme1/
infinito. Una violazione verificata esclude quella classe LIMITATA. Un residuo
H^Tlambda nonzero non certifica impossibilità per coefficienti illimitati.
Un candidato primale verificato su tutti i punti può invece dimostrare
fattibilità. L'export e l'aritmetica richiedono un limite specifico del nuovo
candidato; il drift misurato490 non si trasferisce a nuovi coefficienti.

**Decisione operativa:** preparare un solo filtro di intervalli con testimoni
espliciti,verifica completa del certificato e costo preventivato. La
[eligibility491](METH_491_ROOT_MASS_INTERVAL_ELIGIBILITY_20261006.md) propone un
testimone duale per root costruito su770stati con SVD,pesi interi e residuo
diadico esatto. Il subset conserva la direzione della prova di esclusione;
una futura prova primale richiederebbe tutti i punti. L'SVD costruisce il
testimone,la verifica esatta e gli intervalli ne stabiliscono il significato.
Norma/R derivano dal formato fisico e dal budget di arrotondamento,oppure il
risultato resta un limite parametrico in R. Il prossimo atto è implementare
questa eligibility e il verificatore prima di import numerici. Un esito
inconclusivo conserva l'incertezza e chiude questa singola inquiry.

Esiti che cambiano il piano:

1. Primale utile: qualifica fisica e composizione delle masse,insieme alla
   decisione;poi artefatto completo e qualità/SAMErate.
2. Esclusione verificata della classe fisica: riprogettazione di feature o
   partizione,oppure distillazione della funzione condizionale;nuovo budget
   completo prima dell'apprendimento.
3. Inconclusivo: mantenere la classe aperta e scegliere un percorso di
   trasferimento funzionale giustificato dalle evidenze;registrare il limite.

## 6. RAM e costo completo restano nello stesso problema

```
B_total = B_core + n B_expert + B_route(n) + B_state + B_workspace
T_accepted = T_core + T_head + T_route + T_selected + T_other <=20ms.
```

La RAM può aumentare n mantenendo il costo delle funzioni attive fisso solo
quando rappresentazione,routing e accessi reali lo consentono. Per il formato
originale verificato B_file=265878016+56844288n. A n1280 il solo file vale
73026566656B;questo conteggio non ammette utilità,working set residente o
prestazione. Lo stato del modello e il sistema consumano RAM aggiuntiva.

Aumentare n128 a1280 richiede al massimo11livelli in un albero binario bilanciato
di1280foglie. Un budget uniforme log(1.01) diventa log(1.01)/11 per livello;
la crescita della scelta modifica anche il requisito di accuratezza composto.
L'esposizione media Nk/n e il costo di trasferimento/adattamento vanno dichiarati.

489CPU n128 warm prose54.8773 resta il riferimento positivo;lo specifico layout
GPU per operatore è chiuso per costo. I limiti di fase489 mostrano che azzerare
solo prefill o solo decode non basta per portare quel GPU a50. Core/head/FFN,
LUT,DRAM e generazione fanno parte della verifica dello stesso risultato.

Ripresa esatta:490/R1/audit/finalizer/output-algebra completati,nessun proprio
processo scientifico attivo. Current INDEX/METHOD/NEXT puntano alla
eligibility491 di intervalli;le namespace e tutti i dati precedenti sono immutabili.
Goal completo ancora aperto: trasferimento conveniente,capacità utile n/RAM,
qualità fresca e>=50 SAMEartefatto,CPU LUT/DRAM e altre famiglie/scale effettive.
