# Dopo493: funzione, identificabilità e crescita della capacità

6 ottobre2026. Goal ATTIVO/INCOMPLETO. [493-R2](METH_493_WEIGHTED_TARGET_RESULT_20261006.md)
è completamente ammesso:17540UID e tutti15330816 prodotti per occorrenza
verificati BYTE da un metodo indipendente. Nessuno student, fit, kernel o
nuova misura di qualità/velocità. Questo è un nuovo passo reale del procedimento.

## Che cosa cambia nel procedimento

492 esclude soltanto l'ampiezza costante all'1% nelle sue1499 celle, non una
funzione appresa.493 costruisce la supervisione della funzione selezionata
Y=roundF32(pF), da input/output originali qualificati, per l'intera banca11:
17540UID,11721development,5819consumedvalidation,19962occurrences, tutti128slot.
Non usa il sottoinsieme107ready della ricetta472 respinta. Rifiuto del routing
significa saltare il ramo residuo; qui tutte le occorrenze originali sono
accettate. I target riguardano il prodotto e non la sottrazione post-pre.

Due problemi di apparato sono distinti dalla matematica: originale493 si ferma
al primo ruolo0/2 dell'estensione; R1 non raggiunge NumPy perché scade180s
durante gli hash. R2 corregge la DAG delle dipendenze lette: tutti gli input,
runtime, controlli e documenti di ammissione effettivamente usati sono freschi;
8,998,796,064B di pesi/binari/capture storici non letti restano descrittori
storici, senza dichiararne verificati i byte correnti. Binding R2cda324:
7.968s/34,951,168B,1,203,803,165B hashed. Nessun replay del donor né test
numerico modificato. I due primi errori, ammissioni e parziali sono immutabili.

## 1. Il numero di risposte non identifica la funzione

Nel riferimento reale, a feature A fissate, Phi_e raccoglie i vettori r x N_e.
Per G_e=B_e Phi_e, le risposte identificano B_e soltanto sullo span di Phi_e:

```
min_B ||Y_e-B Phi_e||_F^2 = ||Y_e(I-Phi_e^+ Phi_e)||_F^2
dim{DeltaB : DeltaB Phi_e=0} = D(r-rank(Phi_e)).
```

La template UNADDESTRATA r512/D768/n128 possiede50,331,648 coefficienti REALI
nei readout privati della banca.11721 stati di sviluppo forniscono al massimo
9,001,728 equazioni scalari. Poiché sum_e rank(Phi_e)<=11721:

```
sum_e D(r-rank(Phi_e)) >= 768(128*512-11721) = 41,329,920.
```

È una dimensione minima di ambiguità del modello reale a feature fissate,
non un conteggio di codici I8 alternativi e non una prova di cattiva qualità.
La quantizzazione fisica non eredita una dimensione di spazio vettoriale.
Un readout che interpola pochi stati non identifica la capacità del donor.
Serve un prior ricavato dai pesi/funzioni del donor, condivisione fra esperti,
oppure regolarizzazione e verifica su altri stati; un ID privo di sviluppo
non può diventare 'trasferito' perché gli si assegna un readout libero.
La validation consumata resta diagnostica e non diventa fresh per selezionare
questa variante. Il nuovo protocollo dovrà dichiarare i casi non identificati.

## 2. Rango locale e geometria degli stati

Nel riferimento IDEALE non quantizzato e lontano dalle pareti ReLU:

```
F_e = Wout_e ReLU(Win_e x)
J_Y = p_e J_F + F_e (grad p_e)^T
rank(J_Y) >= rank(J_F)-1
J_G = B_e diag(mask) A ; rank(J_G)<=r.
```

Quindi una regione con rango tangente di J_Y superiore a r non ammette quella
mappa G esatta. Il rango effettivo del donor non è stato misurato qui; non
assumiamo che sia768. Sulla varietà degli input normalizzati conta J_Y T,
con T base del tangente, non il rango ambientale arbitrario. Il programma
F32/I8/A16 ha rounding/quantizzazioni: questi Jacobiani ideali non sono un
certificato del suo errore BYTE o della task loss.

Senza bias, B ReLU(Ax) è anche positivamente omogenea, mentre p_e(x)F_e(x)
non lo è in generale lungo un raggio. La shell degli input normalizzati può
rendere inapplicabile quel confronto radiale al dominio effettivo. Un bias
condiviso r512 costa logicamente24,576B nelle12banche e rimuove il vincolo di
omogeneità; non rimuove il limite di rango. È una possibile modifica da
fissare nel futuro protocollo, non un fit/export già esistente.

## 3. Più esperti cambia anche la normalizzazione

Se, nel riferimento softmax ideale, si conservano i vecchi punteggi e si
aggiungono nuovi esperti, con Z_old e Z_extra:

```
p_old,new(x)=p_old,old(x)/(1+Z_extra(x)/Z_old(x)).
```

Le vecchie funzioni pesate cambiano con la nuova massa. Una G_e appresa a
n128 non è perciò automaticamente riutilizzabile a n1280. Un modello con
funzioni dirette può adottare un controllo diverso dalla softmax del donor,
ma deve validare congiuntamente decisione, contributo e qualità. RAM/file
rendono conservabili più parametri; non garantiscono capacità utile trasferita.

Una struttura che dà massa e scelta economiche è, idealmente, score_ij=a_i+b_j:

```
Z=(sum_i exp a_i)(sum_j exp b_j)
p_ij=softmax(a)_i softmax(b)_j
argmax_ij score_ij=(argmax_i a_i,argmax_j b_j).
```

È algebra esatta per quella classe e costo O(I+J), con n=IJ; non prova che
i128 esperti originali la ammettano o che conservi qualità.53..56 sono altri
artefatti. Eventuali interazioni fra le chiavi vanno pagate e possono togliere
la separabilità. La futura strada è apprendere insieme rappresentazione,
funzioni e controllo sotto il costo attivo, con qualità del risultato completo.

## Prossima decisione concreta

Completata la supervisione: fissare un unico protocollo di costo/apprendimento,
prima del fit. Specificare feature/bias/precisioni/LUT, prior dei readout e
trattamento degli ID senza sviluppo, loss/development-only/consumedval,
decisione economica accoppiata e stop. La template r512 rimane un'ipotesi
di approssimazione, non un rango scelto dopo aver visto uno spettro.

Prima pagare quantizzatori, riempimento/lettura LUT, indici/pesi condivisi,
readout privato, scale/output, router, core/head/stati/workspace. Conteggi
logici di byte non sono una misura DRAM o di latenza. Conservare come
ancora489CPU128 warm54.8773/lower53.4759 sull'artefatto originale, senza
sommare timing di altri componenti/artefatti per fabbricare un nuovo rate.

Poi verificare il candidato nel contesto completo476 già qualificato,
estendere tutte le banche e usare suoi stati/dati freschi/generazione/task
e SAME>=50 sul risultato effettivo. Per lo scaling, verificare massa/scelta,
utilità causale degli ID e traffico fisico a n crescente. Più famiglie/scale,
~10B e~100B effettivi restano richiesti; nessuna percentuale del goal acquisita.
