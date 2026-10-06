# Dopo492: dall'ampiezza alla funzione condizionale trasferita

6 ottobre2026. Goal ATTIVO/INCOMPLETO. [492](METH_492_SELECTED_AMPLITUDE_RESULT_20261006.md)
è completamente ammesso. L'indagine successiva deve costruire il trasferimento
della funzione, evitando un'altra variante equivalente del filtro scalare.

## Evidenza che cambia la scelta

491 impone enormi norme necessarie a quattro root affini; otto testimoni
restano inconclusivi e nessuno ha residuo esattamente zero.492 tratta un
dominio diverso: ogni ID dove è scelto. Anche qui una semplice costante
all'1% è esclusa da1499 delle1522 celle esposte nello sviluppo. Le23 celle
eligibili hanno solo1..4 stati, e27/30 stati di validazione con scalare
disponibile falliscono il candidato fissato. Non emerge una scorciatoia
generale per incorporare p nel medesimo WO attraverso una costante.

Questa è una limitazione del contratto di ampiezza e della classe costante.
Il goal richiede qualità del modello completo; non impone di emettere
separatamente una p del donor entro1% in ogni stato. La classe negativa non
dimostra che un'altra funzione G_e(x) non possa conservare il contributo utile.

## Algebra del nuovo oggetto

Per il riferimento reale, con punteggi lineari s_e=w_e^T x:

```
ell_e(x)=log(p_e(x))=w_e^T x-log(sum_j exp(w_j^T x))
grad ell_e=w_e-sum_j p_j w_j
Hessian ell_e=-Cov_p(w), negativa semidefinita.
```

La log-ampiezza è concava in questo riferimento; la differenza di log-masse
di gruppo491 ha invece Hessiana differenza di due covarianze. Questa
identità non rende automaticamente economica o affine l'ampiezza sul dominio
reale. La funzione vettoriale p_e F_e non è in generale concava o convessa.
Il programma nativo ha quantizzazione/rounding specifici: i target vanno
costruiti da quel programma, senza sostituire queste identità ideali ai byte.

Sia H_e la matrice delle feature del donor su un insieme di stati e
F_e=W_e H_e. Con p positiva e senza maschere:

```
Y_e=F_e diag(p)
rank(Y_e)=rank(F_e).
```

Moltiplicare per p non riduce il rango ESATTO nel riferimento reale: diag(p)
è invertibile. Può cambiare valori singolari, geometria delle risposte e
approssimabilità con una metrica diversa. Rounding e maschere fisiche sono
separati; non si deduce un rango esatto dai piccoli valori singolari F64.

Per un target G_e=B_e phi_A(x), raccogliere Phi_e su sviluppo. Esistenza di
un readout esatto e minima loss quadratica non pesata nel riferimento reale:

```
B_e Phi_e=Y_e possibile iff Y_e(I-Phi_e^+ Phi_e)=0.
min_B ||Y_e-B Phi_e||F^2=||Y_e(I-Phi_e^+ Phi_e)||F^2.
```

Questa geometria distingue imparare una mappa di input condivisa da assumere
che essa contenga già le risposte. La futura mappa non è stata addestrata;
un readout che interpola pochi stati non prova qualità su stati propri.

## Procedimento da rendere reale

| Passo | Stato attuale | Prossima evidenza richiesta |
| --- | --- | --- |
| Donor originale nel C |489: qualità originale e CPU128 warm54.8773/s nel suo contesto |Convenienza/scaling del risultato trasformato |
| Supervisioni di scelta/massa |479 completo;480/490 ricette respinte |Scelta economica accoppiata alle nuove funzioni |
| Source output per il pilot |472:19962 reference768F32;476: percorso nativo/head completo sulla consumedval |493: pF fisico, tutti17540UID/19962occurrence, ruoli e byte qualificati |
| Funzione condizionale nuova |G=B_e phi_A è proposta |Fit sotto budget attivo, export fisico, errore sul contesto completo |
| Pipeline completa |Mancante |Tutte le banche, stati propri, dati freschi, generazione/task e SAME>=50 |
| n utile/CPU LUT/DRAM/famiglie |Precedenti123/183/369/373, ancora perimetri diversi |Utilità causale e costo fisico sul medesimo risultato trasferito |

I due array472 esistono e i loro SHA freschi coincidono con l'inventario
originale ([metadata493](meth493_weighted_target_metadata.json),1.625s).
Non erano nel catalogo diretto492: il nuovo compiler deve legare esplicitamente
entrambi e le ammissioni472/476, oltre agli input479 e relativi primi fault.
Nessun riferimento numerico o old controller è stato rieseguito per questa
verifica. La compilazione Y non è ancora avvenuta.

Il pilot bank11 ha17540UID (11721dev,5819consumedval),19962occorrenze. Include
tutti128slot, senza limitarsi ai107 ready di una vecchia ricetta. Da vecchi
post non si ricostruisce il pre esatto con una sottrazione:476 ha già
qualificato il contesto che servirà a verificare il candidato su validazione.
La maschera/accepted e l'ordine di prodotto/somma devono restare espliciti.

La nuova variabile è una funzione vettoriale appresa sotto un costo fissato,
con feature condivise e readout condizionali.472 già misurava errori pesati
con p: un semplice cambio di metrica della vecchia PCA non è la nuova via.
Il candidato deve sostituire la funzione FFN e partecipare a un percorso
conveniente completo. Product-key appreso è una possibile scelta economica;
53..56 sono precedenti separati, non qualità/timing ereditabili sul nuovo G.
La template r512/ternaryA/I8B/8x16keys è soltanto un envelope logico proposto.
LUT table reads/writes, quantizzazione, core/head/state, working set e DRAM
si devono pagare prima di avviare un fit come candidato finale.

Per scelta d(x), la composizione rimane:

```
||G_{d(x)}(x)-Y(x)||
 <= ||G_{d(x)}(x)-G_{e(x)}(x)||+||G_{e(x)}(x)-Y(x)||.
```

L'informazione del modello finale richiede logits/generazione/task.477-R1
dimostra190/3065 argmax cambiati con solo2.83%RMS FFN: la loss locale serve
a costruire, non a promuovere. Cambiare la decisione e la funzione insieme
richiede verifica congiunta. I dati di validazione già consumati non diventano
fresh per il nuovo candidato.

## Ripresa e limiti complessivi

Next [493 eligibility](METH_493_WEIGHTED_FUNCTION_TRANSFER_ELIGIBILITY_20261006.md):
implementare/fissare compiler pF e verifica indipendente completa, schema/
controlli/provenienza/costi; sole binding/invocazioni180s256MiB/300s256MiB,
output128MiB preventivati prima di leggere le risposte per costruire i target.
Nessun fit/GPU/native da avviare in questa fase. Poi protocollo completo
di costo e apprendimento del target, non un altro scalar sweep.

Nessun nostro processo scientifico attivo;492 è terminale/ammesso, metadata493
è terminale e distinto dalla scienza non ancora implementata. Engine e tre
file estranei hanno gli stessi SHA. Tutti i primi fault e namespace completati
rimangono immutabili. Goal attivo: core conveniente, capacità utile n/RAM,
CPU LUT, DRAM fisica, qualità fresca/SAME>=50 e più famiglie/scale effettive.
