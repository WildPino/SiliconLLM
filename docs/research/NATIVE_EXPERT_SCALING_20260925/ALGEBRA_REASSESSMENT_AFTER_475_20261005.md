# Dopo475: misurare l'informazione osservabile dal modello

5 ottobre2026. Goal ACTIVE/INCOMPLETE. Analisi e prossimo passo proposto;
nessun sorgente/protocollo congelato/binding/risultato476 esiste ancora.

## Rivalutazione dell'insieme

475 ha reso reale un procedimento esatto e un controllo indipendente completo,
ma ha respinto la sua convenienza:11.31% delle righe certificate,5.66% dei soli
coefficienti WI evitati prima dei costi aggiuntivi. Il92.81% delle preattivazioni
è nonpositivo; il certificato individua soltanto12.19% di queste coordinate.
Il problema di questa ricetta è la larghezza del bound, non la correttezza.
Il suo centro/bitwidth/Cauchy restano chiusi; nessuno sweep di raffinamento.

Anche un procedimento corretto può non soddisfare il goal. E un errore relativo
locale non è automaticamente un errore predittivo. L'attuale catena comprende
esattezza, dati sufficienti, rappresentazione, costo attivo, qualità effettiva e
scaling di capacità utile. Occorre controllare che le metriche ai diversi stadi
misurino la quantità che la decisione finale richiede.

## Le prove su n già esistono, con un limite essenziale

[369](METH_369_SWITCH_BANK_USEFULNESS_RESULT_20261004.md) ha verificato che
due permutazioni fissate delle funzioni peggiorano predizione e generazione su
24 libri/96 casi consumati. Dimostra l'utilità della corrispondenza router-funzioni
su quel dominio, non l'utilità di ogni ID o un incremento monotono con n.

[370/371](METH_370_SWITCH_NESTED_BANK_EXPORT_RESULT_20261004.md) hanno già
prodotto e qualificato veri artefatti64/128 da un unico donor256, senza cloni.
[372](METH_372_SWITCH_NESTED_BANK_USEFULNESS_RESULT_20261004.md) è il confronto
causale già eseguito: rimuovere fino a64 peggiora NLL e campi generati;128 dà
risultati misti. La NLL dei token mascherati migliora di.318440nats rispetto256,
mentre i campi generati peggiorano di4.166667punti, con l'intervallo primario che
include zero. Il claim complessivo è FAIL5/7. Non ripetere questa stessa coorte
come un nuovo esperimento di n o dichiarare la crescita monotona già provata.

[373](METH_373_SWITCH_REAL_BANK_COST_RESULT_20261004.md) ha osservato+2.91%
costo completo, upper95+3.62%, passando64->256 funzioni reali per bank sul
workload fissato. Il tempo router cresce invece circa3.83x. Non è accepted rate
generativo, hardware DRAM o una LUT gerarchica. Questa evidenza abilita la tesi
di separare capacità totale e lavoro attivo a QUELLA scala, con qualità mista.

Questi risultati erano già completati anche se alcuni vecchi documenti NEXT
contengono ancora le proposte precedenti. Fonte operativa: INDEX/METHOD e record
dei risultati effettivi, non il testo storico di una proposta. Restano aperti
utile>256/10x, winner E massa, routing/LUT/DRAM e donor reali~100B.

## La quantità algebrica che manca fra funzione e predizione

Per l'ultimo FFN decoder, su un prefisso originale fissato:

    r = pre + p_route f,
    r' = pre + p_route f',
    l = Head_A16(FinalNorm_F32(r) / sqrt(D)),
    l' = Head_A16(FinalNorm_F32(r') / sqrt(D)).

Questa è una composizione della funzione con residuo, probabilità del router,
norme, pesi e quantizzatore reali. Un RMS relativo di f non contiene pre,
p_route, direzione dell'errore rispetto al readout, margine logit o quantizzazione
head. Il RMS aggregato di energia non misura tutti questi termini nemmeno.
Le quantità continue/Jacobiane possono motivare un metodo; le decisioni native
devono verificare la composizione numerica completa.

Scrivendo delta=l'-l e p=softmax(l), l'identità esatta in aritmetica reale è:

    KL(p || softmax(l')) = log(sum_i p_i exp(delta_i)) - sum_i p_i delta_i.

delta+c1 dà lo stesso KL: una direzione costante su TUTTI i logit non cambia
le probabilità. Per perturbazioni piccole, il termine quadratico è
.5 delta^T(diag(p)-p p^T)delta. Questa metrica dipende dall'input/distribuzione
e non è la norma euclidea dei pesi o delle funzioni.

Un bound indipendente dalle probabilità è KL<=range(delta)^2/8. Derivazione:
g(t)=log(sum p_i exp(t delta_i)); g''(t)=Var_p_t(delta)<=range(delta)^2/4,
poiché una variabile in[a,b] ha varianza<= (b-a)^2/4. Integrare
g(1)-g(0)-g'(0)=integral_0^1(1-t)g''(t)dt. Il bound può essere molto largo;
è una deduzione in aritmetica reale, non un certificato F32 o una CI empirica.
Un margine originale maggiore di range(delta) è sufficiente per conservare
l'argmax unico. Per una label y verificata, deltaNLL=LSE(l')-LSE(l)-delta_y.

Questo mostra perché occorre misurare readout e informazione predittiva,
oltre all'energia locale. Non mostra che le ricette respinte siano buone.

## Prossimo passo scelto: ammissione dei contesti e calibrazione del criterio

Nuova incertezza: quanto gli errori delle funzioni472 già conservate si traducono
in perdita informativa quando attraversano l'effettivo residuo/norma/head del
donor? Il record472 resta FAIL per i suoi gates; questa indagine non promuove
quel candidato o modifica il protocollo concluso. È un controllo di allineamento
fra criterio locale e obiettivo finale prima di scegliere un'altra rappresentazione.

1. **Prima ammettere i byte del contesto.** Implementare NEW476 per ricostruire
   senza nuovo capture la corrispondenza fra TUTTE6649 query VALIDATION472
   (3584teacher+3065natural), ledger/source offset, pre, probabilità, norma finale,
   input/codici/scale head e logit nativi originali. Same107/21policy, libri64..127.
   Tutti i join devono conservare identità input/expert/prefisso/ruolo e source SHA.
   Disponibilità e schema si verificano staticamente prima del freeze; nessuna
   scelta dei query/ID in base all'errore. Se il contesto non è ricostruibile per
   TUTTE le query, trattenere il deficit di ammissione prima di osservare qualità.
2. **Poi qualificare la composizione.** Original f deve ricostruire BYTE esatti
   post/final/head input/codici/scale e TUTTI i logit nativi source di ogni query.
   Usare l'ordine F32/F64 e il quantizzatore effettivi, fixture/native controls
   e runtime/assets completi. F64 dot intero I8/A16 è esatto sotto il suo bound;
   nessun Taylor/Jacobiano sostituisce questo oracle.
3. **Calibrazione sulle funzioni già esistenti.** Applicare al medesimo contesto
   i f'472 salvati, senza rifit, nuova base, scelta di rango, coefficiente o soglia
   dopo osservazione. Calcolare KL, deltaNLL solo con label dalla provenance,
   argmax/margine/range e rapporto con l'errore f per query/book/expert. Esporre
   separatamente teacher/natural ed equal-book contro pesatura energetica.
   Concordanza col donor non è accuratezza su label vere; una label del donor
   non diventa automaticamente ground truth.
4. **Congelare interpretazione e stop.** La sola ammissione/coerenza BYTE abilita
   il calcolo. Congelare prima i riepiloghi primari e, se si usano confronti
   numerici con453, i suoi limiti KL(.01media/.05ogni book)/1%argmax con l'esplicito
   cambio di dominio. Entrambi gli esiti sono diagnostici, senza C/native-cost
   promozione della vecchia472. Registrare dove il criterio relativo f è
   conservativo e dove segnala effettiva sensibilità; non ridisegnare gates
   mentre si vedono i logit.
5. **Budget prospettico da rendere concreto.** CPU0/BLAS1, massimo180s/2GiB e
   nuovi64MiB; source/runtime/6557+475outputs/payload e contesti tutti nel binding.
   Verificare staticamente il costo di TUTTI i logit e gli output prima del freeze.
   Streaming a blocchi, SHA per vettore completo più witness/input/scalari
   riproducibili; non campionare il vocabolario o i prefissi per far passare il budget.
   Se inammissibile, mantenere questa evidenza senza iniziare un job fuori budget.

La validation472 è tenuta fuori dal fit della base ma già consumata da indagini
locali: questa non è una nuova promozione su dati intatti. I prefissi sono quelli
originali, quindi anche una buona predizione condizionata non dimostra nuovi
stati o generazione autonoma del modello trasformato. Una nuova futura ricetta
richiede il proprio freeze, costo nativo, artefatto tutti-bank, qualità globale
su nuova coorte e SAME>=50 accepted batch1 IDs/s. La calibrazione può cambiare
quale metrica usare nella progettazione, non rende vero quel percorso da sola.

## Invariante del goal

Capacità distinta, lavoro attivo, bytes letti, perdita informativa e prestazioni
sono problemi collegati. Il metodo finale deve conservarli simultaneamente,
comprendere uno scaling utile di n verificato e applicabilità ad altre famiglie/
scale nei limiti delle risorse. Le ipotesi di trasferimento fra famiglie e
di adattamento congiunto restano aperte; nessun restart automatico del branch
donor-adaptation, nessuna acquisizione~100B o training costoso da questa analisi.
