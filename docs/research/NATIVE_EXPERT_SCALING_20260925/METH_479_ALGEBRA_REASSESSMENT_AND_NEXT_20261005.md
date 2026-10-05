# Dopo479: trasferire due funzioni, pagare ogni direzione

5 ottobre2026. Supervisione479 ammessa da R3; nessun modello appreso.
Questo è il piano della prossima singola rappresentazione, ancora da
implementare e congelare con i suoi controlli prima del fit. Non è un
protocollo già eseguito. [Risultato e limiti](METH_479_SUPERVISION_RESULT_20261005.md).

## 1. Che cosa è diventato reale

Abbiamo ora un compilatore e un audit riproducibili per tutte le scelte e
masse originali128, su387036query/238872input distinti. Il trasferimento può
usare159414input di sviluppo, con ruoli e correlazioni esatti. La validation
diagnostica79458input è già consumata: serve alla decisione locale dichiarata,
non può essere presentata come nuova prova di qualità generale.

Quattordici identità bank/esperto non vincono mai nello sviluppo. Alcune
classi positive hanno un solo input da un solo libro. Addestrare soltanto
sulle etichette globali scelte perderebbe informazione disponibile: i punteggi
originali di TUTTI128esperti forniscono i massimi locali di ogni sottoinsieme.
Su tutte le query di sviluppo, solo1/1524nodi ha una classe locale assente;
sul percorso del vincitore sono14. Non eliminare quelle foglie né confondere
copertura locale, generalizzazione e utilità causale.

La normalizzazione non può ricevere epsilon nascosti: ci sono759masse di
figlio nulle native. Nessuna coppia ha entrambe le masse nulle su questi dati.
Il logaritmo del rapporto tra figli può comunque essere infinito. Il rootZ
nativo è invece >=1 e finito per ogni query ammessa.

## 2. Due oggetti geometrici diversi

In aritmetica reale, con s_e(x)=w_e^T x:

    M_g(x) = max_(e in g) s_e(x)
    A(x) = log(sum_e exp(s_e(x))).

M_g è convessa, omogenea e lineare a tratti. La scelta tra due gruppi confronta
M_L e M_R. Un solo iperpiano non rappresenta in generale le unioni dei domini
poliedrici dei vincitori. A è liscia e convessa; determina il peso della foglia,
una volta conosciuti ID e punteggio. Il confronto di masse non sostituisce il
confronto di massimi: il controesempio479 resta valido.

Per i target native, A_native := s_*F32 + log(Z_native). Non dichiararlo uguale
alla log-partizione reale: Z usa sottrazioneF32, exp/castF32 e sommaF64 ordinata.
Il fit può usare il target native; l'export deve verificarne la probabilità
finita nel preciso operatore C. L'algebra reale motiva la rappresentazione,
la BYTE admission distingue le sue approssimazioni.

## 3. Un'unica prossima rappresentazione proposta

### Scelta: due forme lineari per supporto di gruppo

Per ciascun figlio g con almeno4foglie:

    M_hat_g(x) = max(v_g1^T x, v_g2^T x)
    v_gj = sum_(e in g) a_gje w_e
    a_gje >=0, sum_e a_gje=1.

Sono prototipi appresi nello spazio dei pesi originali. La restrizione
all'inviluppo convesso implica M_hat_g<=M_g in aritmetica reale: ogni forma
lineare è una combinazione convessa di punteggi del gruppo. NON fornisce
un upper del gruppo, un certificato del vincitore o una garanzia di pruning.
Confrontare due lower appresi resta una scelta approssimata da misurare.
La serializzazioneF32 può inoltre uscire leggermente dall'inviluppo reale.

Figli con1o2foglie usano direttamente tutte le forme originali, senza fit o
soppressione di ID. La membership478 è soltanto l'indice dei sottoinsiemi.
Nessun raggio, bound sferico o heap della ricetta respinta viene riaperto.
Nessun riaddestramento modifica le funzioni FFN originali in questa inquiry.

### Peso: una log-partizione separata con16prototipi

    A_hat(x) = log(sum_(j=1..16) b_j exp(u_j^T x))
    u_j = sum_e c_je w_e, c_je>=0, sum_e c_je=1
    b_j>0, sum_j b_j=n.

La somma dei b conserva il valore reale A_hat(0)=log(n). La scelta dell'ID usa
i supporti; la massa viene valutata una volta al root, insieme al punteggio
originale della foglia scelta. Non c'è un prodotto di sette probabilità di
ramo approssimate. Questo rimuove QUEL meccanismo di accumulo, non garantisce
un errore piccolo della partizione o della predizione.

Limite di rappresentazione esplicito: l'Hessiano reale di A_hat è la covarianza
dei16u_j sotto le sue probabilità, quindi ha rango<=15. Quello della partizione
originale può avere rango<=min(d,n-1). Il metodo ipotizza una log-partizione
approssimabile sul dominio effettivo con questo vincolo. Il rango dei dati
FFN472 e il fallimento del vecchio coarse-router non decidono automaticamente
questa diversa funzione; un fit fallito non confuterà tutti i router appresi.
K=2/Kmass=16 restano una sola configurazione, senza sweep adattivo.

## 4. Obiettivi, sviluppo e aritmetica

Proposta del primo fit:

- Tutti12bank, tutti gli input di sviluppo e tutti i gruppi da apprendere.
  Nessun fit di validation, selezione di bank, rimozione di code o politica
  basata su book>=64. Uniformare i libri, non duplicare teacher/natural identici:
  omega_u=(1/128) sum_(b in Dev(u)) 1/|U_b|.
- Supporti: errore quadratico tra M_hat_g e massimo native del gruppo su
  ALL sviluppo. Rendicontare separatamente classi/percorso scelto/ID/libro;
  il solo errore medio di supporto non ammette la scelta finale.
- Massa: errore quadratico di A_hat rispetto ad A_native, distinto dalla
  classificazione. A ID conservato, un errore delta di A produce errore di
  log-ampiezza -delta nella formula reale.
- Proposta deterministica:32passaggi completi, inizializzazione dei due
  prototipi sui primi/ultimi ID ordinati del gruppo;16prototipi di massa
  inizialmente medie dei gruppi8 al livello4, b=8. Adam/proiezione semplice
  per a/c, pesi b normalizzati positivi. Passo0.03 proposto; tutte le formule,
  tie delle derivate, momenti, proiezione, ordinamento, controlli e arresti
  devono diventare codice congelato prima del primo fit. Nessun seed/sweep
  o prolungamento scelto guardando la validation.

L'identità reale v^T x=sum a_e s_e permette di usare i punteggi già salvati
nel fit, con costo offline legato a n piuttosto che a d. I punteggi salvati
sono però F32 arrotondati: sum a_e Q(w_e^T x) NON equivale a Q(v^T x).
Esportare fisicamente v/u e ricalcolare in C tutti gli input è obbligatorio.
Non qualificare soltanto le previsioni ottenute combinando logits teacher.
Controlli sintetici/proiezione/gradiente e full native joins prima dei gate.

## 5. Contabilità prima delle prestazioni

Per n=128/d=768/profondità7, una query valuta:

    winner: 4*(7-1)+2 =26 forme originali/apprese
    massa root:16 forme
    totale:42 forme, 32256coefficienti contro98304originali.

Il punteggio originale selezionato è già prodotto nell'ultimo confronto
esatto e viene riusato. Se l'implementazione lo ricalcola, si addebita una
43esima forma. Non nascondere costruzione LUT, copie, normalization, log/exp,
tie, indici, original fallback o lavoro di core/head/FFN.

Storage:62gruppi appresi*2prototipi=124nuovi vettori di supporto;16di massa.
Mantenendo tutte128righe originali, sono268*d=205824coefficienti,823296B F32
per bank, oltre pesi b/header/indici. Nuovi coefficienti430080B per bank;
12bank nuovi5,160,960B. Questa è RAM logica, non una misura di residenza
in cache o DRAM. L'eventuale riuso di righe deve essere dimostrato nell'export.

Per potenze di2 con H livelli: supporti4H-2, massa16, totale4H+14. Lo storage
delle forme è (2n+12)d. Quindi il lavoro di QUESTA rappresentazione cresce
come d log(n), lo storage come nd. Moltiplicare n per10 aggiunge circa3.32
livelli; non dimostra conservazione di qualità, apprendibilità, LUT o utilità.
Per n non potenza di2 va congelata una costruzione bilanciata e contata ogni
forma effettiva. Non estrapolare128 a~100B senza pesi e funzioni distinti.

## 6. Decisione e arresti proposti

Prima del fit implementare e congelare UNA inquiry con binding completo479,
runtime/native/export/C, dati,32passaggi, protocolli e namespace esclusivi.
Budget proposto CPU0/BLAS1<=1200s/4GiB/new512MiB, nessun GPU/T4; audit separato
<=600s/4GiB/new32MiB. Prezzo offline cache-logit/gradienti/coeff history e
verifier deve essere concretizzato prima dell'esecuzione. Nessuna ora di
test aperta o modello addestrato su validation.

Primo gate locale conservativo proposto: ALL source ID/tie conservati,
probabilità native selezionata entro1% su ogni query, massimo/p95/medie per
bank/mode/book espliciti; lavoro e byte logici addebitati entro80% del flat,
includendo ogni eventuale fallback. Rare/zero/dev/validation nei denominatori.
Un'apparatura valida può avere esito scientifico FAIL; non aumentare K,
passaggi o soglie nello stesso protocollo. Un FAIL chiude questa precisa
configurazione/ottimizzazione, senza dichiarare una impossibilità generale.

Questi gate non sono una nuova qualità generale. Un eventuale PASS locale
richiede audit completo, costo C e poi candidato intero con stati propri,
nuovi dati, qualità donor-relativa e>=50 sullo STESSO artefatto. Se si vuole
ammettere cambi di ID, serve prima un diverso protocollo di contributo e
informazione predittiva:477-R1 vieta di sostituirlo con un RMS pooled.

## 7. Tenere collegato il complesso

Il router era soltanto~1.45/2.28% del costo nei domini originali458; abbassarlo
non risolve da solo dense/core/head/FFN. Lo scaling molto grande rende però
necessario evitare una ricerca lineare gratuita per assunzione. La nuova
inquiry risolve una parte del transfer, senza sostituire il goal con il router.

Punti di rivalutazione: dopo questo primo modello, dopo la composizione
completa, dopo qualità/SAMErate e dopo il primo incremento causale di n.
Restano aperti geometria delle funzioni condizionali, CPU LUT/traffico fisico,
core compatto, trasferimento di altre famiglie e pesi reali~100B. Le evidenze
positive123/183 e125/126 restano riutilizzabili nella propria aritmetica;
nessuna velocità, qualità o capacità viene ereditata tra artefatti diversi.

Ripresa esatta: leggere479result/RET/questo piano; implementare e congelare
il singolo learner/native verifier e la sua ammissione prima del primo fit.
NON rieseguire479/469/471/478 o gli auditor conclusi.
