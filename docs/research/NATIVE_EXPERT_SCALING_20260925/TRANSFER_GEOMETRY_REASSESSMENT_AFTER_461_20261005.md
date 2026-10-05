# Riesame dopo461: funzione condizionale, dominio e informazione

5 ottobre2026. Analisi derivata e scelta del prossimo problema. Nessun462
implementato o eseguito; nessun nuovo fit. Goal attivo e incompleto.

## Che cosa cambia con il risultato

Il riuso esatto dello stesso input funziona numericamente: tutti1536 output
completi sono identici. La ricetta C scelta riduce il tempo medio complessivo
del2,3894%/2,8852%, sotto il5% congelato. La chiudiamo e interrompiamo la sequenza
di kernel. Il numero di quantizzazioni eliminate non misurava la loro quota di
tempo. [Evidenza461](METH_461_SWITCH_COMMON_INPUT_COST_RESULT_20261005.md).

Il vincolo resta trasferire funzioni utili in una rappresentazione conveniente.
Il tempo ha termini distinti, T=T_core+T_expert+T_router+T_head+T_rest.458 misura
solo24,86/27,09% nelle matrici degli esperti: nemmeno eliminarle interamente
produrrebbe un guadagno illimitato (tetto ipotetico1,33/1,37x in quei contesti).
RAM, lavoro attivo e qualità devono essere verificati insieme sull'artefatto
composto. Le velocità dei controlli originali non qualificano una nuova conversione.

## Quattro problemi e le prove che li vincolano

| Problema | Prova disponibile | Condizione ancora mancante |
| --- | --- | --- |
|Conservare informazione privata|443: cambiare/rimuovere vere funzioni native modifica la previsione;440/441: il readout appreso collassa quasi a una funzione|Rappresentazione che conserva il contrasto tra funzioni su esempi indipendenti|
|Comprimere la funzione|444/445: basi globali non generalizzano;446: delta privati/Frobenius costosi;453: WI-I4/WO sparso passa uno schermo locale|Qualità della composizione in tutte le banche e costo dello stesso artefatto|
|Consultare n grande|393: coda della softmax ampia;394/395: approssimazioni specifiche respinte|Certificare identità e massa, pagare ricerca/fallback e letture reali|
|Aggiungere capacità utile|369: identità delle banche utile;372: sottoinsiemi reali hanno qualità mista|Incrementi di funzioni distinte con beneficio causale, core compatibile e dati adeguati|

E128/E256 pretrained hanno core diversi. Unire i loro file non rende compatibili
le coordinate; i ponti403/404/406/414/416 e i pilot426/431/435/440 restano chiusi.
Anche aggiungere ID senza conservarne il contributo condizionale non trasferisce
capacità. Il progetto richiede una prova informativa, oltre al conto dei parametri.

## Nuova variabile: il dominio di ciascuna funzione dello stesso donor

Una matrice può avere rango alto sull'intero spazio e agire in modo comprimibile
su un dominio condizionale. Per un esperto ReLU, in algebra reale:

    f_e(x) = W_o,e ReLU(W_i,e x)
    P_e = U_e U_e^T, con colonne ortonormali
    W_i,e x - W_i,e P_e x = W_i,e(I-P_e)x

ReLU è1-Lipschitz in norma euclidea, quindi, per queste matrici reali:

    ||f_e(x)-W_o,e ReLU(W_i,e P_e x)||
      <= ||W_o,e|| * ||W_i,e(I-P_e)x||.

L'identità del router, la sua probabilità e le perturbazioni a valle aggiungono
altri termini. Il bound può essere largo; non è un certificato misurato di qualità.
Nel native I8/A16 l'input della matrice è q_e=Q_A16(x), con scala alpha: si deve
studiare il dominio dei codici effettivi, mantenendo scale/arrotondamenti e budget
di errore. Una fattorizzazione reale non eredita automaticamente la parità C.

La nuova ipotesi riguarda proiettori INPUT privati sul dominio realmente assegnato
dal router dello STESSO core.446 confrontava delta da esperto0, metrica Frobenius
e input forzati a tutte le funzioni;445 misurava span OUTPUT locali collettivi.
Quei fallimenti restano validi. Cambiano oggetto e dominio, senza assumere che la
nuova variante passi. Una stima su pochi input può interpolare perfettamente e
fallire altrove: per questo il primo passo è l'ammissione dei dati, senza fit.

## Il conto dei byte precede la fattorizzazione

Per D768/F3072, mantenendo WO-I8 originale e le scale delle righe, fattori F32
A_e di dimensione F*r e U_e di dimensione D*r richiederebbero ALMENO:

    B_factor(r) = 4*r*(D+F) + 4*F + D*F + 4*D
                = 15,360*r + 2,374,656 byte/esperto.
    B_original = 2*D*F + 4*(D+F) = 4,733,952 byte/esperto.

Metadata/workspace escluse: sono limiti favorevoli, non un export. Per ridurre
i byte serve r<=153; per almeno20% di riduzione dell'intero esperto serve r<=91.
Il numero di prodotti reali non stabilisce il costo di F32 contro I8/A16 né DRAM.
Non assumiamo fattori I4/I8 già validati. Questi conti impediscono di chiamare
"compatto" un fattore privato che costa più della matrice originale quantizzata.

Un insieme di m input ha rango AL PIÙ min(m,768). Questa è una proprietà del
campione, non una misura del rango della funzione o del suo dominio futuro.
Serve registrare esempi distinti e libri indipendenti per ciascuna banca/esperto;
encoder ripetuto e prefix teacher/natural condivisi non creano informazione nuova.

## Prossima decisione, senza un altro kernel

NEW462 proposto: analizzare soltanto le tracce native393 già qualificate, i route
record e i manifest, per ammettere o respingere un successivo studio del dominio
privato. Per tutti12bank di entrambe le sorgenti: esempi/esecuzioni/rifiuti,
duplicati di input/codici, libri indipendenti, copertura dev/validation e limite
campionario dello span. Banca del futuro probe fissata a priori: ultima banca
sparse decoder di128, già diagnosticata443; niente selezione della banca più
favorevole dopo i conteggi. Costi/budget/gate congelati nel protocollo462 prima
di parsare le tracce per nuove osservazioni.

Se la copertura non consente uno studio informativo, documentare il deficit e
definire nuovi contesti indipendenti prima di costruire fattori o addestrare.
Se lo consente, una sola futura prova geometrica sul dominio proprio, con
controlli di identità/trasferimento al validation e conto dei byte; nessuna
promozione dal solo errore di ricostruzione sul development. Il probe geometrico
rimane distinto dall'eventuale modello composto e dalla verifica finale fresca.
[Proposta operativa](SWITCH_NATIVE_QUERY_DOMAIN_ADMISSION_NEXT_20261005.md).

## Collegamento a n,100B e generalità

Il numero di esperti ospitabili è limitato da byte privati +router +core +memoria
di esecuzione; il numero utile dipende dall'informazione condizionale conservata.
Ridurre i byte può aumentare il primo limite, senza dimostrare il secondo.
Il rapporto circa10 tra n di100B e10B vale solo con geometria comparabile e
parametri realmente distinti. QwenNext397 espone un donor reale di~79,674B, ma
header/forme non sono valori acquisiti né una funzione trasferita. Giga/Ling/
Granite mantengono evidenze e limiti delle ricette specifiche già respinte.

Il nuovo dominio privato sarebbe una parte del procedimento, con prerequisiti
espliciti e fallback alle funzioni originali. Composizione end-to-end, dati
held-out nuovi, generazione/task, SAME50, massa del router, DRAM fisica e un'altra
famiglia/scala restano condizioni necessarie. Nessuna è risolta da questo riesame.
