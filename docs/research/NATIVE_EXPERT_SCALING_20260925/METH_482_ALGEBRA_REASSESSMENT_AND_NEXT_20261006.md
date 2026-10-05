# Dopo482: margine, informazione utile e budget dell'artefatto completo

6 ottobre2026. Goal ATTIVO/INCOMPLETO. [Esito482](METH_482_AFFINE_ROOT_RESULT_20261006.md)
ammesso indipendentemente:12 LP a tempo limite, nessun vettore esposto. Questo
documento contiene deduzioni e un nuovo metodo proposto, non un risultato483.
[Quadro481](METH_481_ALGEBRA_REASSESSMENT_AND_NEXT_20261005.md) resta il riferimento
per massa condizionale, geometria dei gruppi, RAM, CPU LUT e costo completo.

## 1. La domanda geometrica ridotta a un inviluppo convesso

Per ogni root, sviluppo completo con x_i F32 originali, etichetta y_i in{-1,+1}
del programma originale e phi_i=[x_i,1]. Definire h_i=y_i*phi_i e

```
M = max_{||theta||_1 <= 1} min_i h_i^T theta
  = min_{lambda >= 0, sum(lambda)=1} ||sum_i lambda_i*h_i||_infinity
  = distance_infinity(0,convex_hull({h_i})).
```

M>=0 perché theta=0 è ammessa. Prima uguaglianza duale: scrivere il minimo
sui punti come minimo sulle loro combinazioni convesse. Il LP con u,v>=0,
theta=u-v, sum(u+v)<=1 e h_i^T(u-v)>=t è fattibile e limitato. Il suo duale,
eliminando il moltiplicatore della norma, minimizza ||H^T lambda||_infinity
sul simplesso, prendendo t libero; t>=0 non cambia l'ottimo dato M>=0.
Per dualità forte i valori coincidono. È una nostra applicazione
della dualità LP e della coppia di norme1/infinito; fondamenti in
[Boyd/Vandenberghe, sezioni5.2.4 eA.1.6](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).

M>0 equivale a separabilità affine STRETTA di questo dominio finito. M=0 equivale
a origine nell'inviluppo convesso. Un residuo flottante piccolo non prova M=0;
non si sostituisce un'approssimazione numerica a una combinazione convessa esatta.
Pareggi originali restano etichette tie-aware; una separazione stretta è una
condizione sufficiente scelta, non una necessità per ogni programma a tie key.

Per QUALSIASI theta con norma verificata<=1 e QUALSIASI pesi nonnegativi con
somma positiva, indipendentemente dall'ottimalità o timeout del solver:

```
max(0,min_i h_i^T theta) <= M <=
||sum_i lambda_i*h_i||_infinity / sum_i lambda_i.
```

Intervalli esterni devono includere tutti prodotti/somme/divisione. Un candidato
arbitrario diventa una prova soltanto dopo norma e tutti i vincoli controllati.
Un limite superiore<=E_uniform esclude solo il requisito conservativo M>E_uniform;
NON ogni head fisicamente corretto. E_uniform è un limite superiore dell'errore,
non un errore minimo inevitabile. Head specifiche possono avere errori minori.

## 2. Risolvere pochi vincoli, verificare tutti i punti

Per sottoinsieme attivo A, M_A>=M. Ogni peso duale su A, esteso con zeri sugli
altri punti, è un testimone valido per il limite superiore del problema intero.
Ogni theta deve invece essere controllata su TUTTO lo sviluppo prima di attribuire
un limite inferiore. Non si trasferisce la fattibilità del sottoinsieme al totale.

Nuovo metodo prospettico: prima UID sviluppo per ogni sourceID esposto; solve
LP attivo; acquisire vettori e flag value_valid/dual_valid; normalizzare e fare
scansione completa dei margini; aggiungere i256 vincoli inattivi col margine
inferiore peggiore, tie per UID crescente; mantenere il modello/basis quando
si aggiungono righe. Massimo8 round,25s totali di ricerca per root, nessuna
griglia o aumento del budget del solver completo482. Conservare ogni round,
gli indici, i vettori e la scansione completa. Le API sono descritte nelle
[fonti primarie HiGHS](https://ergo-code.github.io/HiGHS/stable/interfaces/python/example-py/);
il binding privato SciPy installato deve essere qualificato dai controlli del
nuovo metodo, non presunto identico a highspy standalone.

Conservare il migliore limite inferiore verificato e il migliore limite superiore
verificato, senza scartare timeout, label, tie o sourceID rari. Costruzione usa
solo sviluppo; ALLconsumedval è valutazione dichiarata, mai nuova prova held-out.
Export F32, intervalli fisici e ALL39/72/4608/9216viste restano completi. Nessun
solver ottimo da solo promuove C/LUT o qualità. Freeze completo codice/protocollo/
runtime prima del primo import numerico; admission e audit propri, nessun replay482.

Stop scientifico: se il candidato passa il dominio consumato con margini fisici,
passare alla massa e al round-trip C; se il requisito conservativo è respinto
dal limite duale, riprogettare feature/partizione per questa variante. Se resta
inconclusivo, documentare il limite e tornare al budget dell'artefatto completo,
senza un terzo LP equivalente con tempi maggiori. Nessuna confutazione universale.

## 3. Il criterio finale è informazione conservata nella funzione

Per output condizionale source p*f_e e candidato p_hat*f_ehat, qualsiasi norma:

```
||p_hat*f_ehat - p*f_e|| <=
|p_hat-p|*||f_e|| + p_hat*||f_ehat-f_e||, per p_hat>=0.
```

La scelta e il peso hanno effetti distinti ma compositivi. Un ID diverso può
avere effetto piccolo se le funzioni coincidono sul contesto; lo stesso ID con
peso sbagliato può perdere informazione. Una RMS piccola non controlla la
distribuzione predittiva:477-R1 mostra190/3065argmax naturali cambiati e KL
media.03947 su prefissi ORIGINALI, non una generazione propria del candidato.
Le verifiche esatte di routing sono diagnostica/prerequisiti delle ricette
scelte; il goal richiede qualità fresca predittiva/generativa/task dell'artefatto.

Un obiettivo futuro orientato alla funzione può minimizzare distorsione
condizionale e divergenza predittiva, con un vincolo esplicito su lavoro/byte.
Deve prezzare feature, training, tutti gli ID e interferenze tra livelli.
Non basta sostituire una metrica dopo un FAIL o promuovere i vecchi fattori.
Nessun nuovo addestramento lungo è avviato da questo documento.

## 4. n è memoria distinta, lavoro selettivo e apprendimento utile

```
B_total = B_core+B_state+B_workspace+n*B_expert+B_router(n)
T_accepted = T_core+T_head+T_route+T_selected+T_other <=20ms.
```

RAM maggiore permette n maggiore solo per byte/esperto e overhead comparabili.
Identità R in n categorie ha H(R)<=log2(n); tale limite non prescrive il costo
del calcolo, né misura conoscenza/capacità dell'intero modello. Nel target gerarchico
il traffico dei coefficienti può essere O(log n) e lo storage O(n), se i margini
e la massa permettono quelle teste. Non è una garanzia per dati/pesi arbitrari.
Con N interrogazioni e k scelte ciascuna, l'esposizione media è Nk/n; nel caso
uniforme è anche l'esposizione attesa di ogni slot. Trasferimento di pesi esatti
e apprendimento di nuovi residui hanno costi d'esposizione diversi.

Esperti distinti vanno verificati causalmente: ablation/permutazioni/interventi
predefiniti e qualità fresca, costo reale quando il pool supera cache. CPU LUT
e DRAM devono essere misurate sull'organizzazione e sui pesi candidati. Non si
confonde payload mappato con working set residente o con traffico fisico.

458 isola solo1.45%/2.28% del tempo nelle MATRICI router; massa/softmax/controllo
sono anche nel residuo. Dense/control40..44%, matrici esperti25..27%, head~6%
impongono un percorso parallelo nel piano dell'artefatto. Sono componenti della
diagnostica ALL-case: non applicarle a un singolo token o al solo decode.
I due sorgenti hanno worker/context diversi; nessuna curva causale n-only.

## 5. Ordine del lavoro e decisioni

1. Chiudere482 con audit completo: FATTO, inconclusivo ammesso; nessun rerun.
2. UNA nuova procedura LP a vincoli generati con vettori espliciti: risolve il
   limite di apparato e può decidere il margine; budget fissato sopra. È
   preparata nel [protocollo483](METH_483_CONSTRAINT_ROOT_PROTOCOL_20261006.md);
   freeze/registrazione/esecuzione e audit devono precedere qualunque risultato.
3. Se utile: massa condizionale normalizzata e aritmetica C della geometria;
   altrimenti scegliere feature/partizione orientata alla funzione finale.
4. Artefatto completo trattabile: dichiarare core/head/FFN e budget<=20ms,
   prerequisiti del donor, perdite esatte/approssimate e costo di adattamento.
5. Verifiche fresche di prediction/generazione/task donor-relativi E velocità SAMEartefatto;
   poi n distinti utili, pool DRAM, altra famiglia/scala.100B effettivi quando
   risorse/formato lo permettono, non equivalenza da soli header o pesi duplicati.

Classificazione del turno precedente: PROGRESSO,481 ammesso.
Turno corrente: PROGRESSO,482 inconclusivo ammesso indipendentemente
e nuova riduzione geometrica formulata. Nessun blocco ripetuto/impasse; goal attivo.
