# Dopo478: separare geometria della scelta, massa e trasferimento

5 ottobre2026. Riesame dopo il completo [478](METH_478_GROUPED_ROUTER_RESULT_20261005.md),
audit indipendente PASS. Nessun NEW479 congelato o eseguito in questo documento.
Il goal resta ACTIVE/INCOMPLETE. Le formule seguenti sono deduzioni in aritmetica
reale, salvo i risultati nativi esplicitamente citati; non nuovi esperimenti.

## Stato del progetto visto nel complesso

| Problema | Evidenza realmente acquisita | Ciò che manca |
| --- | --- | --- |
| Aggiungere capacità condizionale utile |123: gerarchia128->1280, qualità esterna sul piccolo Qwen BF16;183: scelta dei figli causalmente allineata;369: matching funzioni reale Switch utile |Stessa proprietà dopo trasferimento conveniente a~10B/~100B; crescita oltre le scale validate |
| Rendere il costo attivo conveniente |126: A condivisa esatta/RAM ridotta;125: componente route+fattori2.510ms;373:64->256 costoFULL+2.91%;453: geometria locale conservativa |Artefatto completo conveniente, tutte le parti core/head/router/esperti e costi fisici |
| Conservare informazione predittiva |476/477-R1: impatto finito del vero norm/quantizzatore/head; vecchie qualità123 esterne nella propria aritmetica |Trasferimento con obiettivo informativo, nuovi stati del candidato, qualità e velocità sullo STESSO artefatto |
| Scegliere tra molti esperti |104/125: gerarchia appresa a1280 costa poco come componente nel caso piccolo;478: questo certificato geometrico visita quasi tutto |Trasferire un router pretrained piatto in decisioni apprese economiche; mantenere ampiezza/massa e CPU LUT con n grande |

Un fatto storico da tenere visibile: esiste già evidenza positiva di una
gerarchia appresa utile.123 passa le verifiche esterne dichiarate e183 verifica
il matching route-figli, sul Qwen2.5-0.5B-Instruct BF16. La perdita semantica del
precedente105 non è lo stato finale di quella linea.126 condivide A senza errore.
127 ha però un core FP32 diverso dall'artefatto BF16 di qualità e misura16.818
token/s; Q7/Q15 hanno fallito le successive verifiche semantiche131/133. Non
si può comporre qualità123, componente125 e una velocità di un altro artefatto.
Ripetere104/125 o espandere ancora ID sintetici non colmerebbe questo vuoto.

Sui Switch originali esistono invece artefatti realmente grandi7.415B/14.664B,
eseguibili e qualificati nei rispettivi protocolli. La capacità utile n è mista
in372. Queste due linee offrono parti diverse del procedimento: nessuna da sola
dimostra oggi il trasferimento grande, conveniente e generalizzabile cercato.
Non riattivare automaticamente il next congelato donor-adaptation.

## 1. Il nuovo fallimento è localizzato

478 conserva tutti i vincitori e controlla ogni probabilità entro1%, ma paga
circa1.98x coefficienti e2.97x byte logici dei pesi. Dal conteggio esatto delle
tre exp per nodo interno con massa segue che già la fase vincitore espande
~123.886/127 e~242.756/255 nodi interni. La normalizzazione è un problema reale
(393), ma riparare soltanto la massa non risolve questo algoritmo di ricerca.

Il bound usa il contenitore sferico

    max_(e in g) w_e^T x <= c_g^T x + R_g ||x||.

La sfera perde la direzione dei residui rispetto alla query. I contatori
dimostrano insufficiente pruning del preciso algoritmo; non misurano da soli
la causa spettrale, la dimensione intrinseca o l'impossibilità di altre geometrie.
THIS ricetta è chiusa. Cambiare un raggio/soglia o rifare una partizione sarebbe
uno sweep della strada respinta; serve una diversa funzione da trasferire.

## 2. Due funzioni diverse: massimo e log-partizione

Per un gruppo g di esperti pretrained:

    M_g(x) = max_(e in g) w_e^T x,
    A_g(x) = log(sum_(e in g) exp(w_e^T x)).

M_g è la funzione di supporto dell'inviluppo convesso delle righe. È convessa e
lineare a tratti. A_g è convessa e liscia in aritmetica reale. Vale

    M_g <= A_g <= M_g + log(|g|).

Per figli L/R, il figlio contenente il vincitore si sceglie confrontando M_L
con M_R, con pareggi risolti mediante l'ID originale. Il rapporto delle masse
usa invece A_L-A_R. Questa differenza non può essere ignorata:

    gruppo L: logits (0, 0), gruppo R: logits (1/2, -100).

La massa di L è maggiore, ma il vincitore è in R. Scegliere avidamente il ramo
più probabile di una hierarchical softmax può quindi cambiare il top1.
Una gerarchia nata durante l'addestramento è lecita; ricodificare il router
piatto pretrained richiede misurare la conservazione delle sue funzioni.

Il dominio di vittoria di e è la regione poliedrica

    C_e = {x : (w_e-w_j)^T x >= 0 per ogni j},

con frontiere/tie-break espliciti. Il dominio di un gruppo è un'unione di C_e;
non è in generale un semispazio o una regione convessa. Un classificatore
affine di ramo non è quindi garantito esatto da una partizione dei soli pesi.
Questa è una complessità di rappresentazione da ammettere, non un motivo per
presumere che un piccolo MLP o una LUT la risolvano.

## 3. Una massa normalizzata non rende la ricerca automaticamente economica

Con probabilità condizionali di ramo esatte q, la probabilità di una foglia è
il prodotto lungo il suo cammino. Questa rappresentazione conserva la massa1.
Il costo di valutare UNA probabilità può essere proporzionale alla profondità,
se ogni q è economica. Trovare la foglia di probabilità massima è un altro costo.

Un best-first che usa la massa del gruppo come upper delle foglie può essere
lineare: nella distribuzione uniforme, ogni gruppo con m>1 ha massa m/n>1/n.
Quel solo upper non può potare alcun gruppo interno prima del primo candidato
foglia; un albero binario completo richiede espandere tutti i n-1 gruppi interni.
Ulteriori informazioni possono cambiare il risultato; la mera normalizzazione
non certifica O(log n). È lo stesso motivo per tenere separati M_g e A_g.

Se su un cammino si ammettono errori di log-probabilità eta_v per ramo,

    |log(p_hat_e/p_e)| <= sum_(v nel cammino) eta_v.

Per errore relativo<=1% è sufficiente somma eta_v<=log(1.01). Più profondità
consuma budget: 128->256 aggiunge un livello, moltiplicare n per10 aggiunge
circa3.32 livelli nel modello bilanciato. Non si può usare lo stesso errore per
ramo e dichiarare invariato l'errore del prodotto. Se tutte le foglie avessero
errore log<=eta, il vincitore sarebbe preservato quando il margine log del
primo rispetto al secondo supera2eta; i pareggi richiedono un controllo separato.

Per n identità, un codice fisso richiede almenoceil(log2 n) bit per semplice
conteggio. Questo non è un teorema sul costo di calcolo o di traffico per token.
Costo delle funzioni di ramo, indice, training, foglie attive e cache va pagato.

## 4. Il trasferimento va ottimizzato sul contributo osservabile

Per top1 Switch il contributo è a_e(x) f_e(x), non il solo indice.369 mostra
che le funzioni non sono intercambiabili.477-R1 mostra che un piccolo RMS
aggregato non garantisce preservazione predittiva. La gerarchia futura deve
conservare separatamente scelta, ampiezza e contributo nel contesto.

Se la scelta resta e e si apprende logZ con errore delta,

    a_hat/a = exp(-delta),
    delta_r = (a_hat-a) f_e.

Un controllo |delta|<=log(1.01) limita l'errore relativo dell'ampiezza a1%,
ma non garantisce un piccolo errore dopo norma/head o lungo la generazione.
Se cambia anche l'ID, il residuo è a_hat f_(e_hat)-a f_e; non esiste una
riduzione alla sola probabilità o all'accuratezza del classificatore.
Il pullback informativo477 fornisce una motivazione locale; i veri operatori
quantizzati e i nuovi stati restano verifiche finite necessarie.

## 5. Prossimo lavoro scelto: NEW479, supervisione del transfer prima del fit

Incertezza: disponiamo di supervisione pretrained, con ruoli/identità/costo
completi, per apprendere decisioni M_L-M_R e ampiezza/log-partizione su TUTTI
i bank? La nuova variabile è il trasferimento della struttura decisionale e
della normalizzazione, non un altro bound di raggio o shortlist394/395.

Il primo passo sarà un compilatore offline di supervisione, senza training,
nuovo capture o benchmark. Riutilizzo qualificato: i387036 score/input/route
originali128 di469+471, i payload/manifest originali, i ruoli già espliciti e
la sola membership128 della partizione weight-only478. Riutilizzare membership
come indice dei target non riapre il certificato sferico economico respinto.
Per256 manca qui un equivalente sviluppo allargato: nessun fit sui393 consumati
o pretesa di trasferimento causale dalla scala128.

Passi da rendere concreti PRIMA del freeze e della prima osservazione479:

1. Unire le512 vecchie e256 nuove case una sola volta, senza recapture. Ruoli
   espliciti:0..63 sviluppo,64..127 validation diagnostica già consumata,
   128..191 sviluppo aggiunto. Non interpretare semplicemente book>=64 comeval.
   Preservare ogni query/capacity-rejected route e bank/mode/ID nel denominatore.
2. Deduplificare gli inputF32 BYTE identici per bank, preservando legami a tutti
   i libri/case/modi/ruoli; encoder e primo decoder hanno viste correlate.
   Un'identità presente nello sviluppo non può diventare validation nuova.
   Escludere validation dal fit, conservare gli overlap e tutte le code rare.
3. Ammettere il preciso nuovo dominio native: ALL original full-score/winner/
   probability BYTE, prima dei target.469/471 hanno già controlli e validazioni,
   ma la BYTE-full-score admission478 riguarda393, non automaticamente471.
   Un nuovo piccolo verifier di router, senza replay del modello, è quindi
   da progettare/fissare insieme a runtime/assets/ledger prima dell'esecuzione.
4. Costruire target di massimo con tie-ID per tutti i gruppi e log-masse stabili,
   distinguendo la funzione reale dalle sottrazioni/exp/cast/somma nativi.
   Non sostituire logZ reale a una ricostruzione byte del coefficiente nativo.
   Ritenere identità, query, parent/figli, target e conteggi completi/exactEOF.
5. Quantificare offline bytes/lavoro/variazioni dei target ed esposizione per
   decisione/foglia. Questo ammette la supervisione, non dimostra apprendibilità
   o accuratezza. Niente selezione di bank/basis/MLP in base alla validation.
   Audit indipendente completo senza replay o refit, sole first invocation.

Budget PROPOSTO, da concretizzare mediante metadata e dimensioni esatte prima
del freeze: CPU0/BLAS1, main<=600s/4GiB/new2GiB, audit<=600s/4GiB/new32MiB,
NESSUN GPU/T4/training. Se il limite non copre la BYTE-full-score admission e
tutti i target, rivederlo PRIMA di osservare, senza campionare query/bank.
Arresto: primo fault/risorsa/joins/EOF/byte mismatch; conservarlo prima di una
correzione numerata. SupervisioneFAIL ferma il fit. SupervisionePASS consente
di congelare UNA rappresentazione appresa, suo obiettivo e costo/stop prima
del fit; non autorizza un'esplorazione illimitata di profondità/rango/soglie.

La scelta del modello di ramo/massa non è ancora implementata o congelata.
L'apprendimento potrà essere congiunto e condividere feature, ma non basta
imporre un coefficiente affine: la geometria sopra ne mostra il limite.
Il futuro gate locale precede un export C con tutti i costi, poi nuovi dati e
stati propri del candidato. Una perdita di ID può essere studiata con un nuovo
protocollo di contributo/qualità, senza dichiarare PASS retroattivo di394/395.

## 6. Controlli del complesso e ordine successivo

Un PASS del router non risolve da solo il costo attivo completo:458 identifica
anche core/dense/head;50token/s significa20ms complessivi sullo stesso artefatto.
La via informativa delle funzioni e quella del core devono rimanere collegate
al transfer, con memoria residentemente distinta da byte logici e DRAM misurata.
LUT di query, costo della loro costruzione, formati/indice/fallback e cold-state
traffic vanno ammessi prima di estendere il claim a n molto grande.

Prossimo checkpoint di rivalutazione: dopo la supervisione479, identificare
esattamente quale parte del transfer è ora reale e quale costo può sostenere
il fit. Dopo il primo fit: qualità locale/costo; dopo l'artefatto completo:
fresh prediction/generation/task e SAME>=50; poi utilità causale a n crescente
e una seconda famiglia/scala. Nessun esito piccolo si eredita a100B.

Alla ripresa NON rieseguire478 né104/125/469/471/477-R1; leggere INDEX/METHOD
e questa proposta, verificare file/processi, concretizzare il singolo479.
