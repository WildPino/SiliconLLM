# Dopo474: informazione condizionale senza perdita di coordinate

5 ottobre2026. Goal ACTIVE/INCOMPLETE. Rivalutazione e proposta; non è un
protocollo congelato o un esperimento475 già eseguito. Non esistono ancora
sorgenti, binding, conteggi o misure475. Nessun risultato viene riaperto.

## Che cosa sappiamo adesso

| Problema | Evidenza valida | Conseguenza operativa |
| --- | --- | --- |
| Dati privati insufficienti |471: tutti12bank ammessi; bank11 con107ID/97.1615%copertura |La scala dei dati è sufficiente per porre la domanda geometrica; non aumentarla dopo ogni fallimento|
| Conservare gli input con32assi |472:105/107 funzioni falliscono |Chiudere quella ricetta|
| Conservare le preattivazioni con rango variabile |473: almeno7365 ranghi totali contro4684 consentiti dal formato/70%budget |Chiudere quella famiglia di fattori F32 per quel criterio|
| Conservare le uscite complete in32assi privati |474:105/107 limiti ottimistici falliscono |Chiudere queste basi prima di calcolare i coefficienti WO|
| Conservare tutte le dimensioni con WI-I4/WO-I8 |453: qualità locale passa;454: costo C delle ricette fissate fallisce |Nessuna promozione ad export globale sulla sola qualità locale|
| Ridurre il costo con sola organizzazione del kernel |456tile16 e461riuso input non passano le rispettive soglie |Nessun nuovo sweep dei kernel chiusi|

La media energetica è un rapporto di somme: R_pool²=sum_e E_e R_e²/sum_e E_e,
con la STESSA pesatura per E_e e R_e. Nel dominio naturale472 due esperti
raccolgono99.8934% dell'energia delle uscite. Una buona media non controlla
gli esperti debolmente pesati; l'energia non ne dimostra l'importanza semantica.
Non si possono inserire nella formula gli RMS novel/equal-book di474 insieme
alle energie aggregate naturali di472: sono pesature differenti.

Il rango di una matrice, la dimensione di una regione della funzione, i bit
necessari per descrivere i pesi e il traffico fisico per token sono quantità
diverse. I fallimenti dei fattori non provano che ogni consultazione condizionale
sia inefficiente. Impediscono le promozioni delle famiglie effettivamente provate.

## Una via nuova: due gruppi di4bit e certificato esatto di ReLU

Proposta scelta per la prossima ammissione economica. Nuova variabile:
precisione letta condizionata da un certificato intero prima della ReLU.
Nessuna base appresa o riduzione delle dimensioni; tutti i coefficienti I8
del donor restano rappresentati esattamente. Questo non comprime lo storage
complessivo: aggiunge una piccola norma per riga. Può ridurre i bit consultati.

Per ogni coefficiente w in[-128,127], definire:

    h=floor(w/16), b=w-16h-8,
    h,b in[-8,7], w=16h+8+b.

Entrambi h e b hanno4bit con segno. La coppia rappresenta tutti i256 valori
I8 in modo biiettivo, inclusi gli estremi. La divisione dei negativi deve essere
floor; in C lo shift/divisione va reso esplicito e qualificato. Non confondere
questa codifica con il precedente I4 approssimato limitato a[-7,7].

Per una riga WI e gli effettivi codici A16 q in[-32767,32767]^768:

    z=w·q=C+b·q,
    C=16(h·q)+8 sum_j q_j,
    R²=sum_j b_j², Q²=sum_j q_j².

Per Cauchy-Schwarz, |b·q|<=sqrt(R² Q²). Quindi il test intero

    C<0 AND C²>R² Q²

certifica z<0 per QUELLA riga e QUELL'input. Basta leggere h e la norma R².
La riga non contribuisce dopo ReLU; i suoi bit b non servono. Le righe non
certificate leggono b, calcolano la correzione intera e ricostruiscono z esatto.
Nessun neurone è eliminato sulla base di frequenza, piccolo valore o errore medio.

La norma R² è U32 e<=768*64. Somme, C e confronti sono I64. Poiché
|C|<=120*32767*768, C²<(2^63); anche R² Q²<(2^63). I prodotti di4bit/A16
in gruppi limitati devono avere il proprio bound I32 prima dell'accumulo I64.
Non usare norme/sqrt F32 nel test. Uguaglianze e C>=0 vanno all'esecuzione completa.

Scale originali WI positive e alpha A16, ordine F64/cast F32 originali,
ReLU, massimo/scale e codici A16 hidden rimangono nel contratto. Il certificato
può cambiare il segno di uno zero F32 intermedio: non promettere byte-identità
di quell'array. Occorre dimostrare byte-identità dei codici e della scala hidden,
e delle uscite complete FFN. I WO originali restano invariati; l'esatta somma
delle sole colonne con codice hidden nonzero è già qualificata in453.

### Economia prima del C

Se p è la frazione di righe certificate, i coefficienti WI logicamente letti
sono D*F*(1-p/2) byte, più4F byte di norme e i workspace. Il formato mantiene
D*F byte di coefficienti e aggiunge4F byte per esperto. Non supporre che ogni
riga scelta abbia un numero diverso di byte o che la cache renda gratuite letture.
I costi coarse, Q², confronto, fine, quantizzatori, sparse WO e scale vanno pagati.
Le linee cache realmente indirizzate vanno enumerate includendo offset/allineamenti.
Questi sono conteggi logici; i byte DRAM restano da misurare con strumenti adeguati.

Prima di un'implementazione C costosa, predisporre UNA ammissione separata:

1. Congelare nuove sorgenti/protocollo/runtime e binding a474/retention/payload,
  96 capture native teacher original128 (1344posizioni) e fixture intere.
  Nessun nuovo forward/capture, fitting, rango o soglia scelta dopo osservazione.
2. Qualificare biiezione di TUTTI256 coefficienti, estremi dei dot, zero,
  uguaglianza del certificato, scale/cast e identità completa su tiny fixture.
3. Calcolare il certificato su TUTTE1344 posizioni originali bank11, senza scegliere
  gli ID. Ricostruire esattamente tutte le preattivazioni per le righe incerte,
  qualificare codici/scale hidden e uscite native complete. Conservare maschere,
  margini interi e conteggi per query/book/ID; nessuna scelta di centro o bitwidth.
4. Soglia proposta da congelare: p medio>=.60 e p medio in OGNI book>=.40.
  Equivale a risparmio logico coefficienti WI medio>=30%, per book>=20%, prima
  di norme/workspace. Primal non esatto, budget superato o soglia fallita:
  chiudere QUESTA codifica+certificato senza C, soglie adattate o griglie.
5. Budget di progetto per l'ammissione: CPU0/BLAS1,180s,2GiB,nuovi output<=32MiB;
  il limite reale deve includere TUTTI i binding e i margini/metadati prima del
  freeze. Se non sta nel budget, fare prima una revisione statica, senza numerics.
6. Solo dopo PASS e audit indipendente, congelare UNA nuova ricetta C completa
  coarse/fine+WO esatto. Gates di costo prima della compilazione: media FFN<=.80
  originale, ogni book<=1.00, p95<=1.00, workload fisico128esperti e primal esatto.
  I kernel454/456 chiusi non si ritimano come nuove prove.

Un PASS di questo certificato ammette un test di costo; non implica velocità,
qualità globale, risparmio DRAM, LUT scalabile o utile-n. Il limite di Cauchy
può essere troppo largo e certificare poche righe anche se molte ReLU sono zero.
Questo è il rischio concreto che il primo conteggio deve decidere.

## Altre vie e ordine delle dipendenze

- **Residui/sottospazi dipendenti dall'input:** possono seguire regioni diverse
  della funzione, ma bisogna pagare selezione, numero di basi, residui e fallback.
  Richiedono una nuova evidenza di riuso fra input; nessun nuovo clustering ora.
- **Adattamento congiunto del donor alla geometria:** può cambiare la distribuzione
  delle funzioni invece di proiettarla soltanto. Ha bisogno di una rappresentazione
  conveniente già ammessa, obiettivi donor-relative e held-out; il branch donor-
  adaptation resta una fonte di evidenze, non un operativo ripreso automaticamente.
- **Espandere n:** per n ID distinti, storage B(n) cresce; il costo per token dipende
  da k funzioni realmente consultate, dai loro byte e dal router. Non dedurre k o
  utilità dal solo numero di pesi. Top1 esatto non preserva la massa softmax.
  Con logit esclusi s_i in[l_i,u_i], il denominatore è contenuto fra la somma
  esatta dei logit visitati piùsum exp(l_i) e quella piùsum exp(u_i). Questo può
  dare intervalli per le probabilità; servono limiti geometrici utili e costo
  misurato per produrli. Nessun nuovo router dimostrato qui.

La catena resta: rappresentazione condizionale ammessa -> costo nativo ->
artefatto reale con tutti i bank -> nuovi stati/routing -> qualità donor-relative
su predizione/generazione/task -> SAME>=50 accepted batch1 IDs/s e DRAM ->
incrementi causali di utile-n -> altre famiglie/scale/~100B dove le risorse bastano.
Ridurre il costo esperti non cancella il costo del core, della head o del routing.
L'obiettivo non è ancora raggiunto.
