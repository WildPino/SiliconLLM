# Obiettivo della ricerca: quantizzazione ternaria progressiva

Determinare se, e a quali condizioni, un modello linguistico già addestrato
può essere convertito progressivamente da pesi floating point a codici ternari
`{-1, 0, +1}`, scegliendo le trasformazioni in funzione del comportamento del
modello originale, conservandone le capacità predittive e generative entro
tolleranze dichiarate e verificate su dati indipendenti, senza riaddestrarlo
da zero.

Quando praticabile, sviluppare una procedura riproducibile che migliori il
compromesso tra qualità, memoria e costo di esecuzione rispetto alla
ternarizzazione diretta e a baseline pertinenti. Scale ed eventuali componenti
di compensazione devono essere esplicite e conteggiate: un modello misto deve
essere descritto come tale. La compressione deve essere utile al percorso
`native-expert-scaling`, preservando le funzioni pretrained degli esperti e
rendendo più conveniente la loro memorizzazione e consultazione nel runtime
CPU del progetto.

Il risultato atteso è una decisione di fattibilità sostenuta da esperimenti,
con artefatti e istruzioni riproducibili. Se emerge un metodo valido, mostrarne
benefici, costi e limiti sullo stesso artefatto. Se le soluzioni esaminate non
sono convenienti, documentare precisamente quali ipotesi falliscono e perché,
senza trasformare un fallimento locale in una conclusione universale.

Un esito negativo rigoroso è un risultato completo della ricerca. La chiusura
deve seguire una ricognizione della letteratura e un insieme finito e motivato
di prove entro risorse e criteri dichiarati: non richiede di esaurire ogni
algoritmo possibile. Distinguere fallimenti misurati, limiti di risorse e
questioni rimaste non verificate.

Scegliere e adattare metodi, modelli e prossimi passi in base alle evidenze.
La ricerca casuale è una possibilità, non un vincolo. Calibrazione,
distillazione o adattamento limitato del checkpoint sono ammessi, purché il
loro costo sia misurato. Un buon fit locale non basta a dimostrare qualità
generale o velocità nativa.

Rigore e documentazione sono requisiti della ricerca: fissare ipotesi e criteri
prima delle prove, separare ottimizzazione e verifica, conservare risultati
grezzi e fallimenti, identificare codice e input e distinguere misure,
deduzioni e ipotesi. Operare nel worktree `research/progressive-ternary`,
preservando il lavoro dell'altro ricercatore. Le risorse disponibili sono una
RTX 3060, 80 GB di RAM e tre account Kaggle con quota nominale di 30 ore
settimanali ciascuno su T4×2; verificare quote effettive e coordinare l'uso
delle risorse condivise.

Il goal e la conversazione restano in italiano. La documentazione tecnica e
scientifica deve essere in inglese. Non firmare i documenti o commit e non citare o
attribuire il lavoro all'assistente, al suo modello o ai suoi strumenti.
