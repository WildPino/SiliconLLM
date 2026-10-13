# Runtime locale: protocollo e qualificazione

La finestra è un'applicazione C++17 con Qt Widgets. `EngineSession` avvia un
processo locale con `QProcess`, usa `stdin` e `stdout` per il protocollo e legge
`stderr` separatamente. Non apre porte di rete e non interpreta il testo del
modello come comandi. Il protocollo `silicon-chat-v1` qui descritto è il
contratto lato app per il futuro adattatore; non dichiara che un motore di
ricerca già disponibile lo implementi o sia qualificato per la chat.

## Avvio e proprietà del processo

L'applicazione avvia il programma e gli argomenti dichiarati nel catalogo, con
la directory di lavoro configurata. Sostituisce `{artifact}` e `{tokenizer}`
negli argomenti con i percorsi canonici verificati. Un solo processo e una sola
generazione sono attivi alla volta. L'app non accoda invii.
Per i processi Python imposta `PYTHONUTF8=1`, così la codifica del protocollo
non dipende dalla code page di Windows.

Su Windows il processo è assegnato a un Job Object configurato per terminare i
processi del gruppo quando il job viene chiuso. L'adattatore deve restare nel
processo avviato e non creare processi figli durante l'avvio o prima della
prima richiesta. Questo rende verificabile la proprietà del gruppo prima che
cominci il lavoro del modello e consente a Stop e alla chiusura dell'app di
terminarlo. Stop chiude il job, interrompe il processo e ne crea uno nuovo al
successivo invio; l'adattatore non deve dichiarare che una richiesta interrotta
continua nel successivo processo.

## Flusso JSON Lines

Gli eventi sono oggetti JSON UTF-8, uno per riga su `stdout`, terminati da `\n`.
Una riga non può superare 1 MiB; la risposta visibile non può superare 16 MiB.
Le righe vuote vengono ignorate. `stderr` è riservato ai log e l'app conserva
soltanto gli ultimi 16 KiB per i messaggi di errore.

All'avvio, l'adattatore invia una sola dichiarazione di capacità:

```json
{"type":"ready","protocol":"silicon-chat-v1","max_context":8192,"streaming":true}
```

`protocol`, `max_context` e `streaming` devono coincidere esattamente con il
catalogo. L'app non invia richieste prima di aver ricevuto e verificato `ready`.

La richiesta contiene il transcript completo e riparte sempre da stato vuoto:

```json
{"protocol":"silicon-chat-v1","type":"generate","request_id":"uuid","conversation_id":"uuid","reset":true,"messages":[{"role":"user","content":"Ciao"}],"max_new_tokens":256}
```

`request_id` identifica la richiesta e deve comparire in ogni evento successivo,
compresi errore e completamento. `conversation_id` identifica la chat. `reset`
è sempre `true`; non riutilizzare cache o stato di una richiesta precedente.
`messages` è l'intera sequenza ordinata di messaggi con ruolo `system`, `user` o
`assistant` e contenuto testuale. `max_new_tokens` non supera il limite
dichiarato nel catalogo.

Prima di avviare il decode, l'adattatore deve applicare il tokenizer, il template
e le regole di generazione legati alla ricevuta. Deve misurare il prompt
tokenizzato completo e verificare prompt più limite di generazione contro
`max_context`. Se non entra, deve inviare un errore leggibile senza inferenza.
Non troncare il transcript e non riassumerlo automaticamente. Il controllo di
contesto è parte dell'adattatore qualificato: l'attuale `EngineSession` invia
il transcript e verifica le capacità `ready`, ma non possiede il tokenizer e
non può da solo fare questo controllo.

Durante la richiesta, il processo può inviare:

```json
{"type":"status","request_id":"uuid","stage":"prefill"}
{"type":"status","request_id":"uuid","stage":"decode"}
```

Sono ammesse le fasi `prefill` e `decode`. Un nome di fase sconosciuto è un
errore di protocollo.

Solo un modello dichiarato `streaming: true` può inviare delta. `text` è il
frammento nuovo da aggiungere alla risposta; `generated_tokens` è il conteggio
cumulativo degli ID generati, non il numero di caratteri e non il numero di
delta:

```json
{"type":"delta","request_id":"uuid","text":"Ciao","generated_tokens":1}
```

Il testo deve arrivare mentre il decode è in corso. Un client non può simulare
delta dopo aver già prodotto tutta la risposta. Con `streaming: false` non
inviare eventi `delta`: mostrare soltanto il testo completo al termine.

Il completamento restituisce il testo canonico completo, le statistiche e il
motivo di terminazione:

```json
{"type":"completed","request_id":"uuid","text":"Ciao!","statistics":{"generated_tokens":2,"prefill_seconds":0.04,"decode_seconds":0.12,"load_seconds":1.3,"working_set_bytes":734003200},"termination":"eos"}
```

`generated_tokens` è obbligatorio, intero e non supera il limite richiesto.
`prefill_seconds`, `decode_seconds`, `load_seconds` e `working_set_bytes` sono
facoltativi; se presenti devono essere numeri finiti non negativi. Indicare
`null` o omettere una misura non disponibile. `termination` vale `eos` o
`length`. Se sono stati inviati delta, `text` deve coincidere byte per byte,
come stringa Unicode, con la loro concatenazione e il conteggio finale non può
essere inferiore al conteggio già annunciato.

Un errore usa una risposta associata alla richiesta:

```json
{"type":"error","request_id":"uuid","message":"Il prompt supera il contesto supportato."}
```

Il runtime termina la richiesta dopo l'errore. L'app interrompe anche il
processo se il protocollo non è valido, la richiesta va in timeout, il processo
termina inatteso o un evento supera i limiti. Non aspettare altri comandi su
`stdin` mentre il processo è bloccato nel decode: Stop termina il gruppo di
processi dall'esterno.

## Statistiche mostrate

`generated_tokens` e i tempi di prefill, decode e caricamento provengono dal
runtime. L'app calcola `total_seconds` con un timer monotono a partire dall'invio,
quindi include avvio/caricamento al primo turno e trasporto. La velocità mostrata
è `generated_tokens / decode_seconds`; non è una misura di parole o caratteri.

`first_token_seconds` misura l'attesa fino al primo contenuto disponibile nel
widget di testo della finestra. Per il percorso progressivo parte dall'invio e
termina quando la UI inserisce il primo contenuto non vuoto; non misura la
presentazione fisica del frame sul display. Se il modello non
trasmette progressivamente, questa statistica resta non disponibile: non si
sostituisce con il tempo interno del motore o con il tempo di completamento.

Su Windows `working_set_bytes` è il working set del processo radice avviato da
Qt, campionato al completamento o all'interruzione. Se l'eseguibile radice è un
wrapper, la misura esclude la memoria dei processi figli; l'adattatore deve
riportare chiaramente il perimetro della misura e non sommare i figli senza
qualificare la metodologia.

## Catalogo e ricevuta

Il file del catalogo ha `schemaVersion: 1` e un array `models`. Un catalogo
valido può contenere zero modelli: `apps/local-chat/resources/catalog.example.json`
è intenzionalmente vuoto. Ogni voce deve dichiarare `id`, `name`, `revision`,
`protocol`, `executable`, `artifact`, `tokenizer`, `working_directory`,
`arguments`, `max_context`, `max_new_tokens`, `timeout_seconds`, `streaming` e
`qualification`. `python` è facoltativo. La sola versione supportata è
`silicon-chat-v1`; gli altri protocolli restano indisponibili.

I percorsi relativi sono risolti dalla directory del catalogo. I percorsi dei
file vengono canonicalizzati; `working_directory` deve essere una directory
esistente. `max_context` e `max_new_tokens` sono interi positivi e il limite di
generazione non può superare il contesto. `timeout_seconds` è un intero da 1 a
3600. Gli argomenti sono stringhe letterali; solo `{artifact}` e `{tokenizer}`
vengono sostituiti.

La qualificazione non si deduce dalla presenza dei file. La voce deve avere:

```json
"qualification": {
  "approved": true,
  "receipt": "qualification.json",
  "executable_sha256": "<64 cifre esadecimali>",
  "artifact_sha256": "<64 cifre esadecimali>",
  "tokenizer_sha256": "<64 cifre esadecimali>",
  "files": [
    {"path":"adapter.py","sha256":"<64 cifre esadecimali>"}
  ]
}
```

La ricevuta JSON deve contenere `admitted: true`, `scope: "cpu-runtime"`, gli
stessi tre hash e gli stessi valori per `protocol`, `max_context`,
`max_new_tokens`, `streaming`, `timeout_seconds` e `arguments`. Deve inoltre
contenere `working_directory`, risolto rispetto al catalogo e confrontato con
la directory canonica configurata per il processo. Se `qualification.files` è
presente, `receipt.files` deve essere identico, compreso l'ordine delle voci.
Ogni helper dichiarato è controllato con SHA-256 a blocchi da 1 MiB. Se nella
directory del tokenizer esistono `tokenizer_config.json`, `chat_template.jinja`
o `chat_template.json`, devono essere inclusi in `qualification.files`. Anche
i file effettivamente indicati negli argomenti devono essere inclusi. Se
`executable` è l'interprete Python, l'argomento dello script `.py` deve
esistere ed essere coperto dalla lista. Un percorso `python` non vuoto va
anch'esso incluso e hashato in `files`.

La verifica del catalogo avviene in un worker e può leggere l'intero artefatto
per validarne l'hash. Cambiare binario, pesi, tokenizer, template, helper o
configurazione degli argomenti richiede una nuova ricevuta coerente. I modelli
non qualificati o non disponibili restano esclusi dal selettore; l'app non
scarica né sostituisce artefatti.

## Stato della qualificazione

Al momento non è disponibile un runtime CPU ammesso per questa app. I test usano
un processo fixture deterministico per verificare il protocollo e il
comportamento del ponte. Non provano la qualità di un modello, la parità del
tokenizer con il runtime di ricerca o le prestazioni. Il catalogo vuoto è quindi
lo stato iniziale previsto; non aggiungere una voce approvata solo per far
apparire un modello nell'interfaccia.
