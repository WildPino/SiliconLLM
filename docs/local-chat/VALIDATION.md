# Stato e verifiche della chat desktop

13 ottobre 2026. Il goal è in pausa e incompleto. La finestra desktop e il
collegamento ai processi sono implementati; la prova su un modello reale
dipende ancora da un runtime CPU ammesso. Nessun artefatto di ricerca è stato
aggiunto al catalogo né modificato per ottenere una demo.

## Build e test locali

Build Release riuscita su Windows 11 con Qt 6.8.3, MinGW GCC 13.1, CMake 4.2.3
e Ninja. Il wrapper `apps/local-chat/build.ps1` è stato eseguito realmente e
compila con un solo job. I subprocessi di test usano Python 3.14 e JSON UTF-8.

Quattro suite CTest passate, nessuna fallita:

| Suite | Esito QtTest | Evidenza |
| --- | --- | --- |
| Store e catalogo | 8 pass, 0 fail | [store.txt](validation/20261013/store.txt) |
| Protocollo e processo | 18 pass, 0 fail | [engine.txt](validation/20261013/engine.txt) |
| Finestra | 11 pass, 0 fail | [ui.txt](validation/20261013/ui.txt) |
| Controller e persistenza | 6 pass, 0 fail | [controller.txt](validation/20261013/controller.txt) |

I conteggi QtTest comprendono inizializzazione e chiusura di ogni suite.
Le fixture sono processi deterministici dedicati ai test, non modelli LLM.

Le verifiche comprendono salvataggi e ricaricamento, isolamento tra chat e
modelli, hash modificati, Unicode, testo prima del completamento, misura del
primo contenuto UI distinta dal completamento, timeout, protocollo invalido,
richieste concorrenti rifiutate, Stop e chiusura che terminano anche un figlio
del processo su Windows. Il Markdown non carica risorse locali, remote o data URL.

## Pacchetto e verifica della finestra

Il pacchetto portabile è generato da `apps/local-chat/package.ps1`. La verifica
ha avviato l'eseguibile confezionato con PATH limitato alle directory Windows,
senza Qt o compilatore nel PATH, e ha acquisito una schermata tramite
`--capture`; il processo è terminato con codice 0. La schermata è stata
ispezionata. Non è presente un browser o una WebView.

Il packaging esclude i plugin generici touch e di rete non utilizzati. Il
pacchetto include DLL Qt, runtime MinGW, plugin di piattaforma/stile, licenze
e istruzioni. Non include pesi, tokenizer, Python o fixture di test.

Il workflow Windows in `.github/workflows/local-chat.yml` è predisposto ma non
è stato eseguito su GitHub. Nessun push o merge è stato effettuato.

## Redesign 0.2.0

La revisione successiva 0.2.1 traduce tutta l'interfaccia in inglese, usa pulsanti
affiancati nelle impostazioni e un selettore del modello piatto, con uno stile
unico per entrambi i temi e senza opzione dei contorni. Le quattro suite passano;
schermate e verifiche del pacchetto sono in
[validation/20261013-ui-english](validation/20261013-ui-english/README.md).
I numeri riportati sotto restano riferiti al confronto 0.2.0.

Quattro suite passate sulla versione finale: store 8, engine 18, UI 15,
controller 6, zero fallimenti. I nuovi casi verificano preferenze salvate,
menu delle conversazioni, larghezza/altezza dei messaggi e aggiornamento
visibile durante un flusso continuo più rapido del timer della finestra.
I log sono in [validation/20261013-ui-refresh](validation/20261013-ui-refresh/).

Il timer single-shot ora parte solo se non è già attivo. Prima ogni delta
riavviava l'attesa di 75 ms: un flusso continuo a 20 ms impediva gli
aggiornamenti intermedi dopo il primo testo. Questa correzione è applicata
anche alla baseline della misura per confrontare lo stesso lavoro visibile.

Temi chiaro/scuro e impostazioni sono stati acquisiti dall'eseguibile
portabile finale con PATH limitato alle directory Windows, quattro avvii
terminati con codice 0. La revisione visiva comprende anche dodici messaggi
sintetici con Markdown/codice, in entrambi i temi. Nessun download manager
o modello utilizzabile è incluso nel pacchetto 0.2.0.

Le misure della UI usano finestre native 1180×780, build Release identiche
per toolchain, sette ripetizioni di cinque secondi per tema/scenario,
settling iniziale di 600 ms e testo sintetico a 50 eventi al secondo.
La baseline è `0c75b42` con la sola correzione del timer e il medesimo probe.
Metodo e dati grezzi: [method.txt](validation/20261013-ui-refresh/method.txt)
e [comparison.json](validation/20261013-ui-refresh/comparison.json).

Mediane delle sette ripetizioni; CPU è tempo cumulato del processo nei cinque
secondi della misura, non percentuale di utilizzo dell'intero computer:

| UI | Scenario | CPU ms (min–max) | Memoria privata MiB | Working set MiB |
| --- | --- | --- | --- | --- |
| Precedente | Inattiva | 0 (0–0) | 16,59 | 51,05 |
| Chiara | Inattiva | 0 (0–0) | 10,51 | 35,86 |
| Scura | Inattiva | 0 (0–0) | 10,52 | 35,92 |
| Precedente | Streaming sintetico | 703,125 (671,875–734,375) | 19,10 | 53,20 |
| Chiara | Streaming sintetico | 609,375 (546,875–656,25) | 11,82 | 37,07 |
| Scura | Streaming sintetico | 609,375 (562,5–656,25) | 11,98 | 36,85 |

Nel carico misurato la CPU mediana diminuisce del 13,3% in entrambi i temi;
nessuna regressione emerge nelle sette ripetizioni. È un risultato locale
per questi scenari, non una garanzia per ogni hardware e lunghezza di chat.
Il pacchetto portabile occupa 28,81 MiB (21 file), l'archivio ZIP 12,32 MiB;
non include probe, test, Python, pesi o runtime di inferenza utilizzabili.

Le misure escludono startup e formattazione finale del Markdown. CPU del
processo, memoria privata e working set provengono dalle API Windows;
zero CPU significa zero rilevato alla granularità della misura, non una
garanzia matematica. Non esercitano inferenza, tokenizer, download né
contesa con un motore CPU: il requisito sul decode resta da verificare.

## Requisiti ancora da provare sul runtime

| Requisito del goal | Stato |
| --- | --- |
| Conversazione reale utile per più turni | Manca il runtime CPU ammesso e il suo artefatto completo |
| Tokenizer e template identici al riferimento | Richiede l'adattatore del runtime finale e una prova di parità |
| Streaming del modello durante il decode | Protocollo/UI verificati con fixture; capacità reale ancora da qualificare |
| Stop sul motore reale e ripristino dello stato | Processo e discendenti verificati con fixture; runtime reale ancora da provare |
| Statistiche del motore reale | Gestione e perimetro delle misure implementati; misure reali ancora da acquisire |
| Overhead decode entro il 5% | Nessun risultato: richiede confronto riproducibile sullo stesso runtime/artefatto |
| Integrazione sul ramo principale dopo la ricerca | Attende la chiusura positiva e l'integrazione della ricerca |

Il primo passo della connessione reale è identificare eseguibile, artefatto,
tokenizer/template e politica di generazione ammessi, quindi implementare e
qualificare il loro adattatore `silicon-chat-v1`. Non sostituire questo passo
con l'approvazione manuale di una fixture o di un modello diverso.
