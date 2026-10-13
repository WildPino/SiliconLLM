# Stato e verifiche della chat desktop

13 ottobre 2026. Il goal è attivo e incompleto. La finestra desktop e il
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

## Requisiti ancora da provare

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
