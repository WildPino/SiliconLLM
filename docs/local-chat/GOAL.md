# Goal della chat locale

Stato: proposta da rileggere e modificare prima di iniziare l'implementazione.
Data: 13 ottobre 2026.
Branch: `feature/local-chat`.

Realizzare un'interfaccia web locale per SiliconLLM, semplice come una normale
chat: scegliere un modello supportato, aprire conversazioni, inviare messaggi,
leggere le risposte e consultare statistiche reali. Tutta l'inferenza deve usare
il runtime approvato del progetto sul computer dell'utente.

Il prodotto deve rendere facile provare il motore preservandone qualità e
prestazioni su CPU. Il modello non deve avere strumenti per eseguire comandi,
leggere file dell'utente o interagire con il sistema operativo. Il ponte può
leggere gli artefatti configurati, gestire il processo di inferenza e salvare
le conversazioni: queste sono operazioni dell'applicazione, non capacità del modello.

## Risultato richiesto

- Un comando avvia il servizio locale e apre l'interfaccia nel browser.
- Selettore dei soli modelli installati, supportati e validati per il runtime.
- Cronologia locale persistente con creazione, rinomina e cancellazione delle chat.
- Messaggi leggibili, Markdown, codice copiabile, stati di caricamento ed errori comprensibili.
- Risposta progressiva soltanto se i token arrivano durante la generazione.
  Nessuna simulazione di streaming su una risposta già completa.
- Stop interrompe davvero il calcolo; se richiede di riavviare il processo,
  l'interfaccia lo comunica e la successiva richiesta riparte con stato coerente.
- Statistiche con definizioni esplicite: token generati, durata, velocità,
  memoria del processo e tempo al primo token quando misurabile.
- Una sola generazione attiva e un solo modello caricato nella prima versione.
- Nessuna dipendenza da un servizio remoto per l'uso di modelli già installati.

## Condizioni di completamento

Una conversazione reale su un artefatto approvato funziona per più turni,
si conserva dopo il riavvio e non mescola lo stato tra chat o modelli.
Stop, errori del motore, limiti di contesto e browser disconnesso sono gestiti.
Il risultato generato coincide con il percorso di riferimento, a parità di
artefatto, messaggi, tokenizer, template e politica di generazione.

Un confronto riproducibile misura l'overhead del ponte, dello streaming e della
pagina sullo stesso hardware. Budget proposto: perdita della velocità mediana
di decode non superiore al 5% rispetto al percorso diretto equivalente.
La soglia è un requisito proposto, non un risultato già ottenuto.
Se fallisce, si corregge o si propone esplicitamente una revisione del progetto.

La chiusura richiede inoltre istruzioni di avvio, verifiche mirate e integrazione
verificata con il ramo principale dopo la ricerca. Una demo simulata non soddisfa
il goal.

## Limiti della prima versione

Esclusi agenti, tool calling, RAG, allegati, ricerca web, account, cloud,
multiutente, generazioni parallele, download automatico dei modelli e un
installer desktop completo. Prima piattaforma: Windows x86-64 sull'hardware
supportato dal motore. Le altre piattaforme richiedono qualificazione separata.

Il progetto tecnico e il percorso di merge sono in [PROJECT.md](PROJECT.md).
