# Distribuzione e installazione dei modelli

Questo è un piano per una fase successiva. La prima versione resta una finestra locale che usa solo runtime e cataloghi già installati. Non implementa un download manager, un registro operativo o un servizio in background. Il goal rimane subordinato alla qualificazione di un runtime e di un artefatto reali.

## Scelta di distribuzione

Distribuire l'applicazione e i pesi come prodotti separati:

- Un installer Windows piccolo e firmato installa l'app, il runtime CPU nativo qualificato, tokenizer, template, ricevute e un catalogo iniziale minimo. Pubblicare anche un archivio portabile con gli stessi componenti per chi non vuole installare. I pesi non entrano nell'installer.
- Un indice statico, versionato e firmato descrive i soli modelli approvati. Pubblicarlo come piccolo asset della release GitHub dell'app, insieme ai metadati di release. Non serve un server personale sempre acceso: il client scarica l'indice quando apre la sezione modelli o su richiesta. Il runtime parte alla prima richiesta e resta caricato per la sessione, senza calcolo continuo quando è inattivo; si chiude con l'app o quando il modello viene scaricato dalla memoria.
- Ospitare i pesi nei repository modello di Hugging Face, fissando ogni URL a un ID di repository e a una revisione immutabile. È una scelta adatta agli artefatti ML: le linee guida del Hub raccomandano file sotto 200 GB e suggeriscono di suddividere i file grandi; il limite singolo dichiarato è 500 GB. Lo storage pubblico è best-effort e soggetto a policy e quote, quindi il progetto deve verificare costi, licenze e capacità prima di pubblicare. [Limiti e raccomandazioni di storage Hugging Face](https://huggingface.co/docs/hub/en/storage-limits)

Per le release GitHub, ciascun asset deve essere inferiore a 2 GiB. Per questo GitHub è adatto all'app, all'indice e a piccoli bundle di runtime; non è la destinazione dei pesi. L'indice deve restare leggero anche quando cresce il catalogo. Questi limiti descrivono la documentazione corrente dei servizi, non una garanzia permanente di spazio, velocità, disponibilità o gratuità. [Limiti degli asset GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

Se volumi, termini o costi rendono inadatto l'hosting pubblico, confrontare prima un object storage con CDN gestito dal progetto. Non introdurre un account utente o un servizio personale sempre attivo come prerequisito per l'uso locale. Il repository modello deve autorizzare la redistribuzione dei pesi e dei file accessori; un URL pubblico non concede di per sé tale licenza.

## Percorso dell'utente

1. L'utente installa l'app oppure apre l'archivio portabile.
2. La sezione **Modelli** mostra una scelta consigliata compatibile con la CPU, la RAM e il sistema rilevati. Ogni scheda espone revisione, lingua e capacità, RAM di sistema consigliata, spazio occupato, dimensione del download, licenza e sorgente. Le stime sono prudenti e indicano i limiti del profilo misurato; l'utente decide prima di iniziare un download grande.
3. **Scarica** scarica o riprende i file in una directory temporanea. Sono visibili progresso, velocità, tempo stimato, pausa/cancellazione e spazio richiesto. Prima di iniziare, l'app controlla lo spazio libero per i file temporanei e per l'installazione finale.
4. L'app verifica firma dell'indice e SHA-256 di ogni file. Solo dopo una verifica completa installa con rinomina atomica nella directory dei modelli e rende il modello selezionabile. Un errore lascia il modello non disponibile e consente di riprovare; non lascia artefatti parziali utilizzabili.
5. Dopo l'installazione, la chat funziona offline. Una nuova versione dell'app non scarica di nuovo i pesi già validati; un aggiornamento del modello è esplicito e può mantenere la revisione precedente finché la nuova non è verificata.

L'utente normale non deve scegliere percorsi, argomenti di processo o compilare ricevute. L'importazione di un catalogo locale può restare disponibile nelle impostazioni avanzate per sviluppo e modelli approvati installati manualmente. Il catalogo pubblico contiene solo modelli già convertiti nel formato nativo supportato e qualificati. Non offrire un selettore generico di GGUF/Ollama: supportare un formato richiede un runtime, una conversione e una qualificazione espliciti.

Tutti i download e le verifiche sono rinviati mentre una generazione è attiva. Evita contesa di CPU, memoria e disco con il processo di inferenza. La chat non avvia un demone HTTP locale: la finestra possiede i processi di download e verifica, e li termina quando chiude o quando l'utente cancella l'operazione.

## Indice firmato e pacchetto del modello

La chiave privata di firma resta offline nel processo di release. L'app include la chiave pubblica radice e verifica la firma dell'indice prima di usarlo. Ogni release dell'indice ha una versione e un digest; ogni modello ha una revisione immutabile upstream. La firma del manifesto attesta l'approvazione del catalogo; gli hash dei file verificano che il download corrisponda ai byte dichiarati. La ricevuta di qualificazione è un documento separato, emesso dal processo di qualifica del runtime, e lega runtime, tokenizer, template, pesi e parametri. Non trattare un campo `approved: true` scritto dall'utente come prova di approvazione del catalogo pubblico.

Schema iniziale proposto per l'indice `schemaVersion: 1`:

```json
{
  "schemaVersion": 1,
  "catalogVersion": "2026.10.0",
  "models": [{
    "id": "example-cpu-int4",
    "name": "Example CPU 4-bit",
    "revision": "1.0.0",
    "runtimeId": "silicon-cpu-runtime-1",
    "protocol": "silicon-chat-v1",
    "platforms": ["windows-x86_64"],
    "capabilities": {"streaming": true, "maxContext": 8192},
    "requirements": {"ramBytes": 8589934592, "diskBytes": 4300000000},
    "license": {"id": "license-id", "url": "https://example.org/license"},
    "source": {"repository": "owner/model", "revision": "immutable-commit-id"},
    "files": [{
      "path": "weights-0001.safetensors",
      "url": "https://huggingface.co/owner/model/resolve/REV/weights-0001.safetensors",
      "sizeBytes": 2140000000,
      "sha256": "<64 hex characters>"
    }],
    "runtimeBundle": "silicon-cpu-runtime-1",
    "qualificationReceipt": "receipts/example-cpu-int4.json"
  }]
}
```

Le URL sono generate e controllate durante la release, non accettate dal testo del modello. Il client accetta soltanto host e percorsi consentiti nell'indice firmato. Prima dell'estrazione controlla che ogni nome di file resti sotto la directory temporanea designata; rifiuta percorsi assoluti, `..`, archivi con collisioni o dimensioni diverse dal manifesto. Il pacchetto include tutti i file necessari al runtime, tokenizer, template e ricevuta, mentre i pesi possono essere scaricati in più shard. Il runtime aggiornato viene installato separatamente dai pesi quando la compatibilità lo consente.

Stato visibile per ogni versione del modello:

`not installed → downloading → verifying → ready`

Un errore porta a `failed` con azione di riprova; una versione rimossa o non compatibile diventa `unavailable` e non viene avviata. La cancellazione durante `downloading` elimina o conserva solo segmenti temporanei con etag e digest validi per una ripresa successiva. La cancellazione durante `verifying` ferma la verifica e impedisce l'installazione. Lo stato `ready` si ottiene solo dopo firma, hash, ricevuta, dipendenze e spazio finale verificati.

## Dati necessari prima della pubblicazione

Per pubblicare modelli utilizzabili servono un runtime e un modello reali approvati, licenza e permessi di redistribuzione documentati, endpoint a revisioni immutabili e misure di spazio/RAM per le configurazioni supportate. Il client può essere sviluppato e testato prima usando file dedicati ai test, senza presentarli come modelli disponibili. Prima del rilascio verificare firma e hash, ripresa, cancellazione, spazio insufficiente e aggiornamenti; stimare storage e traffico sul servizio scelto. La lista online descritta qui è ancora un piano, non una funzione disponibile.
