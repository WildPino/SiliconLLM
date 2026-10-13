# Direzione visiva della chat locale

Il redesign 0.2.0 usa Qt Widgets, superfici opache e sottili bordi luminosi statici. Sono implementati temi chiaro/scuro, dimensione del testo, statistiche opzionali e preferenze salvate localmente. Non usa blur, ombre grafiche o animazioni continue. Il confronto misurato della UI e i suoi limiti sono in [VALIDATION.md](VALIDATION.md).

## Gerarchia

Il contenuto della conversazione occupa il piano principale: messaggi, codice, stato della generazione e compositore restano leggibili e stabili. Controlli e comandi formano un piano distinto e compatto, con selettore del modello, cronologia e azioni riconoscibili. La trasparenza è una scelta di gerarchia per la navigazione, non un effetto applicato ai messaggi. Questo segue la separazione Apple tra un livello funzionale di controlli e navigazione e il canvas dedicato al contenuto; Liquid Glass è una direzione specifica delle piattaforme Apple, non un materiale da imitare letteralmente in Qt. [Apple, Meet Liquid Glass, WWDC25](https://developer.apple.com/videos/play/wwdc2025/219/)

La finestra deve sembrare un'app desktop calma: superfici opache per testo e trascrizione, contrasto chiaro, separatori sobri e un solo accento per l'azione principale. Preferire bordi statici e angoli coerenti. Modalità chiara e scura devono rispettare la preferenza dell'utente e mantenere lo stesso ordine e la stessa gerarchia. Evitare pannelli di vetro sovrapposti, blur sull'area dei messaggi, ombre diffuse e gradienti decorativi.

Windows Mica è un materiale opaco e dinamico previsto per finestre di lunga durata e campiona lo sfondo del desktop una volta per la visualizzazione; non è il Liquid Glass di Apple e non va presentato come tale. Se il materiale di sistema è sperimentato in futuro, usarlo solo come base opzionale della finestra, rispettare la modalità chiara/scura e lasciare opaco il piano del contenuto. [Microsoft, Mica material](https://learn.microsoft.com/en-us/windows/apps/design/style/mica)

## Regole pratiche per il rendering

- Nessun `QGraphicsBlurEffect`, ombra diffusa o effetto continuo dietro i messaggi. Nessun ciclo di rendering o animazione decorativa mentre la chat è inattiva.
- Aggiornare il testo progressivo a intervalli raggruppati e solo nelle viste che cambiano; mantenere visibile Stop senza spostare la composizione.
- Usare i controlli Qt nativi dove danno comportamento desktop coerente. Per adattare i materiali di sistema è preferibile un'integrazione Win32 puntuale e misurata al porting dell'intera finestra a un altro framework.
- Dare priorità a contrasto, navigazione da tastiera, dimensione del testo, selezione e copia del codice. Lo streaming non deve provocare ricalcolo e rilayout costante dell'intera cronologia.

## Prove di revisione

I test coprono temi e preferenze, menu delle conversazioni, stati dei controlli e streaming continuo. La revisione visiva comprende finestra vuota, impostazioni e transcript sintetico in chiaro e scuro. Il confronto delle risorse usa la stessa dimensione e lo stesso flusso su Windows; non attesta ancora l'overhead del motore CPU reale.

La cronologia mantiene Nuova chat come azione visibile, con Rinomina/Elimina nel menu contestuale. Il selettore del modello e le impostazioni sono nel piano di navigazione superiore; il compositore espone una sola azione primaria, Invia oppure Stop. Il testo mantiene un carattere di sistema e una larghezza di lettura contenuta, senza font o risorse scaricate.
