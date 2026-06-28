# Ripensare l'LLM per la CPU — Spiegazione completa

*Serie pedagogica, 2026-06-26. Scritta per essere letta da sola, anche a distanza di settimane, senza dover ricordare il gergo tecnico.*

---

## A cosa serve questa serie

Il 26 giugno 2026 abbiamo fatto uno **sprint di ricerca**: prendere un LLM "tradizionale", smontarlo pezzo per pezzo come fosse una macchina, e chiederci per ognuno *"si può sostituire? fondere con un altro pezzo? girare al contrario?"* — sempre con un unico obiettivo: **farlo girare velocissimo su una CPU normale** (non una scheda grafica costosa).

Questi 5 documenti spiegano **cosa abbiamo trovato e perché**, in ordine, dal più semplice al più specifico.

Esiste anche un quaderno tecnico denso (`../exploration.md`): è il mio appunto di lavoro, scritto in stenografia. **Questa serie è la versione spiegata.**

---

## I 5 documenti

| # | Documento | Cosa spiega |
|---|-----------|-------------|
| **01** | [Fondamenti: cos'è un LLM e perché su CPU il collo è la memoria](01_Fondamenti_LLM_e_CPU.md) | Le basi. Un LLM è una grossa funzione matematica. Su CPU il problema non è "quanto calcola" ma "quanto in fretta legge i suoi numeri dalla RAM". Qui costruiamo questa intuizione da zero. |
| **02** | [Le cinque idee](02_Le_Cinque_Idee.md) | Le 5 idee buone sopravvissute allo smontaggio, spiegate una a una in parole semplici: cosa sono, perché aiutano su CPU, quanto sono originali, cosa dice la ricerca. |
| **03** | [L'architettura unificata](03_Architettura_Unificata.md) | Le 5 idee non sono una lista: convergono in **una sola architettura**. Il principio: *"separa il pensare dal sapere"* / *"rendi la parte pesante un consultare, non un moltiplicare"*. |
| **04** | [Le prove e i prossimi passi](04_Prove_e_Prossimi_Passi.md) | Le 3 ricerche di approfondimento (cosa abbiamo chiesto, cosa è venuto fuori, cosa abbiamo **scartato**), e gli esperimenti economici da fare per verificare. |

---

## Sommario in una pagina (se hai fretta)

**Il problema.** Vogliamo un modello linguistico (un "LLM") che scriva testo molto velocemente su una CPU comune, non su GPU. Su CPU, quando si genera testo una parola alla volta, il collo di bottiglia **non è la potenza di calcolo**: è la **velocità con cui si leggono i numeri del modello dalla memoria RAM**. Quindi tutto si riduce a: *leggere meno byte dalla memoria per ogni parola prodotta.*

**Cosa abbiamo fatto.** Abbiamo smontato le 10 "stazioni" di un LLM tradizionale e per ognuna abbiamo provato a sostituire / fondere / invertire il meccanismo, scartando ciò che su CPU non paga.

**Cosa è emerso — 5 idee che convergono in una.** Invece di un muro di idee scollegate, ne sono rimaste 5 che si incastrano in **una sola architettura coerente**:
- **K2** — usa *un solo* blocco di calcolo, riusato più volte, abbastanza piccolo da stare nella memoria veloce della CPU (la "cache"). Così "pensa di più senza rileggere niente". *Testabile subito.*
- **K1** — trasforma la parte più pesante del modello (e l'uscita finale) in una **grande tabella da consultare** invece che in una moltiplicazione enorme: tocchi solo le poche voci che servono. *Conviene solo sui modelli grandi.*
- **K3** — **allena il modello a essere prevedibile** su *quali* numeri gli serviranno dopo, così li si può andare a prendere in anticipo (come fa la CPU coi salti di programma). *È il pezzo più originale: nessuno lo fa così.*
- **K4** — entra ed esce dal modello a "unità di compressione" (l'anima storica del progetto, SEE).
- **K5** — produce più parole per ogni lettura dei pesi.

**La frase che riassume tutto:** *"Rifattorizza l'LLM così che la parte pesante sia un **consultare** (lookup), non un **moltiplicare**."* Il moltiplicare obbliga a leggere tutti i numeri; il consultare ne tocca pochi, e — se il modello è prevedibile — quei pochi sono già pronti nella cache.

**Cosa abbiamo scartato (vale quanto ciò che teniamo).** I modelli "a diffusione" (non-autoregressivi) su questa CPU **perdono**, verificato. E l'esplorazione di strumenti presi da altri campi (trasformate di Fourier, automi cellulari, ecc.) **non ha prodotto nuove idee vincenti**: ha solo rafforzato le 5 che già avevamo. L'abbiamo detto onestamente invece di inventare novità inesistenti.

**Dove sta l'originalità.** Non nei singoli mattoni (esistono tutti in letteratura). Sta nella **fusione** e nel fatto di progettare tutto **da zero, su misura per una CPU comune** — una nicchia che la ricerca pubblicata lascia sistematicamente vuota.

**Cosa NON cambia adesso.** Niente di tutto questo tocca il lavoro in corso (la "rampa recall run-6"). Sono idee per il **modello grande futuro**. L'unica testabile da subito, con un esperimento economico, è **K2**.
