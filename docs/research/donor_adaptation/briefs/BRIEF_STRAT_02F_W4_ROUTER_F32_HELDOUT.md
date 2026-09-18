# STRAT-02F — heldout follow-on del W4 con router F32

**Preregistrato il 18 settembre 2026, prima di qualunque heldout score di
questa cella.** Si testa esattamente un candidato: l'export completo
W4-v2 già verificato, con soltanto le 16 matrici router ripristinate F32
dalla sorgente StdMoE fissata. È lo stesso braccio
`ROUTER_F32` di STRAT-02D, il cui paired calibration output ha SHA-256
`f7913767de5c65396d90473cfd7302961c615ec09bcc0eeb65b3ff428a78550b`.
Nessun altro braccio, tuning o selezione dopo uno score è autorizzato.

Questa è una verifica paired della sola ipotesi che il gain router-F32 visto
in calibration possa portare il candidato heldout entro il gate. Il W4
heldout precedente è `FAIL_BPB` con Δ **+0.0260248576280077**; il gain
router-F32 in calibration è **+0.00691488289323727**. L'heldout è quindi
stato già usato per l'adjudication W4, ma questo candidato router-F32 non è
stato ancora scored su heldout: è un follow-on preannunciato, non un test
esterno intatto dell'intero programma di ricerca.

## Identità congelata

La sorgente è `allenai/StdMoE_1b14b_1T_Preanneal` @
`d2a4949c9d4ad6cf47fbac131f7e020077332b21`; verificare tutti gli 11 shard
F32 contro il manifest pin-nato prima di aprire valori. Il candidato è
esattamente l'export in
`benchmarks/donor_adaptation/density/results/strat02_w4_bf16_full_export_20260917_143753`,
con questi vincoli già pubblicati:

- `artifact_manifest.json` SHA-256
  `06924c13b5ad2106929ee51c84442bf63b4b47c6528e90b1ea91de891d4a0a5c`;
- `plan_manifest.json` SHA-256
  `1c1512b3735f0f9a80b638908801dfba7894b781047c767078a8b0cba54c3e42`;
- export `supervisor_result.json` SHA-256
  `d52e09748d594b26af89b8223dfcd372343e7b344bc9f10199d379e5e4c7faca`.

Il W4-v2 completo resta identico salvo le 16 matrici router F32 copiate
dalla sorgente pin-nata. Tutti gli altri record W4 e i 34 passthrough F32
restano invariati. Usare lo stesso tokenizer, EOS-prefix, assenza di
troncamento, chunk head 128, operatori F32/eager e costanti di corpus del
quality runner W4; registrare le identità di corpus e tokenizer nel manifest
senza rigenerare o sostituire la slice.

## Preflight e parità

Prima di qualunque heldout forward, verificare:

1. gli 11 shard sorgente, gli hash dell'export e i file di controllo;
2. il keyset esatto delle 16 matrici router previste e l'assenza di qualunque
   altra sostituzione;
3. la sentinella W4 prima del copy;
4. un ciclo di prova copy F32 → sentinella calibration contro la prima riga
   `ROUTER_F32` entro `1e-5` bit → rollback bitwise ai byte W4 →
   sentinella W4 entro `1e-5` bit;
5. la seconda copia F32 delle sole 16 matrici router, con gli SHA dei
   tensori copiati verificati prima di iniziare lo scoring heldout.

Dopo lo scoring heldout, ripristinare ancora i router W4 e verificare SHA
bitwise e sentinella W4. Se questo controllo finale fallisce, l'intero run
è `VOID_APPARATUS` anche se le 96 righe sono state scritte.

Il controllo calibration è operativo e non autorizza tuning o scelta di arm.
Qualunque mismatch di identità, hash, score sentinella, copia o rollback è
`VOID_APPARATUS` e vieta lo heldout. Non aprire testi, token ID o logits
oltre a ciò che serve al core di scoring; non leggere score heldout durante
il preflight o per cambiare configurazione.

## Heldout e adjudication

Se tutti i controlli passano, score **una sola volta** esattamente le 96
righe heldout nell'ordine originale. Il corpus è quello già congelato:
746161 byte, con teacher scores SHA-256
`96d284da8f223d613039cb9e2a87b24e7b6130701c0873eea0854c3e8290b224` e
W4 heldout candidate control SHA-256
`0ecbac5406b926240ddba3dc9eb25ca5239eb37ba717a5970aee7fb192a8039f`.
Riutilizzare senza variazioni le costanti del W4 quality runner, inclusi
tokenizer, corpus SHA, EOS-prefix, chunk head 128 e bootstrap paired a
20.000 draw con seed `20260916`, stratificato 32/32/32, con quantile
unilaterale superiore 95% a interpolazione lineare.

L'output è write-once: una riga per documento, nell'ordine congelato, con
ID documento per il binding ma senza testo raw, token ID o logits. Riportare
BPB candidato, Δ paired contro il teacher,
upper one-sided CI95, gain paired contro il W4 control e gli stessi valori
per code/prose/technical_general. Non ricalcolare il denominatore, non
selezionare documenti, non estendere il corpus e non usare score parziali.

L'adjudication è ammessa solo dopo 96/96 righe e tutti i controlli:

- `PASS_BPB` se `upper one-sided CI95(Δ teacher) <= +0.02`;
- `FAIL_BPB` se il gate fallisce in modo conclusivo;
- incompletezza, mismatch o limite operativo restano `VOID_APPARATUS`,
  `VOID_RESOURCE` o stato parziale secondo la causa, senza promozione.

Non eseguire task, generazione, rollout, T4, port C, timing o misura di
50 tok/s, indipendentemente dall'esito BPB. Un eventuale `PASS_BPB` autorizza
solo la registrazione di questa qualità heldout, non un deployment o una
promozione di rate.

## Costo teorico e risorse

Il ledger attivo teorico del candidato è
`661782528 + 14614528 = 676397056 B/token`. A 50 tok/s richiederebbe
**33.82 GB/s** prima di overhead, compute, gather, cache, KV, glue e
sampling. È aritmetica di traffico, non una misura di rate né una previsione
di inferenza.

Worker CPU unico, supervisione ogni 5 secondi, senza download o T4. Preflight
con almeno 55 GiB di RAM disponibile e 1 GiB di spazio output; fermare sotto
8 GiB di RAM disponibile, oltre 70 GiB di private commit o working set del
child, oppure oltre 6 ore di wall time. Preservare ogni output parziale e
classificarlo `VOID_RESOURCE` per limite risorse o `VOID_APPARATUS` per
errore di protocollo, dati, hash o parità. Nessun resume o overwrite
implicito.
