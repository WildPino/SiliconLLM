# STRAT-02D — attribuzione controfattuale del danno W4 per organo

**Preregistrato il 17 settembre 2026, prima di qualunque score di questa
cella.** Diagnostico CPU su sola calibrazione; nessuna T4, nessun heldout,
task, rollout o timing di inferenza. Il gate W4-v2 precedente è concluso
`FAIL_BPB` e non viene riaperto. Il suo primo tentativo `VOID_RESOURCE` resta
separato. [Risultato](../probes/STRAT_02_W4_BF16_QUALITY_RESULT.md).

## Domanda e controllo

Quanto della perdita del modello W4-v2 è recuperabile *sullo stesso modello*
rimettendo F32 in un organo alla volta? Si carica una volta l'esatto export
W4-v2 già verificato, si riusano come controllo write-once le sue 48 righe
`calibration_checks.jsonl` SHA-256
`ef1d7f42ad30ff5727432bcc4f0c1a5299ff681f52a5a7690f55602f0171a142`,
e si cambia solo il contenuto delle matrici nominate nell'arm. La revisione
sorgente F32 resta `allenai/StdMoE_1b14b_1T_Preanneal`@
`d2a4949c9d4ad6cf47fbac131f7e020077332b21`; verificare tutti gli 11
shard contro il manifest pin-nato prima di copiare valori. Il candidato è
l'export in
`results/strat02_w4_bf16_full_export_20260917_143753`, vincolato
dagli SHA-256 già pubblicati: manifest
`06924c13b5ad2106929ee51c84442bf63b4b47c6528e90b1ea91de891d4a0a5c`,
plan `1c1512b3735f0f9a80b638908801dfba7894b781047c767078a8b0cba54c3e42`,
export supervisor
`d52e09748d594b26af89b8223dfcd372343e7b344bc9f10199d379e5e4c7faca`.

## Bracci congelati e ordine

Score di tutti i 48 documenti `calib.jsonl` nell'ordine originale, una
volta **per braccio**, stesso tokenizer, EOS-prefisso, nessun troncamento,
chunk head 128 e F32/eager come nel gate precedente; registrare i thread
effettivi senza inventare una configurazione non congelata. I quattro
bracci, in quest'ordine, sono:

1. `ROUTER_F32`: soltanto le 16 matrici `organ=router`.
2. `HEAD_F32`: soltanto la matrice `organ=lm_head`.
3. `ATTENTION_F32`: soltanto le 64 matrici `organ=attention`.
4. `NONEXPERT_F32`: le 81 matrici dei tre organi sopra insieme.

Gli altri 6.144 record expert W4 e i 34 record F32 passthrough restano
identici. Non si sceglie un layer, documento o sottoinsieme di expert dopo
aver letto uno score. Prima del primo arm, un documento sentinella in W4
deve riprodurre il punteggio già pubblicato entro `1e-5` bit. Dopo ciascun
arm, ripristinare i record W4 in-place dai byte originali, verificarne
l'identità bit-per-bit mediante SHA-256 dei tensori coinvolti e riscorare
la stessa sentinella entro `1e-5` bit. Qualsiasi mismatch è
`VOID_APPARATUS`, non una stima di beneficio. Nessun secondo modello F32
simultaneo in RAM; sorgente safetensors mmap e copia di un tensore per volta
nelle viste già allocate. Nessun cambio di operatore, routing, ordine o
tokenizer. Gli score parziali non orientano il resto del run.

## Misura e interpretazione

Per ogni arm, registrare score per documento senza testo/token/logit,
BPB aggregato e separato per code/prose/technical_general, e
`gain_cal_BPB = (bits_W4 - bits_arm) / bytes_cal`. Confrontare ogni arm
con le medesime 48 righe W4, non con il teacher heldout. Registrare anche
il costo teorico addizionale *attivo* se quel ripristino F32 fosse
effettivamente deployato: router `14.614.528`, head `716.111.872`,
attention `935.329.792`, nonexpert `1.666.056.192` B/token oltre il
ledger W4-all di `661.782.528` B/token. I valori derivano da
`(4 - 66/128) × numero_pesi_attivi` e vanno ricontrollati contro il plan.

Il ripristino F32 è un **oracolo di attribuzione**, non un formato candidato
o un'approvazione di più byte. Un gain locale non predice da solo la
qualità heldout o la velocità; i gain dei bracci non si sommano perché
nonlinearità, routing e logits interagiscono. `NONEXPERT_F32` serve proprio
a osservare questa interazione. Se il gain è nullo o negativo, non prova
che quell'organo sia irrilevante in una quantizzazione/training diverso.
Non leggere o riusare gli score heldout nel processo. Dopo questa cella
si decide una sola ipotesi successiva con brief e dati disgiunti; niente
passaggio automatico a W2 o port C.

## Audit, risorse e stop

Output nuovo/write-once, JSONL per arm e JSON finale con identità, hash
di codice/dati/sorgente/export, contatori, score aggregati e risorse.
Worker CPU unico, supervisione ogni 5 secondi; preflight RAM fisica
disponibile almeno 55 GiB e spazio output almeno 1 GiB; stop RAM fisica
disponibile sotto 8 GiB, private commit o working set worker oltre 70 GiB,
wall oltre 6 ore. `VOID_RESOURCE`/`VOID_APPARATUS` mantengono output
parziale senza essere promossi. Verificare unit test sintetici (selezione
esatta degli organi, copia, rollback, hash e score sentinella negativo)
prima del donor. Non usare il run per rivendicare BPB heldout, qualità di
generazione, `engine.c` o token/s.
