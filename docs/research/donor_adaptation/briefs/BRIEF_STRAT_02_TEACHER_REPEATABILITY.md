# STRAT-02 — ripetibilità del teacher F32 prima del gate W4

**Preregistrato il 17 settembre 2026 prima della seconda misura teacher.
Stato: PROPOSED.** La [baseline heldout](../probes/STRAT_02_STAGE0_TEACHER_BASELINE_RESULT.md)
è completa, ma il [brief di precisione](BRIEF_STRAT_02_STDMOE_PRECISION_GATE.md)
richiede ancora due forward deterministici dello stesso testo. Questa prova
usa soltanto una riga di **calibrazione**, non riapre il denominatore heldout
e non fornisce qualità del W4 o tok/s.

## Input e misura congelati

Checkpoint/revisione, codice remoto, tokenizer, ambiente e 11 SHA-256 sono
quelli della baseline. Selezionare **la prima riga nell'ordine originale**
di `corpus/strat02_document_holdout_v1/calib.jsonl`, senza scegliere
in base alla score. ID:
`file:data/external/the_stack_python/cpython/Lib/test/test_sqlite3/test_userfunctions.py`,
categoria `code`, 8.192 byte UTF-8, SHA-256 del solo testo
`dd1ca41399ccc61f42f1a8984c35c704d6c7449025a1f1e21337d70e42a63b4a`.
Verificare manifest corpus, split/ID/hash e tokenizer prima di usare i pesi.
Nessun testo grezzo, token ID o logit è persistito.

In un **solo worker fresco**: hash degli 11 shard una volta, istanza
mmap del medesimo `EmoForCausalLM` F32 su `meta` con assegnazione/pointer
sharing come la baseline; `eval`, CPU, `torch.inference_mode()`, autocast
disabilitato, `use_cache=False`. Tokenizzare l'intero documento con la
regola congelata EOS-prefisso/no-truncation. Eseguire due score completi
consecutivi tramite `strat02_score.score_document`, `chunk_size=128`,
**sullo stesso oggetto modello**, senza cambiare seed, thread o stato;
registrare bit NLL, BPB, token e byte dei due pass. Il controllo numerico
è `abs(BPB_2-BPB_1) <= 1e-7`; token e byte devono coincidere esattamente.

Inoltre, per i **primi 64 token payload** dello stesso documento, eseguire
due forward consecutivi del backbone e della head sulla sequenza
`[EOS] + payload[:63]`, senza cache, e confrontare tutti i logits F32
del blocco `1×64×100352` con `atol=1e-5, rtol=1e-6` elemento per
elemento. Registrare massimo errore assoluto e numero di elementi fuori
tolleranza, non il tensor. Verificare tutti i valori finiti. Questo
controllo copre un vettore di output, mentre il BPB controlla l'intero
documento: nessuno dei due certifica altri domini o altri contesti.

## Apparato e stop

Selftest sintetico di selezione riga, hash, formula di tolleranza,
nonfinite e mismatch piantato. Il parent crea directory nuova/write-once,
registra hash del brief/runner/sorgenti e PID, lancia solo il proprio worker
e lo campiona ogni 5 s. Preflight RAM fisica disponibile >=55 GiB e spazio
output >=1 GiB; stop a RAM disponibile <8 GiB, private commit o working
set del worker >70 GiB, wall >45 min. I parziali restano `INCOMPLETE`.
Mismatch di hash, corpus, keyset, pointer, forma, nonfinite o protocolli è
`VOID_APPARATUS`; superamento risorse è `VOID_RESOURCE`; una differenza
numerica oltre le tolleranze, con apparato altrimenti valido, è
`FAIL_REPEATABILITY`. Nessun rerun con testo o tolleranza cambiati.

Un `PASS_REPEATABILITY` autorizza soltanto l'uso della baseline F32 come
oracle stabile **per questo controllo** nel gate W4-v2. Non riporta BPB
heldout nuovo, non attribuisce l'errore del W4, non usa T4 e non dimostra
compatibilità con `benchmarks/phase60/engine.c` o >=50 tok/s.
