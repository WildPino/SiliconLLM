# STRAT-02F — attempt 1: `VOID_APPARATUS`

**Eseguito il 18 settembre 2026. Stato: `VOID_APPARATUS`.** Questa è la
prima esecuzione della [preregistrazione congelata](../briefs/BRIEF_STRAT_02F_W4_ROUTER_F32_HELDOUT.md)
per il solo candidato W4-v2 con le 16 matrici router F32 ripristinate.
L'output parziale è conservato in
`benchmarks/donor_adaptation/density/results/strat02f_w4_router_f32_heldout_20260918_124259`.

## Causa

Il supervisor ha riportato `ok=false`, child exit code `1`, durata
`26.532 s`, sei campioni e
`partial_preserved=true`. Il worker è terminato prima di qualunque load di
valori F32/W4 o score heldout con:

```text
AttributeError: 'str' object has no attribute 'get'
```

La causa è in `_read_heldout_rows`, runner line 293: il nuovo runner assume
che `w4_quality._audit_full_corpus` restituisca `report["manifest"]` come
oggetto, mentre il report esistente lo restituisce come stringa di path.
La chiamata a `.get("schema")` fallisce nell'audit del corpus nel worker,
prima di caricare i pesi.

## Portata e adjudication

Il preflight metadata iniziale era passato, ma sono state persistite **0/96**
righe heldout. Non sono stati caricati valori F32 o W4, non sono stati
eseguiti gli hash delle sorgenti, e non esiste alcuna conclusione di qualità.
Non sono stati riportati testo, token ID o logits. Questo tentativo non
modifica né adjudica la preregistrazione e non è un risultato scientifico.

Il fix successivo deve rimuovere l'assunzione sull'oggetto schema e aggiungere
un regression unit test; la verifica offline deve poi passare gli 11 test e
l'audit del corpus di tutte le 96 righe heldout. Qualunque nuovo tentativo
richiede un output root fresco; non si fa resume o overwrite di questo root.
