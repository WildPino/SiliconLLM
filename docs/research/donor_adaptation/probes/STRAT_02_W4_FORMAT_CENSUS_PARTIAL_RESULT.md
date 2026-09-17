# STRAT-02 W4 v1 — censimento interrotto, controesempio decisivo

**17 settembre 2026. Raw `INCOMPLETE/WORKER_ERROR` per cap output; decisione
condizionale del [brief](../briefs/BRIEF_STRAT_02_W4_FORMAT_CENSUS.md):
la sola canonicalizzazione della riga shared del router è insufficiente.**
Non è un censimento completo del checkpoint, una misura BPB, un benchmark
CPU, un fallimento di W4 con scale diverse o un risultato `engine.c`.

Il runner del commit `f58bbb0` ha verificato gli 11 shard pin-nati una
volta e visitato i lineari in ordine lessicografico, con slice massime di
256 righe. Gli [artefatti write-once](../../../../benchmarks/donor_adaptation/density/results/strat02_w4_format_census_20260917_124350/)
riportano exit code 1 dopo 156,797 s e 32 campioni di supervisione.
Sono stati completati **731/6.225 tensori**, **13.617.152/104.398.848
gruppi g128**; il tensore seguente è solo parziale. Fra i gruppi
completi: 4.455.268 underflow della scala F16 a `c=0,50`, pari al
32,718% di **quel prefisso lessicografico**, non a una stima dell'intero
modello; zero source-nonfinite, scale-nonfinite e gruppi realmente zero.
Il controllo positivo della riga 127 del router layer 0 ha trovato i 16
underflow attesi. Il file delle coordinate invalide ha raggiunto il cap
preregistrato di 512 MiB e il worker ha rifiutato di troncarlo fingendo
un risultato completo. Non aumentare il cap soltanto per ottenere una
lista enorme già smentita da un controesempio.

Il **primo invalid fuori dal router** è
`model.layers.0.mlp.experts.0.down_proj.weight`, riga 0, gruppo 3.
Una lettura diagnostica diretta dei 128 F32 pin-nati trova valori finiti
e non tutti zero, `absmax=1,0720082861768315e-15`; le scale F16 sia a
`c=0,50` sia a `c=1,00` sono zero. Il codec W4 **scalare** congelato
solleva anch'esso `ValueError`. Già questo gruppo basta a falsificare
l'ipotesi «solo le 16 righe shared del router sono invalide»; non dipende
dalla vettorizzazione del censimento. Molti expert gate/up dello stesso
prefisso hanno migliaia di gruppi invalidi; ciò non prova ancora quanti
expert/layer del modello totale siano coinvolti né la loro importanza
funzionale.

La macchina non ha toccato i cap RAM: massimo private commit worker
7.469.867.008 B contro 8 GiB consentiti; RAM disponibile minima
66.726.543.360 B contro 4 GiB. Il private commit è cresciuto molto
durante la scansione: un eventuale censimento completo richiederà anche
un metodo di lettura più parsimonioso, non solo un file output più grande.
Nessuna T4, heldout, inferenza o conversione è stata usata.

SHA-256 degli artefatti di controllo e parziali:

| File | SHA-256 |
|---|---|
| `census_manifest.json` | `39c503f00b7fef4112c85d2cf15b5ebd907acaa4ac40e1df2c710ef89067695a` |
| `tensor_counts.jsonl` | `080115d2493d9daad9f75eb6c622c97f8f6c219cd69b4fa418f2aa2343885432` |
| `invalid_groups.jsonl` (parziale, 512 MiB) | `cd4afe77be89dbfa087e8867ff060fc2245aacde9a29e7e31a4ce8242331dbd0` |
| `worker_result.json` | `0ea8213ae954e04708460fd6ec9387b5e22c54e1d537135ef28864387a1f24cc` |
| `supervisor_result.json` | `0e8302d023f27fcd9b9fe7b87e5001c3a4a0e54d920bad36d162e157fa1f79f0` |
| `supervisor_log.jsonl` | `a4584a21c93d0b0723b8367a65e577448ec343775073af9395691c97d2479fff` |

**Prossima decisione:** non promuovere il v1 e non applicare soltanto lo
zero alla riga shared. Serve un nuovo formato/precision map per scale
di pesi non nulli molto piccoli. Una scala di due byte con esponente più
ampio, per esempio BF16, è una *ipotesi* interessante perché il payload
dei codici resterebbe W4; vanno congelati rounding/layout/decoder,
verificata la validità su un campione reale e poi la fedeltà paired. Il
risultato qui non autorizza ancora un export o un heldout.
