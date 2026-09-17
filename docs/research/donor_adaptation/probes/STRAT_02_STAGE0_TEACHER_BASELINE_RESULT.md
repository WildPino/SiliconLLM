# STRAT-02 Stage 0 — baseline teacher F32 heldout

**Misura:** 17 settembre 2026. **Stato:** `COMPLETE` per il solo braccio
teacher; nessun verdetto sulla qualità di W4/W2, nessuna misura di tok/s,
nessuna esecuzione su `engine.c`.

## Identità e protocollo

- Checkpoint pretrained `allenai/StdMoE_1b14b_1T_Preanneal`, revisione
  `d2a4949c9d4ad6cf47fbac131f7e020077332b21`, F32. La snapshot
  contiene 11 shard (54.275.336.216 byte); l'input è stato verificato dal
  worker contro gli hash pin-nati. Il modello è Transformer MoE, non l'SSM
  nativo: questa è una baseline del donor, non il risultato finale.
- Corpus `strat02_document_holdout_v1`: 96 documenti nell'ordine JSONL
  originale, 32 per categoria, nessun troncamento, 181.385 token payload e
  746.161 byte UTF-8. Manifest SHA-256
  `56f3d707040785b21e657d7ba721814fb32fe63e85feb418fafe38889d8ca749`;
  heldout JSONL SHA-256
  `450da27e25755bb7c71215e148f5863d197af6893385033e6100bb212deafd0e`;
  token-ID digest
  `5a4cf31ec7db0f3bc541f2e75a78b133a50cd3d6174abb26ef81e08c45e65289`.
- Tokenizer GPT2TokenizerFast di Transformers 4.57.1 / tokenizers 0.22.2;
  EOS 100257 come prefisso per documento. Score causale di tutti i token
  payload, chunk di 128, `BPB = sum(bits_d)/sum(bytes_d)` come nel
  [protocollo preregistrato](STRAT_02_STAGE0_TEACHER_BASELINE_PROTOCOL.md)
  e nel [protocollo di scoring](STRAT_02_STAGE0_TOKEN_SCORING.md).
- Artefatti write-once in
  [`strat02_teacher_baseline_20260917_105941`](../../../../benchmarks/donor_adaptation/density/results/strat02_teacher_baseline_20260917_105941/).
  Il manifest registra l'interprete Python 3.12.10 e l'ambiente
  Torch 2.6.0+cu124, safetensors 0.8.0, `USE_HUB_KERNELS=NO`.
  **Nessuna T4 è stata usata.**

## Risultato

| Dominio | Documenti | Token | Byte UTF-8 | Bit NLL | BPB |
|---|---:|---:|---:|---:|---:|
| Code | 32 | 57.004 | 245.843 | 94.458,71153387963 | 0,38422371812042494 |
| Prose | 32 | 63.820 | 262.143 | 223.765,41593680484 | 0,8536005765433555 |
| Technical/general | 32 | 60.561 | 238.175 | 132.760,4424202373 | 0,5574071267775261 |
| **Totale** | **96** | **181.385** | **746.161** | **450.984,56989092194** | **0,604406515337738** |

Il supervisor riporta `ok=true`, `status=COMPLETE`, worker exit code 0,
nessuna classificazione di errore, 96 righe persistite. Tempo wall
2.661,375 s (circa 44 min 21 s); questo **non è** un benchmark di decode
e i tempi per documento non sono tok/s. Su 476 campioni del PID worker:
working set massimo 54.757.421.056 B, private commit massimo
56.924.246.016 B, OS peak pagefile massimo 59.843.682.304 B,
RAM fisica disponibile minima 16.009.150.464 B. Nessuna soglia di
supervisione è stata superata.

## Controllo indipendente degli artefatti

Un secondo processo read-only, **senza ricaricare o eseguire i pesi**, ha
rieseguito l'audit di corpus/tokenizer, letto tutte le righe persistite,
verificato ID/categoria/numero di token/byte contro i 96 documenti,
ricostruito somme e BPB globali e per categoria, e confrontato l'hash
dello score con il supervisor. Esito `PASS`, token-ID digest uguale al pin,
96/96 righe. Questo controlla integrità e aggregazione degli output; non
costituisce una seconda misura indipendente dei logits del teacher.

SHA-256 dei file grezzi:

| File | SHA-256 |
|---|---|
| `teacher_scores.jsonl` | `96d284da8f223d613039cb9e2a87b24e7b6130701c0873eea0854c3e8290b224` |
| `worker_result.json` | `658fa659fb61b0d910be4334a7f7894f70cbc17ace246cca1af47b29af9b6257` |
| `supervisor_result.json` | `ffbe6d893db48c0ac10a783d1b53060ce6f77e0d86e7ed2ef302f62a246de01f` |
| `supervisor_manifest.json` | `97144c9499e8a98753c85d4161c23f7592960c5c3600dd6be91a6d70bb82b4e7` |
| `supervisor_log.jsonl` | `5d7a3584c7ef9d6b112bf8170c7f329ae932de53d09d6497f5689657fe8b5788` |

## Decisione e prossimo gate

Questo è il **denominatore teacher** del confronto paired STRAT-02. Il
controllo preregistrato di ripetibilità di due forward sul medesimo testo
rimane ancora da eseguire prima di adjudicare Stage 1: i 96 documenti
distinti di questo run non lo sostituiscono. Il prossimo dato utile è il
BPB dello *stesso* checkpoint con il quantizer W4
congelato nel [formato weight-only](STRAT_02_STAGE0_WEIGHT_FORMAT.md), sulla
stessa heldout. Solo il limite superiore CI95% del delta candidato-teacher
`<= +0,02 BPB` può passare il gate qualità. Prima servono conversione
streaming, parità del codec/operatore e un cap operativo: il riferimento
Python row-wise è troppo lento per convertire ciecamente tutti i 13,36B
pesi lineari. W2 e i test di velocità vengono dopo il gate W4 pertinente.
Neppure un eventuale PASS W4 nel runtime donor dimostrerebbe la conversione
all'SSM nativo o ≥50 tok/s; per il goal servono qualità e rate sul medesimo
artefatto C e il bridge architetturale richiesto dalla roadmap.
