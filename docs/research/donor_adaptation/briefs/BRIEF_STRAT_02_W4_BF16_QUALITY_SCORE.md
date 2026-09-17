# STRAT-02 W4-BF16-v2 — scoring paired dello stesso artefatto esportato

**Preregistrato il 17 settembre 2026, prima di eseguire la qualità W4-v2.
Stato: CONDITIONAL, non eseguito.** Procedere solo dopo
`PASS_REPEATABILITY` del [teacher](BRIEF_STRAT_02_TEACHER_REPEATABILITY.md)
e `COMPLETE` con verifier dell'[export completo](BRIEF_STRAT_02_W4_BF16_FULL_EXPORT.md).
La baseline F32 heldout è già write-once: 96 documenti, 746.161 byte,
0,604406515337738 BPB, file `teacher_scores.jsonl` SHA-256
`96d284da8f223d613039cb9e2a87b24e7b6130701c0873eea0854c3e8290b224`.
Non ricalcolare il denominatore scegliendo un'altra baseline.

## Identità e caricamento

Il candidato è esattamente il payload W4-v2 completo che avrà passato
il verifier; nessuna selezione di shard, organo o checkpoint. Il loader
controlla manifest, checksum degli undici contenitori e di ogni record,
keyset/shape/byte ledger. Istanzia la stessa classe StdMoE/revisione
su `meta` e ricostruisce ogni lineare in F32 **dai soli scale/codici W4
esportati**, gli altri 34 tensori dai record F32 passthrough; niente
correzione con shard F32 originali. `load_state_dict(strict=True,
assign=True)`, nessuna chiave mancante/extra, nessun tensor `meta`,
RoPE nonpersistente ricostruita come nel loader teacher. Il sorgente
F32 può restare sul disco per provenienza, ma il processo candidato non
apre i suoi valori. Un solo modello completo in RAM; niente doppia copia
teacher+candidato.

Prima dello score, selftest piantati per decoder, record corrotti,
nonfinite e `nn.Linear` dai soli byte packed. Su calibration, verificare
operatori/forma/routing 7 routed+1 shared con i controlli piantati
preesistenti e score dei 48 documenti di calibrazione nell'ordine
congelato: solo un controllo operativo di finitezza, identità/token/byte,
nessuna scelta di formato/scala/soglia dal loro BPB. Ogni problema di
integrità o parità è `VOID_APPARATUS` e vieta lo heldout.

## Heldout e decisione

Se tutti i controlli passano, score **esattamente una volta** tutte le
96 righe heldout nell'ordine originale, con il
[core BPB preregistrato](../probes/STRAT_02_STAGE0_TOKEN_SCORING.md):
EOS-prefisso, nessun troncamento, chunk head 128, 181.385 payload token,
746.161 byte e il medesimo tokenizer. Un record write-once per documento,
senza testo/logit/token ID. Applicare `strat02_score.adjudicate_pinned_files`
al file W4-v2 completo e al teacher SHA sopra: bootstrap paired per
documento, stratificato 32/32/32, seed `20260916`, 20.000 draw, quantile
unilaterale superiore 95% con interpolazione lineare. Riportare BPB,
ΔBPB, upper-CI95 e risultati separati code/prose/technical-general.
Il gate obbligatorio è **upper-CI95(ΔBPB) <= +0,02**. Se il CI è
impreciso/inconclusivo, non selezionare documenti né estendere il corpus
retroattivamente. Un failure conclusivo è `FAIL_BPB`; non correggere
il formato v2 o passare automaticamente a W2.

Solo se BPB passa, eseguire le suite task e rollout fissate in
[STRAT-02 task/rollout](../probes/STRAT_02_STAGE0_TASK_ROLLOUT_PROTOCOL.md)
sullo **stesso** artefatto, con teacher comparabile e stop `catastrophic`
già fissati; se BPB fallisce, registrarli `NOT_RUN_BPB_FAIL`.
HumanEval rimane `PENDING_SANDBOX` finché l'esecuzione non ha un
isolamento verificato: non scambiare l'assenza di quel task per un pass.
Nessuna velocità o conversione `engine.c` viene inferita dal BPB.

## Risorse, audit e stop

Worker CPU unico monitorato ogni 5 s, output nuovo/write-once, nessun
download/T4. Preflight RAM fisica disponibile >=55 GiB, spazio output
>=1 GiB. Stop a RAM disponibile <8 GiB, private commit o working set
worker >70 GiB, wall >6 h per calib+heldout; parziale `INCOMPLETE`,
risorsa `VOID_RESOURCE`, errore di protocollo/dati/hash/parità
`VOID_APPARATUS`. Registrare manifest SHA, codice, runtime, sorgente,
tokenizer, stdout/stderr, log/risorse, output per documento e stato
finale; nessun resume o overwrite implicito.

Anche `PASS_BPB` più task validi sarebbe solo fedeltà del donor
Transformer ricostruito in Python. Il byte ledger W4-all
661.782.528 B/token e i costi di full attention, routing e head
restano da misurare nel runtime C sul **medesimo artefatto**. Lo SSM
`benchmarks/phase60/engine.c` ha operatori e dimensioni diversi:
senza bridge e gate propri l'obiettivo originale non è raggiunto.
