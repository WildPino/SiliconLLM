# STRAT-02 Stage 0 — acquisizione pesi

**16 settembre 2026. Esito: acquisizione completa, non misura di qualità o velocità.**

Donor `allenai/StdMoE_1b14b_1T_Preanneal`, revisione immutabile
`d2a4949c9d4ad6cf47fbac131f7e020077332b21`. Il runner
[`strat02_acquire_weights.py`](../../../../benchmarks/donor_adaptation/density/strat02_acquire_weights.py)
del commit `3b5709b` ha passato il preflight offline con Transformers 4.57.1,
tokenizers 0.22.2 e `USE_HUB_KERNELS=NO`; RAM disponibile 62.909.988.864
byte, cache libera 7.283.021.737.984 byte. Il processo PID 20436 è stato
avviato alle 21:28:21 locali e ha concluso tutti gli 11 shard; il suo
[`stdout`](../../../../benchmarks/donor_adaptation/density/results/strat02_acquire_20260916_212821.out.log)
registra per ciascuno il download e la verifica di dimensione e SHA-256;
`stderr` è vuoto.

Payload finale verificato: **54.275.336.216 byte**. L'allowlist e i digest
attesi dei singoli shard sono in
[`strat02_weight_sources.json`](../../../../benchmarks/donor_adaptation/density/strat02_weight_sources.json),
SHA-256 `7aab6424b2390d5b1956793df58aa6be56a279e2d7ec01568d79a29a525a179e`.
L'audit write-once nella snapshot cache HF `E:\AI_Cache\huggingface\hub\models--allenai--StdMoE_1b14b_1T_Preanneal\snapshots\d2a4949c9d4ad6cf47fbac131f7e020077332b21\strat02_acquire_weights.audit.json`
ha SHA-256 `59d67353fa01be34f453ecaf9dd7ae9ca8c85df0b080cef6b5f96642b1774819`
ed è stato scritto alle 21:53:43 locali. Tutti gli shard risultano
`downloaded`, non preesistenti. I **byte di rete** restano `unknown`: il
cap dichiarato riguarda solo i payload finali validi, non retry o overhead.

Nessun peso è stato ancora caricato in un modello, nessun forward del donor
pretrained è stato eseguito e nessuna T4 è stata usata. Il prossimo gate è
il primo documento di calibrazione con supervisione RAM/tempo, **non** un
verdetto di BPB. Non ripetere il download per ottenere una seconda misura.
