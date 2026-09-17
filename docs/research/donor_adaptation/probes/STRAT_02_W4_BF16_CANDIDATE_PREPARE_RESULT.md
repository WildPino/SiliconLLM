# STRAT-02 W4-BF16-v2 — candidate prepare gate

**17 settembre 2026. Stato: `PREPARED_METADATA_AND_FORMAT`, non quality pass.**

Il loader del candidato è in `benchmarks/donor_adaptation/density/strat02_w4_bf16_candidate_loader.py` (commit `1a34066`, senza firma). I nove test sintetici del decoder, passthrough F32, binding dei tensori `meta` e controlli di corruzione sono passati sull'interprete Python 3.12 pin-nato. Sono test dell'apparato, non misure di fedeltà del modello.

È stato poi eseguito `prepare_candidate` sull'unico [artefatto vincolato](../briefs/BRIEF_STRAT_02_W4_BF16_QUALITY_ARTIFACT_BINDING.md), con `USE_HUB_KERNELS=NO`, modalità Hugging Face/Transformers offline e Transformers 4.57.1. La preparazione ha verificato i controlli, la topologia completa istanziata su `meta`, il keyset del checkpoint e il formato/hash di tutti i payload esportati. Output:

```text
PREPARED 11 {'records': 6259, 'linears': 6225, 'groups': 104398848, 'linear_params': 13363052544, 'other_params': 205588480, 'linear_bytes': 6890323968, 'f32_bytes': 822353920, 'payload_bytes': 7712677888} 06924c13b5ad2106929ee51c84442bf63b4b47c6528e90b1ea91de891d4a0a5c
```

Il digest finale è lo SHA-256 di `artifact_manifest.json` già preregistrato. `prepare_candidate` non ha decodificato l'intero modello F32, non ha letto i valori dei pesi F32 sorgente, non ha eseguito forward, calibrazione, heldout, task, rollout o benchmark. Il prossimo controllo deve caricare **questo** artefatto una volta in un worker supervisionato e misurare la qualità secondo il [brief paired](../briefs/BRIEF_STRAT_02_W4_BF16_QUALITY_SCORE.md). Nessuna conclusione su `engine.c` o tok/s deriva da questa preparazione.
