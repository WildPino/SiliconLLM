# STRAT-02 Stage 0 — tokenizer e unità BPB congelati

**Data:** 2026-09-16. **Stato:** preflight tokenizer/corpus, non qualità.
Questo addendum precede qualsiasi download di shard peso, forward del donor o
lettura di loss. Non è un PASS di STRAT-02 né autorizza da solo Stage 1.

## Identità osservata

Checkpoint `allenai/StdMoE_1b14b_1T_Preanneal`, revisione
`d2a4949c9d4ad6cf47fbac131f7e020077332b21`. Sono stati scaricati **solo**
i cinque asset del tokenizer nella cache Hugging Face locale, non i pesi.
`GPT2TokenizerFast` con **Transformers 4.57.1 / tokenizers 0.22.2** caricato
da quello snapshot riporta vocabolario 100278, `bos_token_id=None`,
`eos_token_id=100257`, contesto dichiarato 4096. Gli hash
SHA-256 degli asset sono:

| Asset | SHA-256 |
|---|---|
| `tokenizer.json` | `73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca` |
| `tokenizer_config.json` | `733b2ec0f743cf206dd20335eacbd4427c85487fd6f5216bb361f92a85314315` |
| `special_tokens_map.json` | `4cb2b52960bababa8ec27153831e405ef7811e3fad1226106e09c28e715cfb21` |
| `vocab.json` | `9e14712c91b37c7aab74b1306baa46ac342d620637a4b44523cdc3aec7d24195` |
| `merges.txt` | `b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502` |

Su tutti i 48 documenti calibrazione e 96 heldout del manifest
`strat02_document_holdout_v1`, `encode(text, add_special_tokens=False)` seguito
da `decode(ids, clean_up_tokenization_spaces=False)` ha ricostruito il testo
esattamente; nessun documento ha prodotto token payload con ID >=100256.
Calibrazione: **88756** token, min 967, max 2869; heldout: **181385** token,
min 799, max 3025. Quindi tutti gli span attuali entrano nel contesto 4096
anche con un token di prefisso; non servono truncation o stride. Questi sono
controlli strutturali, **non** output del modello.

**Controllo negativo che ha cambiato il protocollo:** la `.venv` locale con
Transformers 5.13.1 restituisce una classe chiamata `GPT2Tokenizer` anche
quando si invoca `GPT2TokenizerFast.from_pretrained` sugli *stessi asset*.
Roundtrip e hash dei file passano, ma produce 100859/206091 token per calib/
heldout, non 88756/181385: i token ID differiscono. Pertanto gli hash degli
asset e il roundtrip non bastano a garantire il medesimo esperimento.
Il runner deve rifiutare una versione/classe diversa e verificare gli hash
SHA-256 delle liste ordinate di token ID serializzate con
`json.dumps(ids_by_doc, separators=(',', ':')).encode('ascii')`:
calib `430e44946673ed06cc2afcbfb373611c366dd2c363a3af652ffa42bf5a0869f4`;
heldout `5a4cf31ec7db0f3bc541f2e75a78b133a50cd3d6174abb26ef81e08c45e65289`.
Il manifest JSON ha SHA-256
`56f3d707040785b21e657d7ba721814fb32fe63e85feb418fafe38889d8ca749`.
I numeri della prima verifica con Transformers 5.13.1 non sono il conteggio
canonico del donor e non vanno usati per un gate qualitativo.

Il [validatore read-only](../../../../benchmarks/donor_adaptation/density/strat02_token_audit.py)
controlla hash del manifest e dei cinque asset, versioni/classe, integrità
delle righe, disgiunzione, roundtrip, token speciali, hash degli ID ordinati e
contesto. `--selftest` ha passato le mutazioni piantate; l'audit locale con
Transformers 4.57.1/tokenizers 0.22.2 ha restituito `ok=true, errors=[]`;
quello con la `.venv` 5.13.1 ha restituito exit code 1. Per riprodurre senza
alterare la `.venv`, installare le due versioni in un target isolato esterno
alla repo e anteporlo a `PYTHONPATH`, poi eseguire il validatore. La
directory temporanea usata qui è un ambiente locale, non un artefatto
distribuito; il runner futuro deve registrare path/versioni e rifiutare
qualsiasi fallback implicito.

## Protocollo causale da usare in Stage 1

Per ogni documento `d` si usa l'esatto campo UTF-8 `text` del JSONL. Il
tokenizer deve riprodurre gli hash sopra e verificare ancora il roundtrip.
Si costruiscono `p = encode(text, add_special_tokens=False)` e
`x = [100257] + p`, dove `100257` è il prefisso EOS/document-boundary, non un
byte contato nel testo. `1 <= len(p) <= 4095` è obbligatorio: nessun documento
può essere eliminato, accorciato, spezzato o sostituito se il vincolo fallisce;
in quel caso l'apparato va dichiarato `VOID` e riprogettato prima di misure.
Il forward causale usa `x[:-1]` per predire nell'ordine **tutti** i token `p`.
Con `nll_d = -sum_j log P(p_j | x[:j+1])` in nat, la misura per documento è
`bits_d = nll_d / ln(2)` e `bytes_d = len(text.encode('utf-8'))`. EOS di
chiusura non viene predetto né contato. Così il primo token del testo non è
gratis nonostante l'assenza di BOS; la probabilità è condizionata da un
confine di documento esplicito in *entrambi* i bracci.

Il BPB aggregato è `sum_d bits_d / sum_d bytes_d`, **non** la media semplice
dei BPB per documento. Teacher e candidato usano le stesse `p`, `bytes_d`,
ordine e impostazioni di forward; i bit per documento sono persistiti prima
dell'aggregazione. Il delta paired è
`sum_d (bits_candidate,d - bits_teacher,d) / sum_d bytes_d`.

Per il limite superiore di confidenza si fanno 20000 draw con seed intero
`20260916`, campionando con rimpiazzo **32 documenti per categoria** fra
`code`, `technical_general`, `prose` nell'heldout congelato. A ogni draw si
ricalcola il rapporto delle somme paired; l'upper CI95 unilaterale è il
quantile empirico 0.95 dei 20000 delta (interpolazione lineare definita dal
runner). Il gate primario rimane quello del brief: upper CI95 <= +0.02 BPB.
Si riportano separatamente punto e intervallo per ciascuna categoria e il
numero di documenti/byte/token; non si fa bootstrap di token correlati.
L'eventuale impossibilità di calcolare 20000 draw o una CI troppo ampia è
inconclusiva, mai un PASS ottenuto scegliendo un subset.

Questo documento fissa solo scoring e resampling. **Ancora da congelare prima
dei pesi:** suite generativa/task e relative soglie, quantizer W4/W2 e layout,
versione/runtime finale, piano di loading bounded-memory e provenienza dei
testi. Nessun risultato di E63, E66 o del corpus legacy entra in questo gate.
