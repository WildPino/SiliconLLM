# Interaction binding repair1: physical lengths, before FIRST tokenizer execution

Original sources/goldens/protocol/binding frozen7fb1a38c97aba587cb02ff30491b668a5ee58a4e.
Actual first launcher174284 exits1 before worker launch/fixture observation.
[Failure](chatbot_qwen_interaction_20261007.launcher_failure.json): actor20812/
create1791378323.579076, .141s/pre-failure OS snapshot27996160B; worker peak0
because no worker exists. Do not treat that snapshot as a through-exit peak.

PowerShell Get-Item reports Length0 for six donor snapshot symbolic links;
Get-FileHash follows the target and correctly matches original source bytes.
The Python launcher's physical length comparison correctly rejects config.json
before any tokenization. This is administrative descriptor metadata failure,
not changed source content or a scientific/quality outcome. The initial source,
binding and actual fault remain retained; no first fixture/native/model replay.

Fix the administrative Entry function to read file-stream Length (following
links). Extend the original binding in a NEW namespace: verify all old content
SHA unchanged except the explicitly corrected setup source, update physical
lengths, append repair source/protocol/original binding/failure inputs. Reuse
the existing sealed package view/tree hashes; never rebuild it or run original
setup again. All worker/launcher/expected fixtures/BPE/criteria/resources remain
exactly the original frozen body. Freeze before the FIRST tokenizer execution.

Earlier metadata-only shell7771be mistakenly used case-insensitive
ConvertFrom-Json on BPE vocabulary; it failed on A/a and printed null fields.
Those fields are INVALID. Shella6badb uses -AsHashtable and establishes actual
NFC/ByteLevel metadata before goldens were frozen; no token IDs were queried.
No dataset, threshold, expected text or normalization rule is changed now.
