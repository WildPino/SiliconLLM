# Numbered pre-function repairs to SRC007 input admission

6 October2026. Preserve both failed preparation attempts; no model function,
fit, selected split or quality result has been observed. All scientific source,
selection, token/mask/split and resource limits remain fixed.

Inputs001 at3d3e84c5 rejects an inherited acquisition report's CRLF working
serialization against its LF Git blob before tokenizer copies or corpus access.
Identity diagnostics prove every own file exact, and only143 CR bytes differ
for the inherited JSON: working7,058 bytes SHA89add8d24541423dfa61827f785bf125c2062cf1aa6976214f03da94b7c582b7;
Git6,915 bytes SHA f0e16a24e14629b312e9893eb0b353358a9ad9935f1514fd4e662e1429875218.
Repair preserves raw inherited metadata, binds canonical Git identity and
requires only CRLF normalization. Do not edit inherited/main files.

Inputs002 at98bcbce7 downloads the complete40,435,873-byte corpus once and
verifies its pinned SHA, then stops on an oversized JSON line. This is not
an admitted scientific split. Preserve full compressed source and raw worker/
controller evidence. Complete child9.092168s, nested8.344s/RSS336,973,824B;
download2.515s. Inputs001 complete4.108917s/nested3.5s/RSS205,529,088B.

Before inputs003, revise oversized-line handling: discard the oversized row
without JSON decoding or tokenization, consume its remaining bytes in bounded
1 MiB readline blocks through newline, and charge every byte to the unchanged
512 MiB observed decompression cap. Record oversized-line discards. The row
memory cap remains2 MiB plus the one-byte oversize detection marker; no large
row buffer is admitted. The first4,096 eligible candidates and all downstream
rank/split/mask rules remain unchanged. This changes source admission handling
before model functions, not fidelity/quality criteria after results.

Reuse only the exact unchanged compressed source from inputs002. Verify full
length/SHA anew; record its owned path, identity and historical download cost.
No second HTTP corpus download or repeated original function is authorized.
Keep inputs001/002/controller namespaces exclusive. Freeze changed worker and
this repair note before003, perform fresh live CPU admission, then independent
source/document/reconstruction/retention/Git audit before original execution.
