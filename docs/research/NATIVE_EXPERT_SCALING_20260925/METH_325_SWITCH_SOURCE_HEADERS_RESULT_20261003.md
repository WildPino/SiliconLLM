# METH-325: actual original source metadata passes

**PASS, actual metadata only.** Freeze a8c1b6e, observations e11dcbb.
[Raw result](meth325_switch_source_headers_result.json) SHA256
f49bd77012045ce2ba2d1ad20d29f86833d229674a644ee0d250e622c39d71e8.
[Protocol](METH_325_SWITCH_SOURCE_HEADERS_PROTOCOL_20261003.md), driver
benchmarks/native_expert_scaling/meth325_switch_source_headers.py.

Both exact321 revisions/API commits/LFS SHA/size bound; ALL original6+3 shard
ZIP central directories and data.pkl decoded with restricted structural reader,
no Torch/checkpoint code execution. ALL actual tensor names/shapes match321 and
original base source dtype F32; every storage-record length matches metadata.
All serialized tensor bytes match original index. Actual observed counts:

| Source | Shards | Tensor names | Serialized tensor bytes incl aliases | Physical ZIP storage bytes | Whole archive bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
|base256|6|6392|58952709120|58854011904|58856206797|
|base128|3|3320|29956961280|29858264064|29859401788|

META architecture-unique counts remain14,664,154,368/7,415,217,408 parameters,
not actual function/hash proof. Source physical storages exceed inferred unique
architecture bytes; tied aliases can be physically serialized separately.
Full acquired tensor equality/hash checks are required before deduplication.
No whole source LFS hash checked by this range-only experiment.

28.328s,38HTTPcalls,**3,448,548B** total response bodies, endRSS58,900,480B.
36 verified64KiB-aligned ZIP windows; each bound to exact Content-Range,
source commit and expected LFS SHA in original redirect headers. Original API
JSONs, all9 data.pkl files and range/header hashes retained locally. Windows
can include incidental adjacent weight bytes; no complete learned weight tensor
or expert function reconstructed. No selected first-shard-only inference.

Decision: permit bounded primary full acquisition with per-shard expectedLFS
SHA verification. Full source payload/tensor/function diversity, source baseline,
native conversion/precision, actual routing/LUT/DRAM cost and same-artifact
quality/>=50accepted rate still missing. Base128 comparison30GB remains available
for later actual bank-count scaling; metadata does not prove useful larger-n.
