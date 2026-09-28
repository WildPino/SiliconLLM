# METH-167: exact expanded source-child trace

The pinned BF16 E1,280 source model selected four children at each of 442,766 input positions in the 1,280 new METH-165 raw/chat pairs. The trace records all 24 layers, 1,771,064 selections per layer, token IDs, previous IDs, local positions and 2,560 sequence offsets. The five raw files total 89,580,378 bytes. Every old-route all/content histogram for raw, chat and combined cells reproduces METH-165 exactly in every layer, so the trace is valid for offline attribution and replay.

The result is `meth167_expanded_source_route_trace.result.json`, SHA-256 `9ed2e2a008fbde79c6ba2032b830ab9d66c7400e19f0fda8e918e8361d1ff0ff`. The recoverable five-file ZIP is `meth167_expanded_source_route_trace.zip`, 49,088,442 bytes, SHA-256 `15c7dba9ef5d5e58b4a693a8a69580949e7c328661055a784b6e40a6170888ae`. Each archived entry was decompressed and checked against the byte size and SHA-256 in the result. Source bank, child checkpoint, manifest and teacher hashes are in that result.

Local RTX 3060 capture took 508.906 s, peak allocated GPU memory 2.945 GB and process RSS 4.373 GB, within the frozen limits. This is an exact route trace, not evidence of learned E12,800 quality or accepted-token speed.
