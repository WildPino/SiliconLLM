# METH-396: prior metadata checkout line-ending binding failure

Frozen634733f. Model-free physical committed-input preflight failed on existing
donor-adaptation `configs/_manifest.json`; actual controller command exit1
fully consumed and retained the same first failure.0.140s, zero response bytes,
empty request/shard lists, no metadata directory, network/source tensor values
or inference. No source cost/header/capacity observation.

The existing physical manifest has10,212B/CRLF, SHA
bca52715780adc1d98a995142ad63491133f68df70daede1492f2615aa4172ab;
HEAD cat-file --filters has9,948B/LF, SHA
fcb168f0d2004baf6c0f2938f997e095c9210e72b004c0a561add8608500e35d.
Normalizing BOTH toLF gives exact equal bytes. Git semantic content is unchanged;
the strict physical binding correctly refused the mismatched checkout.

Raw `meth396_qwen_next_source_headers_result.failure.json`, SHA256
`f96467c48761c338e19e9ab4a20e85fc5952de18b9cf8df73c1bd02935681132`.
Frozen396 controller/protocol remain unchanged; no optional same-ID restart.
After retaining first failure, restore ONLY this clean existing manifest's
physical bytes from HEAD filtered blob, preserving exact tracked content.
NEW397 controller/protocol/output names retain failure binding; all source,
header, analysis, budgets and decision gates remain396 unchanged. Physical
committed-input preflight must then pass before network. No goal conclusion.
