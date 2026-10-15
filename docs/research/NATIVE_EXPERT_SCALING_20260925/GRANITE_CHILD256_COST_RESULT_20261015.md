# Q4 expert projection-bundle cost result

15 October2026. **Numeric PASS / prospective joint cost FAIL.**
Frozen arithmetic/controls atbc1aa8a; same expert format and gates asd224827.
All values are synthetic. The complete projection/FFN/head bundle omits causal
attention/RoPE/KV/generation. This is not a useful-token throughput result.

Both modes passed every Q4 output row against independently decoded scalar
arithmetic:589,824 rows/all children and294,912/one child, relative L2 about
5.48e-8. All832 sampled I8 rows,24 routers/selected masses,24 selectors and
mixtures passed. All12 timing workers exited0 and all288 sample output hashes
are stable across processes/repetitions/cache profiles. Binary bytes match the
qualified numeric stage. Peak combined RSS2,171,191,296 bytes; timing sequence
31.234s, within all limits. Hashing is outside measured calls.

| Expert mode | Cache profile | Three process medians, ms | Worst process/fixture median, ms |14ms/stability gates |
| --- | --- | --- | ---: | --- |
|all children|resident|33.628, 20.892, 19.467|85.205|FAIL/FAIL|
|all children|evicted|21.628, 21.773, 18.210|37.548|FAIL/FAIL|
|one child|resident|14.365, 15.209, 11.995|18.774|FAIL/FAIL|
|one child|evicted|15.434, 19.609, 20.338|54.796|FAIL/FAIL|

All32 fixed mode/profile/fixture groups fail both the14ms and1.10 max/min
requirements. Preserve the large dispersion and every observation; do not
select the11.995ms process or quote a causal speed ratio from separate runs.
Q4 expert payload halves169,869,312->84,934,656 addressed bytes when choosing
one child; that does not establish a stable complete latency allowance or
physical DRAM saving. Source quality and cheap learned selection remain untested.
Do not launch a donor-quality pilot or another thread/warmup grid for this
unchanged representation based on the best observed time.

The first numeric stage passed, but the first timing launcher stopped before
any worker because PE link timestamps differed. Its fault receipt is retained;
`--no-insert-timestamp` repairs only build metadata. The second numeric and
fixed timing sequence above use byte-identical binaries. No fault was deleted.

[Protocol](GRANITE_CHILD256_COST_PROTOCOL_20261015.md),
[binding](granite_child256_cost_binding_20261015.json),
[numeric receipt](granite_child256_cost_numeric_20261015.json),
[all timing receipts/gates](granite_child256_cost_result_20261015.json),
[first fault](granite_child256_cost_first_fault_20261015.json).
Raw logs and exclusive receipts remain in the bound results directory.

Next: the large-family protected-core screen, not promotion of a small model.
