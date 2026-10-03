# METH-359: actual physical topology/consumed cost analysis PASS

Freezeac39fd6; foreground exit0; raw SHA
`45fd854eb7032a2c6a8804079410f0ecf197b70eabc02fc9d5b6f756d1424381`.
MAIN.375s/endRSS36,331,520B. Read-only exact358 raw, no inference/timing worker
or affinity change. Windows actual RelationProcessorCore reports six disjoint
SMT pairs[0,1],[2,3],[4,5],[6,7],[8,9],[10,11], all12 currently process-allowed.
Prospective one-logical-per-actual-core selection[0,2,4,6,8,10], mask1365;
physical masks/counts validated against psutil6C/12T, single processor group.
This is an observed hardware mapping, not an assumed odd/even convention.

Source64/CPU6 decode primary totals271.7196/251.5323/246.9911ms, max/min
1.100118992. FULL totals553.1811/503.0068/519.4443ms, ratio1.099748751.
CrossKV varies17.9915/11.5028/12.5041ms, ratio1.564097; encoder ratio1.097921.
Decoder row step medians8.14485/7.73820/7.66685ms; max/median1.360934/
1.129216/1.120186. Slow first primary has17.0141ms summed positive excess
above its own step median, largest single excess2.93975ms. Variation therefore
is not one single >24ms stalled token in that row. Source9/CPU6 likewise has
broader variance and10.3134ms maximum step versus6.7566ms row median.
No causal claim about migration/SMT/frequency/OS noise or DRAM from these
summaries.358 frozen decode-repeat criterion FAIL remains, even though FULL
repeat ratio happens to pass. No threshold reinterpretation/rerun.

## Next isolated scheduling variable

Freeze360 SAME356 executable/338 payload, constrain EACH native child process
to the observed one-logical-per-physical-core CPU set, read back actual mask.
Fresh topology closure and original full output/counter bridges required;
fixed358 fixtures/order/one warmup/3 primaries and ALL unchanged gates.
All model mathematics/learned capacity stay the same, runtime scheduling
profile explicitly changes. This test is motivated, not already a stability
improvement. PASS requires NEW full original-primary quality before accepted
natural FULL rate.351 quality/358 cost failures retained, final goal open.
Reproduction: [359 protocol](METH_359_SWITCH_COST_TOPOLOGY_PROTOCOL_20261004.md).
