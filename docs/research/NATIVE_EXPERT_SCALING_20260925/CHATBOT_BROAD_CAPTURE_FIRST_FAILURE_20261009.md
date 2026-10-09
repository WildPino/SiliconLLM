# Broad source capture: first resource failure retained

9 October2026. Original48 capture FAILED;no cap increased and no completed
generation is to be replayed. Freeze8f8e1fbe433b2909a6c8ff0102b744ea84d98a2e,
binding36aa4945bcdc339d7ffca33ce11809c0767fb307457ec14e53d55b092677cefd.
Source package/new48 cohort remain the originally bound bytes.

Worker24232/launcher28288 exit1/session63719 closed. Worker first fault
`AssertionError('GPU allocated cap')` at44.062s,after saving TWO source replies
and full packets in `results/native_expert_scaling/chatbot_broad_capture_20261009`:
broad_fit_apigen_80k_023:118 labels/EOS; broad_fit_apigen_80k_002:741 prompt IDs,
33 labels/EOS. Both finite/lossless BF16/argmax flags pass. Total151 labels/
19,792,174B packets. These are durable partial acquisition,not all48 admission.

Actual GPU allocated peak10,764,644,352B exceeds the fixed10GiB=10,737,418,240B;
reserved12,425,625,600B also exceeds fixed11GiB=11,811,160,064B. Guard stopped at
the first allocated assertion;the reserved violation is visible in its receipt.
Family48.515s/held worker OS3,742,986,240B. GPU values are actual worker-reported
peaks;no missing telemetry is invented. No successful aggregate corpus/result/
terminal receipt exists.6 retained files/20,207,779B,including firstfault and
interaction preflight. No learner/update/native/T4 work occurred.

Static source inspection identifies large SSD broadcast intermediates:
G_intermediate has shape[B,ceil(T/C),C,C,H,N]. At T741/C128/H48/N256/F32 it
contains1,207,959,552 coordinates/4,831,838,208B. Other state/output products
scale as paddedT*H*d*N;the source Python retains several intermediates to return.
This is an algebraic extent deduction,not a measured allocation attribution.

Next new variable:tile ONLY the source chunk axis of the three independent
contractions and concatenate their reduced results. Preserve weights,precision,
chunk128/cache/SSM/attention/multipliers/rounding operators. Each output chunk
uses the same elementwise products and same-axis reductions;no cross-chunk
reduction is split. Peak-size reduction is expected,not yet observed.
[Qualification protocol](CHATBOT_FALCON_SSD_TILES_PROTOCOL_20261009.md) compares
all151 saved source trajectory rows without regenerating either label sequence.
Only if its fixed bit/ID/resource gates pass may a separately frozen repair
reuse these two packets and capture the remaining46.Original resource admission
stays false even if a later family succeeds. Final engine target is unchanged.
