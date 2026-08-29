#!/bin/sh
set -e
cd "$(git rev-parse --show-toplevel)"
R=benchmarks/donor_adaptation/p1/results/stage2
rm -f "$R/STOP_SAMPLER"
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/donor_adaptation/p1/sample_machine.ps1 \
    -Out "$R/machine_samples.csv" -Stop "$R/STOP_SAMPLER" -IntervalMs 1500 &
SAMP=$!
sleep 2
for inv in 1 2 3 4 5; do
    echo "=== invocation $inv ===" 1>&2
    ./bin/nibble_pack_bench.exe --inv $inv --target-gb 20 --pool-mb 512 --seed $((0x1000+inv)) \
        >> "$R/stage2_raw.csv" 2>> "$R/stage2_raw.err"
done
touch "$R/STOP_SAMPLER"
sleep 3
kill $SAMP 2>/dev/null || true
echo "STAGE2 RUNS COMPLETE"
