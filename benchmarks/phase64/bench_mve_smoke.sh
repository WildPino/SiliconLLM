#!/usr/bin/env bash
# Phase 64.4 / MVE — the apparatus smoke, one command: `bash benchmarks/phase64/bench_mve_smoke.sh`
#   Validates the WHOLE pipeline at tiny scale on the local 3060, in the order the real run executes it.
#   Nothing here is a gate: the gates are in benchmarks/phase64/MVE_PREREG.md and are read on the owner's run.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/Scripts/python.exe}
M=benchmarks/phase64/mve

echo "== (A) data + tokenizer + anchors + decomp (2 MiB slice) =="
$PY $M/mve_data.py --smoke 2>/dev/null | grep -E "^\[A|anchor rate"

echo
echo "== (B) teacher logits, chunked (teacher-forced prefill; the KD economic input) =="
$PY $M/mve_logits.py --tag smoke --backend hf --quant fp16 --ctx 2048 --stride 1536 --batch 4 \
    --max-tok 60000 2>/dev/null | grep -E "self-check|SCORING|chunks"

echo
echo "== (D3) the information diagnostic: how much of the teacher survives the projection? =="
$PY $M/kd_information.py --tag smoke 2>/dev/null | sed -n '/====/,/^$/p' | grep -vE "^====|^$"

echo
echo "== (C-F) the curriculum, three arms (120 steps -- a SMOKE, not a gate reading) =="
printf "  %-14s %-9s %-9s %s\n" "arm" "BPB C" "BPB F" "(val BPB, TinyStories)"
for cfg in "ce anchor" "kd anchor" "kd span"; do
  set -- $cfg
  r=$($PY $M/mve_train.py --smoke --arm $1 --kd $2 --device cuda:0 --fp16 2>/dev/null | grep -E "^  [CF] " | tr -s ' ' | cut -d' ' -f5 | tr '\n' ' ')
  printf "  %-14s %s\n" "$1/$2" "$r"
done

echo
echo "== (E) recall arm: does the InfoNCE stage destabilize? (gate = D4 clause 2, read on the real run) =="
for r in on off; do
  v=$($PY $M/mve_train.py --smoke --arm kd --kd anchor --recall $r --device cuda:0 --fp16 2>/dev/null | grep -E "^  E " | tr -s ' ')
  echo "  recall=$r  ->$v"
done

echo
echo "== licences + Stack-v2 content path =="
$PY $M/data_smoke.py 2>/dev/null | grep -E "apache|qwen-research|gemma|content column|401"

echo
echo "STOP. Apparatus smoke above. Gates and the pre-registration: benchmarks/phase64/MVE_PREREG.md. No commit."
