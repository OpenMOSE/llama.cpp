#!/bin/bash
# multikey3 needle check: 3 keys at 10/50/90% depth, one question each.
#
# MK3_N defaults to 256, not 8: a model that opens a thinking block (qwen3.6 does on this
# prompt, and --reasoning-budget 0 does not stop it because -no-cnv means no chat template)
# spends the first tokens reasoning and reaches the answer later. At 8 tokens qwen scored
# 0/3 even DENSE, which hid a real LoD difference underneath (dense 3/3, top32 1/3).
# usage: lod-needle-mk3.sh <model.gguf> <6k|100k> <ctx> [extra llama-cli args...]
M=$1; SZ=$2; CTX=$3; shift 3
VALS=(4172 7391 8253)
PASS=0
for i in 1 2 3; do
    OUT=$(./build/bin/llama-cli -m "$M" -f /home/mose/Projects/llm/lod-mk3-$SZ-q$i.txt \
        -n ${MK3_N:-256} --temp 0 -ngl 99 -c $CTX -ub 2048 -st -no-cnv -fa on --reasoning-budget 0 --simple-io "$@" 2>/dev/null | tail -c 4000)
    if echo "$OUT" | grep -q "${VALS[$((i-1))]}"; then
        PASS=$((PASS+1)); echo "q$i: OK (${VALS[$((i-1))]})"
    else
        echo "q$i: MISS (wanted ${VALS[$((i-1))]}, got: $(echo $OUT | cut -c1-60))"
    fi
done
echo "mk3 result: $PASS/3"
