#!/bin/bash
# multikey3 needle check: 3 keys at 10/50/90% depth, one question each.
# usage: lod-needle-mk3.sh <model.gguf> <6k|100k> <ctx> [extra llama-cli args...]
M=$1; SZ=$2; CTX=$3; shift 3
VALS=(4172 7391 8253)
PASS=0
for i in 1 2 3; do
    OUT=$(./build/bin/llama-cli -m "$M" -f /home/mose/Projects/llm/lod-mk3-$SZ-q$i.txt \
        -n ${MK3_N:-8} --temp 0 -ngl 99 -c $CTX -ub 2048 -st -no-cnv -fa on --reasoning-budget 0 --simple-io "$@" 2>/dev/null | tail -c 800)
    if echo "$OUT" | grep -q "${VALS[$((i-1))]}"; then
        PASS=$((PASS+1)); echo "q$i: OK (${VALS[$((i-1))]})"
    else
        echo "q$i: MISS (wanted ${VALS[$((i-1))]}, got: $(echo $OUT | cut -c1-60))"
    fi
done
echo "mk3 result: $PASS/3"
