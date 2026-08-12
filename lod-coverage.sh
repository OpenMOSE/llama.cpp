#!/bin/bash
# Coverage benchmark: many requirements scattered through a long context, one long
# generation that must satisfy all of them. Unlike the needle test (one fact, asked
# once, answered in the first tokens) this measures whether the model can keep
# re-reading DIFFERENT parts of the context while generating - the axis a decode-side
# page budget is supposed to buy.
#
# usage: lod-coverage.sh <model.gguf> <ctx_tokens> <n_ctx> [extra llama-cli args...]
M=$1; SZ=$2; CTX=$3; shift 3
ASSET=${LOD_ASSET_DIR:-/home/mose/Projects/llm}/lod-cov-$SZ.txt

KEYS=(alpha_timeout bravo_retries cobalt_port delta_window echo_depth foxtrot_limit \
      golf_stride hotel_margin india_offset juliet_quota kilo_budget lima_ceiling)
VALS=(4172 7391 8253 5064 9318 2607 6145 3872 1509 8934 2761 5528)

if [ ! -f "$ASSET" ]; then
    python3 - "$ASSET" "$SZ" "${KEYS[*]}" "${VALS[*]}" <<'PY'
import sys
path, size, keys, vals = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(), sys.argv[4].split()
# filler that reads like a spec document but carries no numbers of its own
topics = ["retention", "throughput", "isolation", "telemetry", "failover", "sharding",
          "quorum", "backpressure", "checkpointing", "reconciliation"]
def filler(i):
    t = topics[i % len(topics)]
    return f"Note {i}: the {t} subsystem was reviewed this cycle."
lines, n = [], len(keys)
lines.append("SYSTEM CONFIGURATION SPECIFICATION")
# a filler line is ~14 tokens for this tokenizer; leave room for the answer
per = max(1, (size - 600) // (14 * n))
for k in range(n):
    for j in range(per):
        lines.append(filler(k * per + j))
    lines.append(f"REQUIREMENT {k+1}: the setting `{keys[k]}` must be set to exactly {vals[k]}.")
lines.append("END OF SPECIFICATION.")
lines.append("")
lines.append("Write the complete configuration file implementing every REQUIREMENT above.")
lines.append("Output exactly one line per setting, in the form: key = value")
lines.append("Output only the configuration lines, nothing else.")
open(path, "w").write("\n".join(lines) + "\n")
# probe sidecar: the same requirements as teacher-forcing prompts, so a search can score
# retrieval as a continuous log-probability instead of a 12-point pass/fail
import json
json.dump([{"prefix": f"\n{k} = ", "answer": v} for k, v in zip(keys, vals)],
          open(path + ".probes.json", "w"), indent=1)
PY
    echo "generated $ASSET (and $ASSET.probes.json)"
fi

# asset-only mode: llama-lod-search wants the document and the probe sidecar,
# not the generation benchmark
if [ "${COV_ASSET_ONLY:-0}" = "1" ]; then
    echo "$ASSET"
    exit 0
fi

OUT=$(./build/bin/llama-cli -m "$M" -f "$ASSET" -n ${COV_N:-400} --temp 0 -ngl 99 \
        -c $CTX -ub ${COV_UB:-2048} -st -no-cnv -fa on --reasoning-budget 0 --simple-io "$@" 2>/dev/null)

HIT=0
MISS=""
for i in $(seq 0 $((${#KEYS[@]}-1))); do
    if echo "$OUT" | grep -q "${VALS[$i]}"; then HIT=$((HIT+1)); else MISS="$MISS ${KEYS[$i]}"; fi
done
echo "coverage: $HIT/${#KEYS[@]}${MISS:+   missed:$MISS}"
