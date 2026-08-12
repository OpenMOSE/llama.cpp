#!/bin/bash
# Per-layer LoD budget search driven by the end-to-end needle benchmark.
#
#   usage: lod-search-niah.sh <model.gguf> [n_ctx] [doc_tokens] [uniform_budget]
#
# Steps, in order, because each one can invalidate the next:
#   0. build the needle documents from the coverage asset (once per doc size)
#   1. confirm the score moves with the budget on THIS model - if it does not,
#      nothing downstream means anything
#   2. per-layer sensitivity: which layers actually carry the retrieval
#   3. search at a fixed total budget, writing <model>-lod.json
#   4. validate the result end to end against the uniform budget it replaces
#
# Run it through simple-run.sh for cluster GPU access:
#   CLUSTER_GPUS=0 bash simple-run.sh bash lod-search-niah.sh /path/model.gguf
set -u

M=$1
CTX=${2:-32768}
SZ=${3:-24000}
TOP=${4:-32}

ASSET_DIR=${LOD_ASSET_DIR:-/home/mose/Projects/llm}
ASSET=$ASSET_DIR/lod-cov-$SZ.txt
NIAH_DIR=$ASSET_DIR/niah-$SZ
OUT=$(basename "$M" .gguf)-lod.json
LOG=$(basename "$M" .gguf)-lod-search.log

# ---- 0. assets
if [ ! -f "$ASSET.probes.json" ]; then
    COV_ASSET_ONLY=1 LOD_ASSET_DIR="$ASSET_DIR" bash lod-coverage.sh "$M" "$SZ" "$CTX" >/dev/null
fi
if [ ! -f "$NIAH_DIR/spec.txt" ]; then
    python3 - "$ASSET" "$NIAH_DIR" <<'PY'
import json, os, sys
src, out_dir = sys.argv[1], sys.argv[2]
probes = json.load(open(src + ".probes.json"))
body = open(src).read().split("END OF SPECIFICATION.")[0] + "END OF SPECIFICATION.\n"
os.makedirs(out_dir, exist_ok=True)
spec = []
for i in range(len(probes)):                       # every planted key, for resolution
    key = probes[i]["prefix"].strip().split(" =")[0]
    path = os.path.join(out_dir, key + ".txt")
    open(path, "w").write(body + "\nQuestion: what value must the setting `%s` be set to?"
                                 "\nAnswer: %s must be set to" % (key, key))
    spec.append("%s=%s" % (path, probes[i]["answer"]))
open(os.path.join(out_dir, "spec.txt"), "w").write(",".join(spec))
print("wrote %d needle documents to %s" % (len(spec), out_dir))
PY
fi
N=$(cat "$NIAH_DIR/spec.txt")
COMMON="-m $M -f $ASSET -c $CTX -ub 2048 -ngl 99 -fa on"

# ---- 1. does the objective respond to the budget on this model?
echo "== step 1: budget response (must be monotonic, or stop here)"
for T in 8 "$TOP" 128; do
    printf "  top %-5s : " "$T"
    ./build/bin/llama-lod-search $COMMON --niah "$N" --eval-top "$T" > "/tmp/lod-niah-$T.log" 2>&1
    grep -E "documents answered|GGML_ASSERT|error:" "/tmp/lod-niah-$T.log" | tail -1
done

# ---- 2+3. sensitivity, then the search itself
echo
echo "== step 2+3: per-layer sensitivity and search (log: $LOG)"
stdbuf -oL -eL ./build/bin/llama-lod-search $COMMON --niah "$N" \
    --search greedy --eval-top "$TOP" --range 8:128 --max-eval 22 -o "$OUT" > "$LOG" 2>&1
sed -n '/ranked by cost/,$p' "$LOG" | sed 's/^[0-9.]* I //'

# ---- 4. validate against the uniform budget it replaces
echo
echo "== step 4: end-to-end validation (searched must not lose to uniform)"
echo -n "  needle mk3 searched  : "; bash lod-needle-mk3.sh "$M" 32k "$CTX" --lod-config "$OUT" 2>&1 | tail -1
echo -n "  needle mk3 uniform   : "; bash lod-needle-mk3.sh "$M" 32k "$CTX" --lod --lod-top-pages "$TOP" 2>&1 | tail -1
echo -n "  coverage   searched  : "; bash lod-coverage.sh "$M" "$SZ" "$CTX" --lod-config "$OUT" 2>&1 | tail -1
echo -n "  coverage   uniform   : "; bash lod-coverage.sh "$M" "$SZ" "$CTX" --lod --lod-top-pages "$TOP" 2>&1 | tail -1
echo
echo "config: $OUT   (use it with: --lod --lod-config $OUT)"
