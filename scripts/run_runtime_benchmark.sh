#!/usr/bin/env bash

set -uo pipefail

NAME="$1"
OUT="$2"

mkdir -p "$OUT"

GPU_LOG="$OUT/gpu_memory.csv"
TIME_LOG="$OUT/time.txt"
STOP_FILE="$OUT/.stop_gpu_monitor"

rm -f "$STOP_FILE"

echo "timestamp_ms,memory_used_mib" > "$GPU_LOG"

# ============================================================
# GPU memory monitor
# ============================================================
(
    while [ ! -f "$STOP_FILE" ]; do

        TS=$(date +%s%3N)

        MEM=$(nvidia-smi \
            --id=0 \
            --query-gpu=memory.used \
            --format=csv,noheader,nounits \
            2>/dev/null \
            | head -n1 \
            | tr -d ' ')

        if [ -n "$MEM" ]; then
            echo "${TS},${MEM}" >> "$GPU_LOG"
        fi

        sleep 0.2
    done
) &

MON_PID=$!

cleanup() {
    touch "$STOP_FILE" 2>/dev/null || true
    wait "$MON_PID" 2>/dev/null || true
    rm -f "$STOP_FILE"
}

trap cleanup EXIT INT TERM

echo "============================================================"
echo "BENCHMARK: $NAME"
echo "============================================================"

START_NS=$(date +%s%N)

set +e

env PYTHONPATH=. \
python -u scripts/eval_zs_split.py \
    --title "$NAME" \
    --model AKGVPModel \
    --load-model "$BASE500" \
    --gpu-ids 0 \
    --images-file-name clip_featuremap.hdf5 \
    --detection-feature-file-name det_feature_22_cates.hdf5 \
    --data-dir datasets/Scene_Data \
    --results-path "$OUT" \
    --results-json metrics.json \
    2>&1 | tee "$OUT/run.log"

STATUS=${PIPESTATUS[0]}

set -e

END_NS=$(date +%s%N)

ELAPSED_SEC=$(python - "$START_NS" "$END_NS" <<'PY'
import sys

start = int(sys.argv[1])
end = int(sys.argv[2])

print(f"{(end - start) / 1e9:.6f}")
PY
)

cleanup
trap - EXIT INT TERM

echo "ELAPSED_SEC=$ELAPSED_SEC" > "$TIME_LOG"
echo "EXIT_STATUS=$STATUS" >> "$TIME_LOG"

echo
echo "============================================================"
echo "TIMING"
echo "============================================================"
cat "$TIME_LOG"

exit "$STATUS"
