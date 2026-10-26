#!/usr/bin/env bash
# Stop the old runner explicitly before launching this script. Never kill unrelated jobs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
python_path="$PWD/.venv/bin/python"
base=runs/detection_17112180_ng_esm2_legacy_complete
output=runs/legacy_remaining_parallel
official=/tmp/bgc-prophet-gonghang/official17112180
test -f "$base/seed_1301/test_metrics.json"
test ! -e "$output"
# Only unfinished seeds may be retrained by this launcher.
for seed in 1301 1302; do
    test -f "$base/seed_$seed/test_metrics.json"
done
for seed in 1303 1304 1305; do
    if test -f "$base/seed_$seed/test_metrics.json"; then
        echo "Seed $seed already completed; revise assignment before launch" >&2
        exit 1
    fi
done
if pgrep -u "$(id -u)" -f 'python.*-m bgc_prophet.train.reproduce' > /dev/null; then
    echo 'Existing reproduction process found; stop the intended old experiment first' >&2
    exit 1
fi
mkdir -p "$output"
run_worker() {
    local gpu=$1
    shift
    for seed in "$@"; do
        CUDA_VISIBLE_DEVICES="$gpu" uv run --no-project --python "$python_path" \
          python -u -m bgc_prophet.train.reproduce \
          --dataset "$official/BGC_train_dataset_inbalence.csv" \
          --lmdb "$official/extracted/lmdb_BGC" \
          --test-dataset data/ng/prepared/dataset.csv --test-lmdb data/ng/esm2_lmdb \
          --seeds "$seed" --learning-rate 0.0005 --batch-size 32 \
          --scheduler original --patience 0 --epochs 200 \
          --output "$output/seed_$seed" > "$output/seed_$seed.log" 2>&1
    done
}
run_worker 0 1303 & p0=$!
run_worker 1 1304 & p1=$!
run_worker 2 1305 & p2=$!
failed=0
for pid in "$p0" "$p1" "$p2"; do
    wait "$pid" || failed=1
done
if (( failed )); then
    echo 'One or more workers failed; preserve logs and outputs, do not claim completion' >&2
    exit 1
fi
uv run --no-project --python "$python_path" python scripts/summarize_parallel.py
