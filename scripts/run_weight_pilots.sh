#!/usr/bin/env bash
# Training-objective pilots: NG is deliberately excluded from all three runs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
python_path="$PWD/.venv/bin/python"
official=/tmp/bgc-prophet-gonghang/official17112180
output=runs/positive_weight_pilot
if pgrep -u "$(id -u)" -f 'python.*-m bgc_prophet.train.reproduce' > /dev/null; then
    echo 'Existing training found; inspect before launching pilots' >&2
    exit 1
fi
test ! -e "$output"
mkdir -p "$output"
run_pilot() {
    CUDA_VISIBLE_DEVICES="$1" uv run --no-project --python "$python_path" \
      python -u -m bgc_prophet.train.reproduce \
      --dataset "$official/BGC_train_dataset_inbalence.csv" \
      --lmdb "$official/extracted/lmdb_BGC" --seeds 1301 \
      --learning-rate 0.0005 --batch-size 32 --scheduler original \
      --epochs 100 --patience 20 --positive-weight "$2" \
      --output "$output/weight_$2" > "$output/weight_$2.log" 2>&1
}
run_pilot 0 1 & p0=$!
run_pilot 1 2 & p1=$!
run_pilot 2 4 & p2=$!
failed=0
for pid in "$p0" "$p1" "$p2"; do
    wait "$pid" || failed=1
done
if (( failed )); then
    echo 'Pilot failed; inspect per-run logs' >&2
    exit 1
fi
uv run --no-project --python "$python_path" python scripts/compare_weight_pilots.py
