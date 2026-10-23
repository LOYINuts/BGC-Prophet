#!/usr/bin/env bash
# Reuse the previously checksum-verified official data and fresh NG ESM2 encoding.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export HTTP_PROXY=http://127.0.0.1:7891 HTTPS_PROXY=http://127.0.0.1:7891
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
python_path="$PWD/.venv/bin/python"
official=/tmp/bgc-prophet-gonghang/official17112180
training_csv="$official/BGC_train_dataset_inbalence.csv"
train_lmdb="$official/extracted/lmdb_BGC"
output=runs/detection_17112180_ng_esm2_paper
mkdir -p runs
status="$PWD/runs/paper-experiment.status"
phase() { printf '%s %s\n' "$(date -Is)" "$*" | tee "$status"; }
trap 'phase "FAILED at line $LINENO"' ERR

phase 'Checking verified data and pinned environment'
test -f "$official/extraction.done"
test -f "$train_lmdb/data.mdb"
test -f data/ng/esm2_encoding.done
test -f data/ng/esm2_lmdb/data.mdb
test ! -e "$output"
printf 'fb7b8969e3a51f7380984c40539c7dd9  %s\n' "$training_csv" | md5sum -c -
uv pip check --python "$python_path"
uv run --no-project --python "$python_path" python -m unittest discover -s tests -v
phase 'Training paper protocol: LR 0.01, batch 64, patience 20, constant LR'
uv run --no-project --python "$python_path" python -u -m bgc_prophet.train.reproduce \
    --dataset "$training_csv" --lmdb "$train_lmdb" \
    --test-dataset data/ng/prepared/dataset.csv --test-lmdb data/ng/esm2_lmdb \
    --seeds 1301 1302 1303 1304 1305 --learning-rate 0.01 --batch-size 64 \
    --patience 20 --output "$output"
phase "COMPLETE: $output/summary.json"
