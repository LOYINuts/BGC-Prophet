#!/usr/bin/env bash
# Diagnose LR using validation only, then run the user's approved legacy baseline.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
python_path="$PWD/.venv/bin/python"
official=/tmp/bgc-prophet-gonghang/official17112180
csv="$official/BGC_train_dataset_inbalence.csv"
lmdb="$official/extracted/lmdb_BGC"
pilot=runs/diagnostic_lr0005_batch64_seed1301
baseline=runs/detection_17112180_ng_esm2_legacy_complete
mkdir -p runs
test ! -e "$pilot"
test ! -e "$baseline"
status=runs/investigation.status
trap 'printf "%s FAILED at line %s\n" "$(date -Is)" "$LINENO" > "$status"' ERR
printf '%s Validation-only LR diagnostic\n' "$(date -Is)" > "$status"
uv run --no-project --python "$python_path" python -u -m bgc_prophet.train.reproduce \
  --dataset "$csv" --lmdb "$lmdb" --seeds 1301 --learning-rate 0.0005 \
  --batch-size 64 --scheduler none --patience 20 --epochs 12 --output "$pilot"
uv run --no-project --python "$python_path" python - <<'PY'
import json
from pathlib import Path
root = Path('runs')
pilot = root / 'diagnostic_lr0005_batch64_seed1301/seed_1301'
reference = root / 'detection_17112180_ng_esm2_paper/seed_1301'
a, b = [json.loads((p / 'config.json').read_text()) for p in (pilot, reference)]
for key in ('dataset_sha256', 'batch_size', 'nhead', 'encoder_layers', 'weight_decay'):
    assert a[key] == b[key], key
assert json.loads((pilot/'split.json').read_text()) == json.loads((reference/'split.json').read_text())
report = {'comparison': 'same seed, split, model, batch and constant LR; only LR differs over first 12 epochs', 'ng_used_for_diagnostic': False}
for label, folder in (('lr0.0005', pilot), ('lr0.01', reference)):
    rows = [json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()][:12]
    best = min(rows, key=lambda x: x['validation']['bce'])
    report[label] = {'epochs_compared': len(rows), 'best_epoch': best['epoch'],
                     'best_train': best['train'], 'best_validation': best['validation']}
(root/'lr-diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
PY
printf '%s Full legacy baseline: 5 seeds, LR0.0005 batch32 original decay 200 epochs\n' "$(date -Is)" > "$status"
uv run --no-project --python "$python_path" python -u -m bgc_prophet.train.reproduce \
  --dataset "$csv" --lmdb "$lmdb" --test-dataset data/ng/prepared/dataset.csv \
  --test-lmdb data/ng/esm2_lmdb --seeds 1301 1302 1303 1304 1305 \
  --learning-rate 0.0005 --batch-size 32 --scheduler original --patience 0 \
  --epochs 200 --output "$baseline"
printf '%s COMPLETE %s/summary.json\n' "$(date -Is)" "$baseline" > "$status"
