#!/usr/bin/env bash
# Run from synchronized server checkout after local validation, commit and push.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export HTTP_PROXY=http://127.0.0.1:7892 HTTPS_PROXY=http://127.0.0.1:7892
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
python_path="$PWD/.venv/bin/python"
official=/tmp/bgc-prophet-gonghang/official17112180
archive="$official/lmdb_BGC.zip"
training_csv="$official/BGC_train_dataset_inbalence.csv"
mkdir -p runs "$official"
status="$PWD/runs/experiment.status"
phase() { printf '%s %s\n' "$(date -Is)" "$*" | tee "$status"; }
trap 'phase "FAILED at line $LINENO"' ERR

phase 'Checking pinned environment and local tests'
uv pip sync --python "$python_path" --find-links /tmp/bgc-prophet-gonghang/wheels requirements-training.txt
uv run --no-project --python "$python_path" python -m unittest discover -s tests -v
phase 'Preparing our NG protein sequences and labels'
if [[ ! -f data/ng/prepared/provenance.json ]]; then
    uv run --no-project --python "$python_path" python scripts/prepare_ng.py \
        --genbank data/ng/raw/ClusterFinder_Annotated_Contigs.full.gbk \
        --annotations data/ng/raw/291BGC.csv --output data/ng/prepared
fi
phase 'Encoding NG with fresh ESM2-8M layer 6 embeddings on GPU 1'
if [[ ! -f data/ng/esm2_encoding.done ]]; then
    if [[ -e data/ng/esm2_lmdb ]]; then
        phase 'FAILED: incomplete NG LMDB exists; inspect before rerunning'
        exit 1
    fi
    uv run --no-project --python "$python_path" bgc_prophet extract \
        esm2_t6_8M_UR50D data/ng/prepared/proteins.faa data/ng/esm2_lmdb \
        --include mean --repr_layers 6 --toks_per_batch 4096 --write_batches 100
    touch data/ng/esm2_encoding.done
fi

phase 'Waiting for complete official Zenodo 17112180 downloads'
while pgrep -u "$(id -u)" -f '^aria2c .*official17112180' > /dev/null; do
    sleep 30
done
# Retry interrupted transfers; aria2 verifies published checksums before returning.
for item in 'lmdb_BGC.zip c7effa7b9b6804409803be975d287224' \
            'BGC_train_dataset_inbalence.csv fb7b8969e3a51f7380984c40539c7dd9'; do
    read -r filename checksum <<< "$item"
    until aria2c --all-proxy="" --continue=true --split=16 \
        --max-connection-per-server=16 --min-split-size=4M --timeout=60 \
        --connect-timeout=20 --file-allocation=none --summary-interval=60 \
        --max-tries=5 --retry-wait=15 --check-integrity=true \
        --checksum="md5=$checksum" --dir="$official" --out="$filename" \
        "https://zenodo.org/api/records/17112180/files/$filename/content"; do
        phase "Retrying official download: $filename"
        sleep 30
    done
done
phase 'Checking official archive and training CSV MD5'
printf 'c7effa7b9b6804409803be975d287224  %s\n' "$archive" | md5sum -c -
printf 'fb7b8969e3a51f7380984c40539c7dd9  %s\n' "$training_csv" | md5sum -c -
if [[ ! -f "$official/extraction.done" ]]; then
    phase 'Validating ZIP members and extracting complete official LMDB'
    uv run --no-project --python "$python_path" python - "$archive" "$official/extracted" <<'PY'
from pathlib import PurePosixPath, Path
import sys
import zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    members = archive.infolist()
    assert members and any(info.filename.endswith('/data.mdb') for info in members)
    for info in members:
        path = PurePosixPath(info.filename)
        assert not path.is_absolute() and '..' not in path.parts and '\\' not in info.filename
        assert (info.external_attr >> 16) & 0o170000 != 0o120000, 'ZIP symlink refused'
        print(info.filename, info.file_size, flush=True)
    Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
    archive.extractall(sys.argv[2])
PY
    touch "$official/extraction.done"
fi
mapfile -t databases < <(find "$official/extracted" -name data.mdb -type f)
if [[ ${#databases[@]} != 1 ]]; then
    phase 'FAILED: expected exactly one official data.mdb'
    exit 1
fi
train_lmdb="$(dirname "${databases[0]}")"
phase 'Recording official paired training dataset statistics'
uv run --no-project --python "$python_path" python - "$training_csv" "$official/dataset-statistics.json" <<'PY'
from collections import Counter
from pathlib import Path
import csv
import json
import sys
with open(sys.argv[1]) as handle:
    rows = list(csv.DictReader(handle))
stats = {'source_record': 17112180, 'rows': len(rows),
         'primary_labels': dict(Counter(row['labels'].split()[0] for row in rows)),
         'sentence_lengths': dict(Counter(len(row['TDsentence'].split()) for row in rows)),
         'label_lengths': dict(Counter(len(row['TDlabels'].split()) for row in rows))}
Path(sys.argv[2]).write_text(json.dumps(stats, indent=2) + '\n')
print(json.dumps(stats), flush=True)
PY
phase 'Checking deterministic GPU forward and backward'
uv run --no-project --python "$python_path" python - <<'PY'
import torch
from bgc_prophet.train.model import transformerEncoderNet
from bgc_prophet.train.reproduce import seed_all
seed_all(1301)
torch.use_deterministic_algorithms(True)
model = transformerEncoderNet(d_model=320, nhead=5, num_encoder_layers=2, dim_feedforward=1280).cuda()
probability = model(torch.randn(2, 128, 320, device='cuda'))
loss = torch.nn.functional.binary_cross_entropy(probability, torch.zeros_like(probability))
loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print('GPU smoke passed:', torch.__version__, torch.cuda.get_device_name(), loss.item())
PY
phase 'Training seeds 1301-1305 on paired official data and testing on our fresh ESM2 NG'
uv run --no-project --python "$python_path" python -u -m bgc_prophet.train.reproduce \
    --dataset "$training_csv" --lmdb "$train_lmdb" \
    --test-dataset data/ng/prepared/dataset.csv --test-lmdb data/ng/esm2_lmdb \
    --output runs/detection_17112180_ng_esm2
phase 'COMPLETE: runs/detection_17112180_ng_esm2/summary.json'
