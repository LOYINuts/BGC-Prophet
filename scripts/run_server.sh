#!/usr/bin/env bash
# Run from server checkout after local validation, commit and push.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export HTTP_PROXY=http://127.0.0.1:7892 HTTPS_PROXY=http://127.0.0.1:7892
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python_path="$PWD/.venv/bin/python"
official=/tmp/bgc-prophet-gonghang/official
archive="$official/lmdb_train_dataset.tar.gz"
mkdir -p runs
status="$PWD/runs/experiment.status"
phase() { printf '%s %s\n' "$(date -Is)" "$*" | tee "$status"; }
trap 'phase "FAILED at line $LINENO"' ERR

phase 'Waiting for verified dependency wheel downloads'
while pgrep -u "$(id -u)" -f '^aria2c .*wheel-(mirrors|downloads)' > /dev/null; do
    sleep 30
done
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

phase 'Waiting for official Zenodo archive download'
while pgrep -u "$(id -u)" -f '^aria2c .*lmdb_train_dataset.tar.gz' > /dev/null; do
    sleep 30
done
phase 'Checking official archive MD5'
printf 'bd839bb1b7f98ea157b9b12b84b44270  %s\n' "$archive" | md5sum -c -
if [[ ! -f "$official/extraction.done" ]]; then
    phase 'Listing and extracting official LMDB'
    mkdir -p "$official/extracted"
    tar --use-compress-program=pigz -tf "$archive" > "$official/archive-members.txt"
    uv run --no-project --python "$python_path" python - "$official/archive-members.txt" <<'PY'
from pathlib import Path, PurePosixPath
import sys
members = Path(sys.argv[1]).read_text().splitlines()
assert members and any(name.endswith('/data.mdb') for name in members), members
assert all(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts for name in members)
print('Archive members:', members)
PY
    tar --use-compress-program=pigz --no-same-owner -xf "$archive" -C "$official/extracted"
    touch "$official/extraction.done"
fi
mapfile -t databases < <(find "$official/extracted" -name data.mdb -type f)
if [[ ${#databases[@]} != 1 ]]; then
    phase 'FAILED: expected exactly one official data.mdb'
    exit 1
fi
train_lmdb="$(dirname "${databases[0]}")"
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
phase 'Training seeds 1301-1305 and evaluating each selected checkpoint on our NG test'
uv run --no-project --python "$python_path" python -u -m bgc_prophet.train.reproduce \
    --lmdb "$train_lmdb" --test-dataset data/ng/prepared/dataset.csv \
    --test-lmdb data/ng/esm2_lmdb --output runs/detection_official_ng_esm2
phase 'COMPLETE: runs/detection_official_ng_esm2/summary.json'
