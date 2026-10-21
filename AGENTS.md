# Agent workflow

## Scope and ordering

- Retrain detection model only, using seeds 1301, 1302, 1303, 1304, 1305.
- Use repository settings. Do not silently change model architecture or data splitting.
- Edit and validate locally, commit and push to origin, then connect to server
  and run.
- Manage Python with uv locally and remotely; use Python 3.12.
- Download large datasets only on server. Local tracked CSV files are already present.
- Never commit credentials, datasets, embeddings, logs, environments or checkpoints.

## Server

Connect with `ssh -p 4618 gonghang@10.157.197.4`.
Check existing repository directories, disk space, GPU use and uv before running.
Use an idle GPU and record its index. Do not stop other users' processes.
Server repository and data paths must be discovered on first connection.
Use persistent logs and detached execution for long downloads and training.

## Data

Source: <https://zenodo.org/records/8104551>.

- `BGC_train_dataset.csv`: 42,096,283 bytes;
  MD5 `e55a046f77c4baef3b6310bfc77dd923` (matches tracked CSV).
- `lmdb_train_dataset.tar.gz`: 5,591,945,924 bytes;
  MD5 `bd839bb1b7f98ea157b9b12b84b44270`.
- Download URL:
  <https://zenodo.org/api/records/8104551/files/lmdb_train_dataset.tar.gz/content>.
- Check archive checksum and list members before extracting into a dedicated directory.
- Training CSV: 16,012 rows; test CSV: 524 rows; each row has 128 gene embeddings.
- Test embeddings are required too. Check coverage before training; do not assume
  training LMDB contains Nine-genomes embeddings.

## Environment and commands

```bash
uv venv --python 3.12
uv pip sync --python .venv/bin/python requirements-training.txt
uv run --no-project --python .venv/bin/python \
  python -m unittest discover -s tests -v
uv run --no-project --python .venv/bin/python \
  python -m bgc_prophet.train.reproduce \
  --lmdb /absolute/path/to/training_lmdb \
  --test-lmdb /absolute/path/to/test_lmdb --check-only
CUDA_VISIBLE_DEVICES=0 uv run --no-project --python .venv/bin/python \
  python -u -m bgc_prophet.train.reproduce \
  --lmdb /absolute/path/to/training_lmdb \
  --test-lmdb /absolute/path/to/test_lmdb --output runs/detection
```

On Windows use `.venv/Scripts/python.exe` instead of `.venv/bin/python`.
Regenerate dependency pins with
`uv pip compile --python-version 3.12 --universal requirements-training.in -o requirements-training.txt`.
PyTorch 2.5.1 Linux default wheels use CUDA 12.4; check driver compatibility.

## Experimental protocol

- ESM2-8M layer 6, 320 dimensions, sequence length 128.
- Transformer: 5 heads, 2 encoder layers, feedforward 1280,
  transformer dropout 0.1, MLP dropout 0.1.
- BCE, AdamW, batch 32, LR 0.0005, weight decay 0.01, 200 epochs.
- Original exponential LR schedule: gamma 0.95 after epochs >= 5 while LR > 5e-6.
- Original per-seed, label-stratified 80/20 row split; save indices.
- Select minimum validation BCE; evaluate selected checkpoint on test once.
- Report threshold 0.5, raw-count precision/recall/F1, AUROC and average precision.
- Aggregate mean and sample standard deviation (ddof=1) across five seeds.
- Deterministic PyTorch algorithms, seeded Python/NumPy/PyTorch; workers 0 for LMDB.
- Preserve outputs: existing seed directories cause an error. Use another
  output directory for a new experiment. Partial runs currently have no resume
  feature.
- Known limitation: original row split overlaps training and validation genes
  (51,976 shared genes for seed 1301). This preserves original splitting.
  Nine-genomes shares no gene identifiers with training CSV.
- Best checkpoint selection is documented here; original scripts save every
  epoch without an automatic selection rule. Results do not claim bitwise
  reproduction of published weights or threshold-0.03 cluster results.

## Formatting

After editing Chinese documents, run `autocorrect --fix FILE`.
After editing Markdown, run `rumdl fmt FILE` then `rumdl check FILE`.
Do not format unrelated files. Under Git Bash, never redirect to `nul` or `NUL`.
