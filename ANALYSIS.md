# Repository analysis and detection reproduction

This repository detects biosynthetic gene clusters from protein sequences, then
classifies their products. Expected use: one researcher processing genomes or
running GPU experiments. Analysis covers package entry points, preprocessing,
embedding extraction, both model architectures, training and evaluation scripts,
packaging, tracked data and duplicate source files.

## Architecture and data

`bgc_prophet/main.py` registers extract, organize, split, predict, output,
classify and pipeline commands. Extraction uses frozen ESM2-8M protein embeddings
(layer 6, 320 dimensions) stored as pickled tensors in LMDB. Genomes become
128-gene windows. A Transformer produces one binary BGC probability per gene;
postprocessing joins predictions into candidate clusters. A second Transformer
produces seven product-class probabilities.

Training has 16,012 windows, with no duplicate complete gene sentences.
Classification has 12,510 windows. Nine-genomes test has 524 windows.
All tracked windows and gene-label sequences have length 128.
Training CSV MD5 matches Zenodo record 8104551. Training contains 1,777,142 unique
identifiers; test contains 67,072; identifier intersection is empty. Different
identifiers do not establish absence of sequence homology.

## Must fix or account for

1. **Installed CLI fails on imports** (`bgc_prophet/main.py:10`).
   The console entry point imports top-level `command`, although installed
   modules live under `bgc_prophet.command`. Fixed with relative imports so
   the ESM2 extraction CLI can run in the installed package.

2. **Original training cannot import the model consistently**
   (`bgc_prophet/train/trainer.py:13`). Trainers use top-level imports, while
   model and classifier use relative imports. Direct script and module execution
   each fail on different imports. The new reproduction module consistently uses
   package imports; original alternative trainers remain unchanged.

3. **Validation shares genes with training** (`bgc_prophet/train/data.py:44`).
   The split shuffles windows independently. Seed 1301 has 51,976 shared gene
   identifiers across training and validation. Keep this for requested original
   protocol reproduction, but do not describe validation as independent.
   A future benchmark should group source clusters/genomes before splitting.

4. **Old reported Precision/F1 use incorrect counts**
   (`bgc_prophet/train/final_val.py:20`, `bgc_prophet/train/eval_all.py:104`).
   Rows of confusion matrix are normalized separately before calculating
   precision and F1. This changes class prevalence and therefore these metrics;
   reported accuracy is balanced accuracy. New reproduction uses raw counts via
   scikit-learn. Original plotting/evaluation scripts remain affected.

5. **Checkpoint format differs between training and inference**
   (`bgc_prophet/train/trainer.py:98`, `bgc_prophet/command/predict.py:74`).
   Original trainer saves a model object; inference expects a state dictionary.
   New reproduction saves compatible `best.pt` and `last.pt` state dictionaries.
   Old full-model checkpoints need an explicit trusted conversion.

6. **Original training loss omits final batches**
   (`bgc_prophet/train/trainer.py:88`). Accumulators enter epoch totals only
   when batch index is divisible by logging interval; remaining batches vanish.
   Tensor losses also retain graph references between logging steps.
   New reproduction uses detached, element-weighted totals across all batches.

7. **Original accuracy does not measure binary accuracy**
   (`bgc_prophet/train/utils.py:14`). Values below 0.1 become -1, so negative
   targets never match those predictions; values between 0.1 and 0.5 remain
   probabilities. Trainers divide by positive labels rather than element count.
   New reproduction uses explicit threshold 0.5 and standard metrics.

8. **Untrusted CSV text can execute code**
   (`bgc_prophet/train/data.py:24`, `bgc_prophet/command/split.py:28`).
   Both originally call `eval` on file contents. Shared DataReader now uses
   `ast.literal_eval` to retain numeric-literal compatibility without execution.
   Genome splitting still uses `eval`; replace it before accepting untrusted
   organize CSVs. LMDB pickle records also require trusted data sources.

9. **Short-genome handling can hang or produce bad labels**
   (`bgc_prophet/command/split.py:30`). Empty genome repeatedly doubles an empty
   list forever. Nonempty short genomes receive two labels for 128 genes.
   Reject empty genomes and create 128 placeholder labels. This preprocessing
   path is outside detection retraining and remains unchanged.

10. **Cluster thresholds and lengths have boundary errors**
    (`bgc_prophet/command/output.py:63`). Threshold conversion mutates values to
    1 before applying `<= threshold`; at threshold 1 positives disappear.
    `max_range` counts end minus start rather than gene count; a two-gene
    cluster fails `min_count=2`. Use one boolean comparison and inclusive length.
    These change cluster metrics, not new gene-level evaluation.

## Should fix

11. **Classification mask uses binary labels as additive attention bias**
    (`bgc_prophet/train/classifier.py:55`). Float 0/1 padding mask adds attention
    scores instead of excluding padding; mean pooling includes all positions.
    Decide intended masking semantics before classification retraining. Changing
    it silently would change model behavior. Classification is not retrained here.

12. **Pipeline standalone mode passes strings where Path is required**
    (`bgc_prophet/command/pipline.py:139`). Standalone parser produces strings,
    downstream code calls `.joinpath`, causing an exception. Use same Path types
    and command handler as installed entry point.

13. **Dependencies and worker assumptions prevent portable training**
    (`setup.py:25`, `bgc_prophet/train/trainer.py:11`). Original requirements omit
    tensorboard and timm, and current schedulers reject old `verbose` argument.
    LMDB environments opened before worker spawning are not Windows portable.
    New pinned training requirements remove need for timm, and new runner uses
    workers 0 with shared, explicitly owned LMDB environments.

14. **Duplicate trainers and source copies drift** (`src/`,
    `bgc_prophet/train/trainer_focal_loss.py`). Alternative trainers repeat large
    loops and import issues. Keep archived copies until external usage is known;
    use the new package module as the documented reproduction entry point.

## New experiment outputs

`uv run --no-project --python .venv/bin/python python -m bgc_prophet.train.reproduce`
trains detection for seeds 1301–1305. Each seed has configuration and split
indices, per-epoch JSON logs, best/last weights, test metrics and predictions.
Model selection uses minimum validation BCE; test is evaluated once afterward.
Threshold is fixed at 0.5. Aggregate includes mean and sample standard deviation.
This is gene-level evaluation; it does not reproduce cluster-level threshold
0.03 results or classify clusters.

Local synthetic tests check raw-count metrics on imbalanced labels, two complete
seed runs, independent output files, reproducible repeated predictions,
checkpoint loading and refusal to overwrite previous runs. Real embedding and
GPU validation must run on server before training starts.

## Reproduction settings and commands

Use Python 3.12 and `requirements-training.txt`. Pins include PyTorch 2.5.1;
Linux wheels use CUDA 12.4, compatible with server driver 560.28.03.

Settings retained from original scripts and checkpoint names: 320 dimensions,
128 genes, 5 heads, 2 layers, feedforward 1280, both dropouts 0.1, BCE, AdamW,
weight decay 0.01, batch 32, learning rate 0.0005, 200 epochs. Exponential LR
uses gamma 0.95 after epochs >= 5 while LR > 5e-6. Split is original per-seed
stratified 80/20 rows. New selection rule is minimum validation BCE; original
trainer saved every epoch without a selection rule. Threshold 0.5 is fixed.

```bash
uv venv --python 3.12
uv pip sync --python .venv/bin/python requirements-training.txt
uv run --no-project --python .venv/bin/python \
  python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv/bin/python \
  python -u -m bgc_prophet.train.reproduce \
  --lmdb /path/to/training_lmdb --test-lmdb /path/to/test_lmdb \
  --output runs/detection
```

Run same training command with `--check-only` before training. Omit `--test-lmdb`
to train and report selected-checkpoint validation metrics only; output explicitly
labels this `evaluation_split=validation`. This is not independent test evidence.
On Windows use `.venv/Scripts/python.exe`. Existing seed directories cause an
error to protect results. Partial experiments have no resume feature.
Deterministic algorithms
and seeded Python/NumPy/PyTorch are enabled. LMDB uses one loader process.

Source: <https://zenodo.org/records/8104551>. Training CSV is 42,096,283 bytes,
MD5 `e55a046f77c4baef3b6310bfc77dd923`.
`lmdb_train_dataset.tar.gz` is 5,591,945,924 bytes,
MD5 `bd839bb1b7f98ea157b9b12b84b44270`. Download only on server, verify checksum,
and inspect archive members before extraction. Test embeddings need separate
coverage verification.

Server repository is `/home/gonghang/codes/BGC-Prophet`. Server proxy is
`http://127.0.0.1:7892`; apply HTTP_PROXY and HTTPS_PROXY to dependency installation
and downloads. User chose GPU 1 despite an existing shared GPU workload.
Official training LMDB must be downloaded directly from Zenodo and verified
with the published archive MD5. No Parquet conversion is used.

Our independent NG test source is
`/home/gonghang/codes/my-bgc-v2/data/raw/ng/`: GenBank
`ClusterFinder_Annotated_Contigs.full.gbk` and annotations `291BGC.csv`.
Migrate copies into this repository's ignored `data/ng/raw/` directory.
`scripts/prepare_ng.py` produces protein FASTA, 128-gene windows, binary labels
and source checksums. Labels follow locus-tag membership in annotation CSV.
Windows stay within contigs; final partial window uses last 128 genes.
Metrics count gene positions in windows, including overlapping tails.

NG has 67,148 CDS proteins, all with translations. Encode these proteins afresh
using official `esm2_t6_8M_UR50D`, layer 6 mean representations, truncation 1022,
on GPU 1. Do not reuse any existing NG embedding files from other projects.

```bash
uv run --no-project --python .venv/bin/python python scripts/prepare_ng.py \
  --genbank data/ng/raw/ClusterFinder_Annotated_Contigs.full.gbk \
  --annotations data/ng/raw/291BGC.csv --output data/ng/prepared
CUDA_VISIBLE_DEVICES=1 uv run --no-project --python .venv/bin/python \
  bgc_prophet extract esm2_t6_8M_UR50D data/ng/prepared/proteins.faa \
  data/ng/esm2_lmdb --include mean --repr_layers 6 \
  --toks_per_batch 4096 --write_batches 100
```

Pass `--test-dataset data/ng/prepared/dataset.csv` and
`--test-lmdb data/ng/esm2_lmdb` when training. This test set is our NG dataset,
not the original tracked test CSV. Original CSV identifiers differ from the
supplied GenBank at 592 positions across 9 windows; original CSV stays intact.

Verdict: original training and metric scripts require corrections before their
numbers are reliable. New reproduction bypasses those failures while preserving
architecture, optimizer, learning-rate schedule and original row split.

Not checked: original published weight equivalence, source-sequence homology,
real embedding coverage and runtime GPU behavior. Large datasets are deliberately
not downloaded locally. No unrelated inference or classification redesign.
