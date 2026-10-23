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
checkpoint loading and refusal to overwrite previous runs. Real embedding
coverage and GPU forward/backward checks passed before the active run.

## Paper protocol correction and restart

The supplied Lai et al. (2025), DOI `10.1093/nar/gkaf305`, PDF page 4,
Materials and methods, "Model architecture and configuration", explicitly states
AdamW, learning rate 1e-2, batch size 64 and no fixed epoch count. Validation
loss must fail to decrease for 20 consecutive epochs before early stopping;
the checkpoint with minimum validation loss is the final model.

The previous 200-epoch run used LR 0.0005, batch 32 and exponential decay,
inferred from repository scripts and model paths before the paper was provided.
It is a repository-configuration experiment, not a faithful reproduction of the
paper's optimization settings. Its results above are retained for provenance.
The user requested stopping that experiment and restarting seeds 1301-1305
under the paper settings. Never mix its partial results with the restarted run.

The revised reproduction module defaults to LR 0.01, batch 64, patience 20,
strictly lower validation BCE as improvement, and no fixed epoch limit.
An explicit `--epochs` is an optional cap for smoke checks only; the server
experiment does not pass it. Equal BCE counts as non-improvement. The best
checkpoint is saved on improvement, last checkpoint at termination, and
`training_status.json` records completed epochs, best epoch, stale count and
termination reason. Every seed starts from scratch and uses its own split.

The paper body specifies no learning-rate scheduler, weight decay value or
train/validation split fraction. Constant LR is used instead of retaining the
old exponential schedule. Weight decay 0.01 and original stratified 80/20 row
split remain explicit repository-derived settings. Runtime is PyTorch 2.5.1,
whereas the paper states PyTorch 2.0.0. These are disclosed implementation
choices; supplementary settings and exact published-weight equivalence have
not been verified. Model architecture, official paired training data and the
fresh ESM2 NG test encoding are retained. NG is never used for stopping.

Server restart command, after terminating only the old experiment processes:

```bash
nohup bash scripts/run_server.sh \
  > /home/gonghang/codes/BGC-Prophet-experiment-paper.log 2>&1 < /dev/null &
```

New outputs: `runs/detection_17112180_ng_esm2_paper/`.
New status: `runs/paper-experiment.status`. Existing verified LMDB and NG
embeddings are reused directly; cleaned download archives are not fetched again.
The runner refuses an existing new output directory. Original results remain
in `runs/detection_17112180_ng_esm2/`.

Local and server tests passed (four tests), covering equal/worse loss,
patience reset after improvement, stopping without an epoch cap, constant LR,
batch size, distinct best/last weights and deterministic seed/evaluation behavior.

Restart verified on 2026-10-23 around 07:55 EDT. The old runner PID 2203147 and
training wrapper/Python PIDs 4149566/4149569 were terminated with SIGTERM by user
request. Original status was marked STOPPED; all original outputs were retained.
Server synchronized training commit `a34b260b8732707fb9ac2fd93606efe8bf4500e8`
via a bundle of the already pushed local commit because proxy 7891 refused
connections. Paper runner PID 1950414 and training Python PID 1952669 started.
Full coverage checks passed again: 3,684,284 training and 67,148 test genes.
Seed 1301 config confirms `epochs=null`, patience 20, batch 64 and LR 0.01;
its first training epoch started with 407 batches on GPU 1. Remaining seeds
start sequentially after preceding training and NG testing finish. No results
from this restarted experiment were available at launch.

## Reproduction settings and commands (original run)

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

Training source is now complete official Zenodo record
<https://zenodo.org/records/17112180>, including its paired
`BGC_train_dataset_inbalence.csv`, as explicitly selected by the user.

- `lmdb_BGC.zip`: 41,451,023,708 bytes,
  MD5 `c7effa7b9b6804409803be975d287224`.
- `BGC_train_dataset_inbalence.csv`: 108,913,694 bytes,
  MD5 `fb7b8969e3a51f7380984c40539c7dd9`.

Files download only on server to
`/tmp/bgc-prophet-gonghang/official17112180/`. Runner verifies both checksums,
extracts ZIP safely, records paired CSV statistics and checks complete gene
coverage before training. No Parquet conversion or old-library substitution.

The earlier record 8104551 was verified but is incomplete for the repository
CSV: 1,471,961 background gene identifiers are absent, leaving only 438 of
16,012 complete windows. No samples were removed or zero-filled; training
stopped before any epoch. The newly supplied record supersedes this source.

Server repository is `/home/gonghang/codes/BGC-Prophet`. Server proxy is
`http://127.0.0.1:7892`; apply HTTP_PROXY and HTTPS_PROXY to dependency installation
and downloads. User chose GPU 1 despite an existing shared GPU workload.
Use the paired CSV and LMDB from record 17112180 for all five seeds.
The earlier repository CSV statistics describe the initial audit, not the final
selected training data. The paper-protocol correction above supersedes the
original run's optimization settings for new experiments.

Our independent NG test source is
`/home/gonghang/codes/my-bgc-v2/data/raw/ng/`: GenBank
`ClusterFinder_Annotated_Contigs.full.gbk` and annotations `291BGC.csv`.
Migrate copies into this repository's ignored `data/ng/raw/` directory.
`scripts/prepare_ng.py` produces protein FASTA, 128-gene windows, binary labels
and source checksums. Labels follow locus-tag membership in annotation CSV.
Annotation tags absent
from GenBank are listed in provenance; no genes or labels are fabricated.
Windows stay within contigs; final partial window uses last 128 genes.
Metrics count gene positions in windows, including overlapping tails.

NG has 67,148 CDS proteins, all with translations, producing 530 windows.
Of 4,203 unique annotation tags, 3,772 match CDS in supplied GenBank; 431 absent
tags are listed in provenance and are not included in gene-level evaluation.
Encode these proteins afresh
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

The server runner uses local verified wheels in
`/tmp/bgc-prophet-gonghang/wheels` to install pinned dependencies. Large wheels
are fetched from PyPI mirrors with published SHA256 checksums; no version changes.

Run `bash scripts/run_server.sh` in the synchronized server checkout to wait for
installation, run tests, encode NG, verify and extract official archive, check
GPU forward/backward, and train/evaluate five seeds. Launch with `nohup` and
redirect output to a persistent log. Progress is in `runs/experiment.status`.
Expected result directory is `runs/detection_17112180_ng_esm2/`.
The NG ESM2 encoding has completed: 67,148 layer-6 vectors of dimension 320.
Encoding model name, PyTorch version and model weight SHA256 are recorded in
`data/ng/esm2_encoding.json`. Official archive and paired CSV MD5 checks,
extraction, and coverage of all 3,684,284 training genes passed before training.

## Experiment results: interim snapshot, 2026-10-23

This snapshot was checked around 07:25 EDT (server time, UTC-4). The experiment
has not finished all five seeds: seeds 1301 and 1302 completed 200 epochs and NG
testing; seed 1303 completed 130 epochs during inspection and is still training;
1304 and 1305 have not started. The original process is preserved. No final
five-seed mean or standard deviation is available yet.

The training run uses commit `63fa8c1fc920a39cf50edf1e3a25112a41133f8a`,
PyTorch 2.5.1+cu124, NumPy 1.26.4, Python 3.12 and GPU 1 (RTX A6000).
The official paired CSV contains 32,510 windows: 20,000 Non, 4,070 NRP,
3,425 Polyketide, 1,660 RiPP, 1,545 Other, 825 Terpene, 685 Saccharide and
300 Alkaloid. Every sentence and label sequence has length 128.
Each seed uses 26,008 training windows and 6,502 validation windows.

The newly encoded NG test has 530 windows and 67,840 evaluated positions:
3,803 positive and 64,037 negative (5.61% positive). These include repeated
positions from overlapping contig tails; there are 67,148 unique genes.
The 3,772 unique annotated CDS become 3,803 positive window positions.
The 431 annotation tags absent from GenBank remain listed in provenance.

Best epochs below use the saved zero-based epoch index. Threshold is fixed at
0.5; average precision (AP) is scikit-learn average precision, not trapezoidal
PR-AUC. All metrics are fractions between 0 and 1.

| Seed | Epochs completed | Best epoch | Validation BCE | NG BCE | NG accuracy | NG precision | NG recall | NG F1 | NG AUROC | NG AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1301 | 200 | 50 | 0.025353 | 0.163388 | 0.948939 | 0.548415 | 0.504865 | 0.525739 | 0.898743 | 0.531919 |
| 1302 | 200 | 58 | 0.029238 | 0.142636 | 0.956088 | 0.641678 | 0.490665 | 0.556102 | 0.908042 | 0.560613 |
| 1303 | 130, running | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| 1304 | Not started | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| 1305 | Not started | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

Confusion matrices use true labels as rows, predicted labels as columns,
with class order [negative, positive].

| Seed | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| 1301 | 62,456 | 1,581 | 1,883 | 1,920 |
| 1302 | 62,995 | 1,042 | 1,937 | 1,866 |

### Interpretation and verification

At the selected checkpoints, validation F1 is 0.922053 / 0.913982 and AUROC is
0.994744 / 0.991215 for seeds 1301 / 1302. NG performance is substantially lower.
This is consistent with a gap between training-distribution validation and NG;
the current evidence does not establish its cause. Original row splitting can
share related context between training and validation. The 51,976 shared-gene
count in the initial audit applies to the old repository CSV only; it was not
remeasured for the final paired CSV and should not be quoted as its count.

Always predicting negative would already give NG accuracy 0.943942. The observed
0.948939 / 0.956088 accuracy therefore needs to be read alongside F1 and AP.
Both seeds recover about half of positive gene positions at threshold 0.5.
Seed 1302 has fewer false positives (1,042 versus 1,581) and higher precision,
but slightly lower recall. Two seeds do not establish stable seed variability.
No test-based threshold tuning or checkpoint selection was performed.

For both completed seeds, all saved NG metrics were independently recomputed
from `test_predictions.npz` and matched `test_metrics.json` within 1e-12.
The confusion matrices contain exactly 67,840 positions. Each history has 200
epochs; selected epochs match the minimum validation BCE. Saved train/validation
row-index sets are disjoint. This verifies row separation, not gene homology
separation. NG results are gene-position metrics, not BGC cluster or product
classification metrics; no claim of reproducing published cluster scores.

### Result locations and continuation

Server root: `/home/gonghang/codes/BGC-Prophet`.

- Experiment directory: `runs/detection_17112180_ng_esm2/`.
- Per seed: `config.json`, `split.json`, `epochs.jsonl`, `best.pt`, `last.pt`,
  `test_predictions.npz`, `test_metrics.json` (last/test files appear on completion).
- Completed-seed aggregate: `results.json` and `results.csv`.
- Final five-seed aggregate: `summary.json`, written only after all seeds finish.
- Runner log: `/home/gonghang/codes/BGC-Prophet-experiment17112180.log`.
- Runner phase: `runs/experiment.status`; this shows the overall training phase,
  not the current seed/epoch. Use histories to inspect actual progress.

Training and test CSV SHA256 values from the run configuration:

```text
training: 5c814849a10c7bfd7cc95d184f78ad993969d24188b29f2d8d6d2fe2412b5c51
NG test:  6f5fccbf2b095667f637598b8acf27d9ef76aebdd9f3e8e80b77aaff724a7e62
```

Update this snapshot and compute mean/sample SD over all five completed NG
results after 1303-1305 finish. Do not restart or overwrite the active run.

## Cleanup after the interim result audit

Cleanup was requested while seed 1303 was still active. The deletion command
is restricted to the already verified and extracted 17112180 ZIP, obsolete
8104551 archive/extraction, abandoned Parquet-to-LMDB output, downloaded wheel
files, transfer bundles, failed clone and speed-test files created in this task.
Training LMDB/CSV, NG raw data and fresh embeddings, virtual environment,
checkpoints, saved predictions and logs are preserved.

The first cleanup SSH call timed out, but a subsequent explicit path check
confirmed removal of the ZIP, obsolete data, abandoned conversion output,
wheels, bundles and failed clone. These targets occupied approximately 64 GiB
before cleanup; this estimate is from target sizes, not a filesystem-wide delta.
The active official LMDB remains present, training PID 4149569 remains running,
and `/tmp` filesystem has 666 GiB available after cleanup.

The ZIP is reproducible from the recorded official URL and MD5. Rerunning the
original server runner after removing it will download it again. Continue the
existing run instead of rerunning that script.

Verdict: original training and metric scripts require corrections before their
numbers are reliable. New reproduction bypasses those failures while preserving
architecture, optimizer, learning-rate schedule and original row split.

Not checked: original published weight equivalence, source-sequence homology,
and final five-seed metrics. Large datasets were not downloaded locally.
No unrelated inference or classification redesign.
