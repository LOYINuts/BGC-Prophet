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

## Authorized convergence investigation and usable baseline

The user requested an experimental investigation and authorized returning to
the old configuration if the abnormal metrics cannot be resolved. The goal is
reliable, measured results; no score is guaranteed and no test labels, threshold
or metric definitions are changed.

`scripts/investigate_training.sh` runs on GPU 1 in two sequential phases:

1. Validation-only diagnostic, seed 1301, LR 0.0005, batch 64, no scheduler,
   patience 20 and a 12-epoch cap. Compare its first 12 epochs against the
   existing LR 0.01 seed 1301 history. Same data hash, split and architecture;
   batch is fixed at 64. This isolates LR over the common epoch budget.
   It does not use NG predictions to choose hyperparameters. Diagnostic report:
   `runs/lr-diagnostic.json`; pilot directory:
   `runs/diagnostic_lr0005_batch64_seed1301/`.
2. Complete five-seed legacy baseline in a separate directory,
   `runs/detection_17112180_ng_esm2_legacy_complete/`: LR 0.0005, batch 32,
   AdamW weight decay 0.01, original exponential schedule gamma 0.95 after
   epoch >=5 while LR >5e-6, 200 epochs, early stopping disabled. All seeds
   1301-1305 train from scratch. Selection remains minimum validation BCE,
   followed by independent NG evaluation at threshold 0.5. This restores the
   previously successful protocol, not a claim of matching paper parameters.

This baseline is authorized regardless of whether the short pilot completely
identifies the cause, so producing the requested usable-model comparison does
not depend on another user confirmation. If low LR improves validation while
all other common-epoch settings match, that establishes LR sensitivity in this
implementation; it does not establish an error in the paper or independently
validate its unspecified scheduler. If it fails, inspect input/gradient and
implementation differences before claiming a diagnosis.

Reproduction CLI now exposes `--scheduler none|original` and `--patience 0`
for the explicitly capped legacy experiment. Paper defaults remain unchanged.
`--patience 0` without an epoch cap is rejected. Tests cover legacy epoch caps,
disabled early stopping and exact onset of LR decay, in addition to paper
patience behavior. Status is `runs/investigation.status`; all previous results
remain intact. No unrelated GPU jobs are stopped.

Launch verified on server 2026-10-24 at 06:07:52 EDT using committed and pushed
revision `18c01ff`. Runner PID 1173992 and diagnostic Python PID 1174121 were
observed. Status reported `Validation-only LR diagnostic`. Persistent combined
log: `/home/gonghang/codes/BGC-Prophet-investigation.log`.
The first diagnostic results and final legacy five-seed results are pending;
do not claim the suspected LR effect has been experimentally established yet.

## Completed comparison baseline, 2026-10-27

All five legacy seeds completed 200 epochs and independent NG testing. The
three-GPU aggregate in `runs/legacy_remaining_parallel/summary.json` combines
preserved seeds 1301/1302 with restarted seeds 1303/1304/1305; no seed is counted
twice. Model selection is minimum validation BCE and test threshold is 0.5.

| Seed | Best epoch (zero-based) | Validation BCE | NG precision | NG recall | NG F1 | NG AUROC | NG AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1301 | 50 | 0.025353 | 0.548415 | 0.504865 | 0.525739 | 0.898743 | 0.531919 |
| 1302 | 58 | 0.029238 | 0.641678 | 0.490665 | 0.556102 | 0.908042 | 0.560613 |
| 1303 | 63 | 0.023945 | 0.631776 | 0.533263 | 0.578354 | 0.900948 | 0.588273 |
| 1304 | 70 | 0.028835 | 0.670500 | 0.444649 | 0.534704 | 0.892890 | 0.545343 |
| 1305 | 73 | 0.023842 | 0.628876 | 0.495924 | 0.554543 | 0.899459 | 0.572149 |

| NG metric | Mean | Sample SD (ddof=1) |
| --- | --- | --- |
| BCE | 0.157216 | 0.008820 |
| Accuracy | 0.954679 | 0.003246 |
| Precision | 0.624249 | 0.045475 |
| Recall | 0.493873 | 0.032054 |
| F1 | 0.549888 | 0.020520 |
| AUROC | 0.900016 | 0.005431 |
| AP | 0.559659 | 0.022080 |

This is a stable, useful baseline across all five seeds, with F1 0.526-0.578.
It substantially outperforms the failed main-text-parameter run (mean F1
0.009185, AUROC 0.621225). The controlled LR diagnostic supports LR sensitivity
as a cause in this implementation. It does not prove exact equivalence to the
published training procedure. Recall remains about one half of annotated gene
positions; accuracy is only about 1.1 percentage points above all-negative
accuracy, so F1, AP and recall are more informative.

The server audit independently recomputed all five NG prediction-array metrics,
checked 200 history epochs, minimum-BCE selection and aggregate mean/sample SD.
It completed with `ALL_BASELINE_RESULTS_VERIFIED`. The aggregate also checks
matching data hashes and training protocols. All completed baseline results
are retained as the comparison baseline, without loss reweighting or retuning.

### Baseline scope clarified by the user

The purpose of this model is comparison against the user's own model using
five-seed mean and sample SD. It must remain a fixed baseline; no further model
optimization or hyperparameter search is wanted. A positive-loss-weight pilot
was prepared after the prior wording was interpreted too broadly, but it was
never launched. Its unused code was removed, restoring the training module
and tests to their baseline versions. The transferred bundle is disposable;
no optimized checkpoints or metrics form part of the baseline.

Final comparison metrics are the five-seed NG results above. Use identical NG
annotations, gene positions, thresholds and aggregation units when comparing
with the user's model. AP is average precision; do not label it trapezoidal
PR-AUC. These settings are the disclosed repository-derived baseline rather
than the failed main-text-parameter attempt. No claim about the user's model's
relative performance can be made without its matching results.

## Three-GPU launch correction after successful slow SSH login

Windows OpenSSH (`C:/Windows/System32/OpenSSH/ssh.exe`) completed public-key
login using the same local ED25519 key after a long delay; Git's bundled SSH
had rejected or timed out on prior attempts. Do not infer password requirements
from those failures. Use Windows OpenSSH and background log capture for slow
sessions in this environment.

The fresh status check found that seed 1302 had completed while waiting for
access. Seeds 1301 and 1302 both have 200 epochs and saved NG metrics. Seed
1303 had 145 completed epochs, and 1304/1305 had not started. User authorization
to stop current training therefore applies to partial 1303; completed 1302
must not be retrained. Launcher and aggregate paths were updated accordingly:
GPU 0 runs 1303, GPU 1 runs 1304, GPU 2 runs 1305. Aggregate preserves both
completed seeds from the original complete-baseline directory. Partial 1303
stays in that directory and new runs use isolated directories.

Stop and launch verified: old runner and training PIDs no longer appeared after
SIGTERM. Server synchronized commit `7b79572`; parallel runner PID 1449295
launched three Python workers: 1449391 (1303), 1449392 (1304), 1449393 (1305).
Commands confirm LR0.0005, batch32, original scheduler, no early stopping and
200 epochs, each in its own new directory. Subsequent logs confirm all three started training batches, and process
environments confirm `CUDA_VISIBLE_DEVICES=0/1/2` for 1303/1304/1305 respectively.
Preserved complete seeds 1301/1302 will be combined only after workers finish.

## Requested three-GPU restart of unfinished seeds

User explicitly authorized terminating the active sequential legacy experiment
and retraining only unfinished seeds 1302-1305 from scratch. Completed seed 1301
must be preserved and used in the final aggregate. Partial seed 1302 outputs
remain archived in the old experiment directory; new outputs never overwrite
those files. Before termination, verify actual current processes and that no
additional seeds have finished since the last check.

`scripts/run_remaining_parallel.sh` assigns GPU 0 to 1302 then 1305, GPU 1 to
1303, GPU 2 to 1304. Every worker uses the same 200-epoch legacy protocol.
It refuses existing new outputs, any already-completed assigned seed, or an
existing reproduction process. Each seed writes to an isolated output root,
preventing concurrent result-file races. New outputs and per-seed logs are in
`runs/legacy_remaining_parallel/`.
`scripts/summarize_parallel.py` combines preserved 1301 with the four completed
workers only after checking data hashes, architecture, LR, batch, scheduler,
patience and 200-epoch completion. Synthetic tests verify aggregation and
rejection of mixed configurations. Five local tests passed.

SSH attempts during preparation repeatedly timed out. No termination or
three-GPU start has been verified yet. The sequential run may still be active;
check before launching. Never stop unrelated GPU processes. Once connected,
terminate the specific verified old runner and Python process, preserve results,
synchronize the pushed commit and launch the parallel script under nohup.

## Investigation update: 2026-10-25

Server was reachable at 03:22 EDT (UTC-4). The 12-epoch validation-only pilot
finished on 2026-10-24 at 07:06:47 EDT; the full legacy five-seed run then
started automatically. It has not finished: seed 1301 completed 200 epochs and
NG testing; seed 1302 completed 14 epochs during inspection; 1303-1305 have not
started. Training Python PID 1602604 remains active. No new run was launched
by this status audit.

### Controlled LR diagnostic result

Both settings use seed 1301, identical split and training data, batch 64,
constant LR, the same architecture and optimizer, and the first 12 epochs.
Checkpoints in this comparison are selected using validation BCE only; NG was
not evaluated for the pilot. Best epoch is zero-based.

| LR | Best epoch within 12 | Train BCE | Validation BCE | Validation precision | Validation recall | Validation F1 | Validation AUROC | Validation AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 3 | 0.171158 | 0.159363 | 1.000000 | 0.000073 | 0.000147 | 0.810114 | 0.312890 |
| 0.0005 | 11 | 0.040514 | 0.035285 | 0.910228 | 0.866080 | 0.887606 | 0.991542 | 0.941259 |

Reducing LR by 20x restores useful learning under the same batch size and
without adding a scheduler. Validation BCE decreases by about 77.9%, and
validation F1 rises from effectively zero to 0.8876. This controlled experiment
supports excessive LR as a major cause of the failed convergence in the current
implementation. It is not proof that the paper is incorrect or that all other
implementation choices are equivalent. One seed and 12 epochs do not quantify
robustness across seeds. No test-based threshold tuning was performed.

### Full normal-configuration run in progress

Output: `runs/detection_17112180_ng_esm2_legacy_complete/`.
Seed 1301 selected epoch 50 after 200 epochs. All saved test metrics are exactly
equal to the earlier seed 1301 run, including BCE, AUROC, AP and confusion
matrix. This is a repeated-run metric check, not a new independent seed.

| Seed | Completed epochs at inspection | NG precision | NG recall | NG F1 | NG AUROC | NG AP |
| --- | --- | --- | --- | --- | --- | --- |
| 1301 | 200, tested | 0.548415 | 0.504865 | 0.525739 | 0.898743 | 0.531919 |
| 1302 | 14, running | Pending | Pending | Pending | Pending | Pending |
| 1303-1305 | Not started | Pending | Pending | Pending | Pending | Pending |

Recommendation: let the already authorized full legacy baseline finish. The
LR diagnostic has identified a credible, experimentally supported convergence
factor and the repeated baseline has recovered its earlier metrics; no extra
hyperparameter sweep is needed now. Assess final mean/sample SD after all five
seeds finish. Keep the main-text parameter attempt as a disclosed unsuccessful
reproduction, and label the usable baseline with its actual repository-derived
LR, batch size and schedule. Exact published-score equivalence remains unproven.

The 14 MiB `/tmp/bgc-prophet-gonghang/investigation.bundle` was deleted and its
absence verified. Complete official LMDB presence was checked afterward.
Active training, environment, all model/prediction/history outputs, diagnostic
report and small audit/provenance files were retained. Previously cleaned large
archives were not downloaded again. No additional large disposable files were
identified in the bounded temporary-directory inspection.

## Final paper-parameter experiment results

Verified snapshot: server 2026-10-24. The paper-parameter runner completed on
2026-10-23 at 20:28:53 EDT (UTC-4). All five seeds have test results and
`training_status.json`; `summary.json` includes exactly seeds 1301-1305.
These results supersede the interim status for the restarted experiment, not
the separately preserved old-configuration results below.

Training commit: `a34b260b8732707fb9ac2fd93606efe8bf4500e8`.
Data, model, ESM2 encoding and 80/20 per-seed splits are described above.
This run uses LR 0.01, batch 64, AdamW weight decay 0.01, constant LR and
validation-BCE patience 20, with no epoch cap. It implements the main-text
training parameters; it does not establish equivalence to the authors' original
training environment or undocumented supplementary choices.

### Five-seed NG metrics and early stopping

All metrics are fractions. AP denotes average precision. Best epoch is
zero-based; completed epochs is a count. Every seed terminated through early
stopping with 20 consecutive non-improving validation epochs.

| Seed | Epochs | Best epoch | Best val BCE | NG BCE | Accuracy | Precision | Recall | F1 | AUROC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1301 | 24 | 3 | 0.159363 | 0.247951 | 0.943942 | 0.000000 | 0.000000 | 0.000000 | 0.617024 | 0.077485 |
| 1302 | 26 | 5 | 0.153642 | 0.361307 | 0.943942 | 0.000000 | 0.000000 | 0.000000 | 0.648324 | 0.093208 |
| 1303 | 48 | 27 | 0.154856 | 0.225576 | 0.939814 | 0.106742 | 0.009992 | 0.018274 | 0.593625 | 0.072631 |
| 1304 | 23 | 2 | 0.165866 | 0.262513 | 0.934685 | 0.083554 | 0.016566 | 0.027650 | 0.616894 | 0.081086 |
| 1305 | 24 | 3 | 0.156426 | 0.315914 | 0.943942 | 0.000000 | 0.000000 | 0.000000 | 0.630256 | 0.086308 |

| NG metric | Five-seed mean | Sample SD (ddof=1) |
| --- | --- | --- |
| BCE | 0.282652 | 0.055137 |
| Accuracy | 0.941265 | 0.004090 |
| Precision | 0.038059 | 0.052756 |
| Recall | 0.005312 | 0.007636 |
| F1 | 0.009185 | 0.013006 |
| AUROC | 0.621225 | 0.020088 |
| AP | 0.082143 | 0.007954 |

| Seed | TN | FP | FN | TP | Predicted positive positions |
| --- | --- | --- | --- | --- | --- |
| 1301 | 64,037 | 0 | 3,803 | 0 | 0 |
| 1302 | 64,037 | 0 | 3,803 | 0 | 0 |
| 1303 | 63,719 | 318 | 3,765 | 38 | 356 |
| 1304 | 63,346 | 691 | 3,740 | 63 | 754 |
| 1305 | 64,037 | 0 | 3,803 | 0 | 0 |

### What these results establish

Three of five seeds predict every test position as negative at threshold 0.5.
The other two recover only 38 and 63 of 3,803 positive positions. The aggregate
accuracy is below the all-negative baseline (0.943942), and AP is only modestly
above positive prevalence (0.056058). AUROC around 0.62 also indicates poor
ranking; this is not explained solely by a badly chosen 0.5 threshold.
The run does not reproduce the paper's reported NG AUROC of 0.919 (main text,
page 8) or precision of 0.592 at threshold 0.5. Differences in NG preprocessing
and runtime still prevent an exact like-for-like published-score comparison.

The previous repository-configuration run had NG F1 0.525739/0.556102 and AUROC
0.898743/0.908042 for seeds 1301/1302. The restarted run's minimum validation
BCE is 0.153642-0.165866 versus 0.025353/0.029238 previously. Thus deterioration
is visible in validation loss before NG evaluation. Several settings changed
together: LR, batch size, scheduler and stopping rule. The evidence does not
isolate the cause, and does not show that early stopping itself is wrong.
A numerically completed experiment is not evidence that the model learned a
useful detector under these settings.

### Verification and outstanding checks

Saved results cover five unique expected seeds. All status files report
`stop_reason=early_stopping`, patience 20 and stale count 20. For each seed,
`completed_epochs - 1 - best_epoch = 20`, consistent with the implementation.
All confusion matrices total 67,840 positions and contain 3,803 positive labels.
Accuracy, precision, recall and F1 are independently consistent with the saved
counts; five-seed means and sample SD recomputed locally match `summary.json`.

All five NG prediction arrays were independently audited using NumPy and
scikit-learn; accuracy, precision, recall, F1, AUROC, AP and confusion matrices
match saved metrics within 1e-12. Predictions are finite and contain the
expected 67,840 positions and 3,803 positive labels. Best epochs match the
minimum recorded validation BCE. Audit ended with
`ALL_FIVE_ARRAY_METRICS_VERIFIED`.

| Seed | Train BCE at selected epoch | Validation F1 | NG median probability | NG maximum probability |
| --- | --- | --- | --- | --- |
| 1301 | 0.171158 | 0.000147 | 0.119275 | 0.383465 |
| 1302 | 0.178536 | 0.000000 | 0.258738 | 0.453145 |
| 1303 | 0.169817 | 0.287952 | 0.072205 | 0.728597 |
| 1304 | 0.182950 | 0.041534 | 0.078505 | 0.695119 |
| 1305 | 0.177997 | 0.000000 | 0.201618 | 0.451402 |

Weak validation F1 and elevated train BCE show that NG mismatch alone cannot
explain this outcome. Seeds 1301/1302/1305 have no NG score reaching 0.5;
this accounts for their all-negative confusion matrices. It does not establish
why optimization failed, nor justify choosing a threshold on the NG test set.

### Further experiments: small diagnostics first

Additional investigation is warranted because the detector is unusable at the
fixed evaluation threshold. Repeating all five seeds without changing the
hypothesis would add little evidence. Recommended next steps, not yet launched:

1. Inspect per-epoch train/validation BCE and positive score distributions in
   more detail; the completed prediction-array audit above already rules out
   incorrect saved NG metric calculations. Check representative official
   training embeddings against freshly encoded proteins where source sequences
   are available. These are read-only diagnostics.
2. Run one controlled pilot at seed 1301: LR 0.0005, batch 64, patience 20,
   constant LR, same split/data/model. Compare validation convergence with the
   completed LR 0.01 run. Only LR changes, isolating its effect. Treat this as
   a diagnostic protocol, not the paper setting.
3. If the pilot converges, investigate the authors' exact optimizer/scheduler
   and supplementary configuration. Use training/validation evidence for
   decisions; do not select LR or threshold by best NG test score.
4. For a separate generalization benchmark, split source BGC/genome groups
   before augmented windows to reduce related-context leakage. Keep this
   distinct from the original paper-protocol comparison.

Run another full five-seed comparison only after a concrete diagnosis and a
fixed protocol. No additional GPU experiments were started by this result audit.

### Artifacts and cleanup

Final output: `runs/detection_17112180_ng_esm2_paper/` under the server repository.
Each seed retains best/last checkpoints, history, split, config, stopping status,
test metrics and prediction arrays. `results.csv`, `results.json` and
`summary.json` provide final per-seed and aggregate results. Runner log:
`/home/gonghang/codes/BGC-Prophet-experiment-paper.log`.

The previous audit already removed approximately 64 GiB of redundant downloads
and abandoned preparation outputs. This audit identified only the remaining
14 MiB `/tmp/bgc-prophet-gonghang/paper-restart.bundle` and small wheel download
lists as transfer leftovers. These three files were deleted and their absence
explicitly checked over SSH. The final experiment summary and active official
LMDB were confirmed present afterward. Retained: complete extracted official
LMDB and CSV, raw NG and ESM2 embeddings, environment, both experiment
directories, provenance and all result logs. No further large disposable
archives were identified.

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

Not checked: original published weight equivalence and source-sequence homology.
All five final NG metric arrays were independently verified. Large datasets
were not downloaded locally. No unrelated inference redesign.
