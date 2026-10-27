"""Read-only audit of the fixed baseline; no training, tuning or file changes to runs."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score(target, probability):
    target, probability = target.ravel(), probability.ravel()
    prediction = probability >= 0.5
    return dict(accuracy=float(accuracy_score(target, prediction)),
                precision=float(precision_score(target, prediction, zero_division=0)),
                recall=float(recall_score(target, prediction, zero_division=0)),
                f1=float(f1_score(target, prediction, zero_division=0)),
                auroc=float(roc_auc_score(target, probability)),
                average_precision=float(average_precision_score(target, probability)),
                confusion_matrix=confusion_matrix(target, prediction, labels=[0, 1]).tolist())


def audit(root, verify_inference=False):
    summary_path = root / "runs/legacy_remaining_parallel/summary.json"
    summary = json.loads(summary_path.read_text())
    require(summary["seeds"] == list(range(1301, 1306)), "Expected exactly five seeds")
    folders = {int(seed): root / path for seed, path in summary["seed_directories"].items()}
    require(sorted(folders) == list(range(1301, 1306)), "Missing seed directories")
    report = dict(audit_time_utc=datetime.now(timezone.utc).isoformat(),
                  audit_script_sha256=sha256(Path(__file__)),
                  server_repository=str(root), threshold=0.5, aggregation="mean and sample SD, ddof=1",
                  metric_unit="gene positions in 128-gene windows, overlapping tails retained",
                  inference_recheck_requested=verify_inference, seeds=[], source_files={},
                  limitations=["Repository-derived optimization settings differ from paper main text",
                               "Window-level random split can share genes/source BGCs across train and validation",
                               "No sequence-homology isolation demonstrated",
                               "Custom NG preprocessing differs from original tracked NG CSV",
                               "Archive checksum was verified at download; archive later deleted"])
    expected_targets = None
    reference_data = None
    training_rows = None
    test_genes = None
    target_hash = None
    for seed, folder in sorted(folders.items()):
        config = json.loads((folder / "config.json").read_text())
        status = json.loads((folder / "training_status.json").read_text())
        saved = json.loads((folder / "test_metrics.json").read_text())
        split = json.loads((folder / "split.json").read_text())
        history = [json.loads(line) for line in (folder / "epochs.jsonl").read_text().splitlines()]
        required_config = dict(seed=seed, epochs=200, batch_size=32, learning_rate=0.0005,
                               patience=0, scheduler="original", max_len=128, nhead=5,
                               encoder_layers=2, transformer_dropout=0.1, mlp_dropout=0.1,
                               validation_ratio=0.2, weight_decay=0.01, threshold=0.5,
                               evaluation_split="test", selection="minimum validation BCE")
        for key, value in required_config.items():
            require(config.get(key) == value, f"Seed {seed}: config {key}")
        training_csv = Path(config["dataset"])
        test_csv = root / config["test_dataset"]
        actual_data = (sha256(training_csv), sha256(test_csv))
        require(actual_data == (config["dataset_sha256"], config["test_dataset_sha256"]), "CSV changed")
        if reference_data is None:
            reference_data = actual_data
            with training_csv.open() as handle:
                training_rows = [row["TDsentence"].split() for row in csv.DictReader(handle)]
            with test_csv.open() as handle:
                test_rows = list(csv.DictReader(handle))
            expected_targets = np.array([[int(x) for x in row["TDlabels"].split()] for row in test_rows], dtype=np.float32)
            test_genes = {gene for row in test_rows for gene in row["TDsentence"].split()}
            target_hash = hashlib.sha256(expected_targets.tobytes()).hexdigest()
            report["dataset"] = dict(training_rows=len(training_rows), test_windows=len(test_rows),
                                     test_unique_genes=len(test_genes), test_positions=int(expected_targets.size),
                                     positive_positions=int(expected_targets.sum()),
                                     negative_positions=int(expected_targets.size - expected_targets.sum()),
                                     targets_float32_sha256=target_hash)
            report["source_files"][str(training_csv)] = actual_data[0]
            report["source_files"][str(test_csv)] = actual_data[1]
        require(actual_data == reference_data, "Seeds used different CSVs")
        require(len(history) == 200 and [row["epoch"] for row in history] == list(range(200)), "History incomplete")
        require(status["completed_epochs"] == 200 and status["stop_reason"] == "epoch_cap", "Run incomplete")
        lr = 0.0005
        for row in history:
            require(abs(row["lr"] - lr) < 1e-14, "LR schedule mismatch")
            if row["epoch"] >= 5 and lr > 5e-6:
                lr *= 0.95
        best = min(history, key=lambda row: row["validation"]["bce"])
        require(best["epoch"] == saved["best_epoch"] == status["best_epoch"], "Wrong selected epoch")
        require(abs(best["validation"]["bce"] - saved["validation_bce"]) < 1e-12, "Wrong validation loss")
        train_indices, val_indices = set(split["train"]), set(split["validation"])
        require(len(train_indices) == len(split["train"]) and len(val_indices) == len(split["validation"]), "Duplicate split rows")
        require(not train_indices & val_indices and train_indices | val_indices == set(range(len(training_rows))), "Invalid split")
        train_genes = {gene for index in train_indices for gene in training_rows[index]}
        val_genes = {gene for index in val_indices for gene in training_rows[index]}
        source_clusters = lambda genes: {gene.split("_protein_", 1)[0] for gene in genes if gene.startswith("BGC") and "_protein_" in gene}
        split_audit = dict(train_rows=len(train_indices), validation_rows=len(val_indices),
                           shared_train_validation_gene_ids=len(train_genes & val_genes),
                           shared_source_BGC_ids=len(source_clusters(train_genes) & source_clusters(val_genes)),
                           shared_training_test_gene_ids=len((train_genes | val_genes) & test_genes))
        del train_genes, val_genes
        with np.load(folder / "test_predictions.npz", allow_pickle=False) as arrays:
            target, probability = arrays["target"], arrays["probability"]
        require(np.array_equal(target, expected_targets), "Saved labels do not match NG CSV order")
        require(probability.shape == target.shape and np.isfinite(probability).all(), "Invalid probabilities")
        require(((probability >= 0) & (probability <= 1)).all(), "Probabilities outside [0,1]")
        calculated = score(target, probability)
        for key, value in calculated.items():
            require(value == saved[key] if isinstance(value, list) else abs(value - saved[key]) < 1e-12, f"Metric mismatch: {key}")
        # Batch BCE uses float32 reductions; allow reduction-order rounding on CPU.
        import torch
        bce = float(torch.nn.functional.binary_cross_entropy(torch.from_numpy(probability), torch.from_numpy(target)))
        require(abs(bce - saved["bce"]) < 1e-6, "BCE mismatch")
        files = {}
        for name in ("config.json", "split.json", "epochs.jsonl", "training_status.json", "best.pt", "last.pt", "test_metrics.json", "test_predictions.npz"):
            path = folder / name
            files[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))
        inference = None
        if verify_inference:
            from torch.utils.data import DataLoader
            from bgc_prophet.train.data import DataReader, BGCLabelsDataset
            from bgc_prophet.train.model import transformerEncoderNet
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = transformerEncoderNet(d_model=320, nhead=5, num_encoder_layers=2, dim_feedforward=1280,
                                          transformer_dropout=0.1, mlp_dropout=0.1).to(device)
            model.load_state_dict(torch.load(folder / "best.pt", map_location=device, weights_only=True))
            model.eval()
            dataset = BGCLabelsDataset(DataReader(test_csv, test_ratio=0), str(root / config["test_lmdb"]), "eval")
            predictions = []
            with torch.no_grad():
                for sentence, _, _ in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
                    predictions.append(model(sentence.to(device)).cpu().numpy())
            fresh = np.concatenate(predictions)
            delta = float(np.max(np.abs(fresh - probability)))
            require(np.allclose(fresh, probability, atol=1e-5, rtol=1e-4), "Checkpoint inference differs")
            require(np.array_equal(fresh >= 0.5, probability >= 0.5), "Checkpoint classifications differ")
            inference = dict(device=device, max_abs_probability_difference=delta, threshold_predictions_identical=True)
            del dataset, model
        report["seeds"].append(dict(seed=seed, run_directory=str(folder), config=config, stopping=status,
                                     selected_validation=best["validation"], test_metrics=saved,
                                     split_audit=split_audit, artifact_manifest=files,
                                     checkpoint_inference=inference))
        print(f"Verified seed {seed}; split audit {split_audit}", flush=True)
    recalculated = {}
    for key, value in summary["metrics"].items():
        values = [row["test_metrics"][key] for row in report["seeds"]]
        computed = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values))
        require(all(abs(computed[k] - value[k]) < 1e-12 for k in computed), f"Aggregate mismatch {key}")
        recalculated[key] = computed
    report["summary"] = recalculated
    for relative in ("data/ng/prepared/provenance.json", "data/ng/esm2_encoding.json"):
        path = root / relative
        report[relative] = json.loads(path.read_text())
        report["source_files"][str(path)] = sha256(path)
    report["audit_passed"] = True
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-inference", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output already exists; preserve previous evidence")
    report = audit(args.root.resolve(), args.verify_inference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("BASELINE_EVIDENCE_AUDIT_PASSED", flush=True)
