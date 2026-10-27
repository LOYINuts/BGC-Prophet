"""Retrain the detection model; select by validation BCE, then test once."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import subprocess

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DataReader, BGCLabelsDataset
from .model import transformerEncoderNet


def metrics(target, probability):
    target = np.asarray(target).reshape(-1).astype(int)
    probability = np.asarray(probability).reshape(-1)
    prediction = probability >= 0.5
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "auroc": float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else None,
        "average_precision": float(average_precision_score(target, probability)) if target.any() else None,
        "confusion_matrix": confusion_matrix(target, prediction, labels=[0, 1]).tolist(),
    }


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detection_loss(probability, target, positive_weight=1.0):
    loss = torch.nn.functional.binary_cross_entropy(probability, target, reduction="none")
    return (loss * torch.where(target == 1, positive_weight, 1.0)).mean()


def run_epoch(model, loader, device, optimizer=None, positive_weight=1.0):
    model.train(optimizer is not None)
    loss_sum, count = 0.0, 0
    targets, probabilities = [], []
    with torch.set_grad_enabled(optimizer is not None):
        for sentence, _, target in tqdm(loader, desc="train" if optimizer else "evaluate"):
            sentence, target = sentence.to(device), target.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            probability = model(sentence)
            bce = torch.nn.functional.binary_cross_entropy(probability, target)
            loss = detection_loss(probability, target, positive_weight) if optimizer is not None else bce
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            loss_sum += bce.item() * target.numel()
            count += target.numel()
            targets.append(target.detach().cpu().numpy())
            probabilities.append(probability.detach().cpu().numpy())
    if not count:
        raise ValueError("Empty dataset split")
    target, probability = np.concatenate(targets), np.concatenate(probabilities)
    return {"bce": loss_sum / count, **metrics(target, probability)}, target, probability


def check_embeddings(reader, dataset, max_len):
    if reader.df.empty:
        raise ValueError("Dataset is empty")
    keys = set()
    for sentence, target in zip(reader.df.TDsentence, reader.df.TDlabels):
        if len(sentence) != max_len or len(target) != max_len or not set(target) <= {0, 1}:
            raise ValueError("Expected fixed-length sentences with binary gene labels")
        keys.update(sentence)
    missing = [key for key in sorted(keys) if dataset.txn.get(key.encode("ascii")) is None]
    if missing:
        raise ValueError(f"LMDB missing {len(missing)} genes; first: {missing[:5]}")
    sentence, _, _ = dataset[0]
    if sentence.shape != (max_len, 320) or not torch.isfinite(sentence).all():
        raise ValueError("Expected finite ESM2 layer-6 embeddings of shape (max_len, 320)")
    return len(keys)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/BGC_train_dataset.csv"))
    parser.add_argument("--lmdb", type=Path, required=True)
    parser.add_argument("--test-dataset", type=Path, default=Path("data/Nine_genomes_test_dataset.csv"))
    parser.add_argument("--test-lmdb", type=Path, help="Omit to train and report validation only")
    parser.add_argument("--output", type=Path, default=Path("runs/detection"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1301, 1302, 1303, 1304, 1305])
    parser.add_argument("--epochs", type=int, help="Optional epoch cap; default trains until early stopping")
    parser.add_argument("--patience", type=int, default=20, help="0 disables early stopping; requires --epochs")
    parser.add_argument("--scheduler", choices=["none", "original"], default="none")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--positive-weight", type=float, default=1.0,
                        help="Training BCE positive weight; validation BCE stays unweighted")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.positive_weight) or args.positive_weight <= 0:
        parser.error("--positive-weight must be finite and positive")
    if (args.epochs is not None and args.epochs < 1) or args.patience < 0 or (args.patience == 0 and args.epochs is None) or args.batch_size < 1 or args.workers != 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive epoch cap/patience/batch size, unique seeds and --workers 0 required")
    for path in ((args.dataset, args.test_dataset) if args.test_lmdb else (args.dataset,)):
        if not path.is_file():
            parser.error(f"Missing CSV: {path}")
    for path in ((args.lmdb, args.test_lmdb) if args.test_lmdb else (args.lmdb,)):
        if not (path / "data.mdb").is_file():
            parser.error(f"Missing LMDB: {path}/data.mdb")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # shortcut: use one LMDB process; add per-worker transactions if loading becomes a bottleneck.
    seed_all(args.seeds[0])
    data = DataReader(args.dataset, test_ratio=0.2)
    training = BGCLabelsDataset(data, str(args.lmdb), "train")
    validation = BGCLabelsDataset(data, str(args.lmdb), "test", env=training.env)
    test = None
    if args.test_lmdb:
        test_data = DataReader(args.test_dataset, test_ratio=0)
        test_env = training.env if args.lmdb.resolve() == args.test_lmdb.resolve() else None
        test = BGCLabelsDataset(test_data, str(args.test_lmdb), "eval", env=test_env)
        print("Test genes:", check_embeddings(test_data, test, args.max_len), flush=True)
    print("Training genes:", check_embeddings(data, training, args.max_len), flush=True)
    if args.check_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / f"seed_{seed}").exists() for seed in args.seeds):
        raise FileExistsError("Seed output exists; choose a new --output to preserve previous results")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(git_commit=revision, torch_version=torch.__version__, numpy_version=np.__version__,
                  device_name=torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                  dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                  test_dataset_sha256=hashlib.sha256(args.test_dataset.read_bytes()).hexdigest() if test else None,
                  evaluation_split="test" if test else "validation",
                  validation_ratio=0.2, nhead=5, encoder_layers=2, transformer_dropout=0.1,
                  mlp_dropout=0.1, threshold=0.5, selection="minimum validation BCE",
                  scheduler_protocol="constant LR" if args.scheduler == "none" else "gamma 0.95 after epoch >=5 while LR >5e-6",
                  weight_decay=0.01,
                  stopping="patience consecutive epochs without strictly lower validation BCE",
                  split_policy="per-seed stratified rows, matching original DataReader")
    results = []
    for seed in args.seeds:
        seed_all(seed)
        data = DataReader(args.dataset, test_ratio=0.2)
        training.data = validation.data = data
        folder = args.output / f"seed_{seed}"
        folder.mkdir()
        write_json(folder / "config.json", {**config, "seed": seed})
        write_json(folder / "split.json", {"train": data.train, "validation": data.test})
        loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=index == 0,
                              num_workers=0) for index, dataset in enumerate((training, validation, test) if test else (training, validation))]
        model = transformerEncoderNet(d_model=320, nhead=5, num_encoder_layers=2,
                                      max_len=args.max_len, dim_feedforward=1280,
                                      transformer_dropout=0.1, mlp_dropout=0.1).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95) if args.scheduler == "original" else None
        best_loss, best_epoch, stale_epochs = float("inf"), None, 0
        stop_reason = None
        with (folder / "epochs.jsonl").open("w", encoding="utf-8") as history:
            for epoch in itertools.count():
                train_metrics, _, _ = run_epoch(model, loaders[0], args.device, optimizer,
                                                 positive_weight=args.positive_weight)
                validation_metrics, _, _ = run_epoch(model, loaders[1], args.device)
                if not np.isfinite(validation_metrics["bce"]):
                    raise ValueError("Nonfinite validation BCE")
                row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                       "train": train_metrics, "validation": validation_metrics}
                history.write(json.dumps(row, allow_nan=False) + "\n")
                history.flush()
                print(json.dumps({"seed": seed, **row}), flush=True)
                if validation_metrics["bce"] < best_loss:
                    best_loss, best_epoch, stale_epochs = validation_metrics["bce"], epoch, 0
                    torch.save(model.state_dict(), folder / "best.pt")
                else:
                    stale_epochs += 1
                if scheduler is not None and epoch >= 5 and scheduler.get_last_lr()[0] > 5e-6:
                    scheduler.step()
                if args.patience and stale_epochs >= args.patience:
                    stop_reason = "early_stopping"
                    break
                if args.epochs is not None and epoch + 1 >= args.epochs:
                    stop_reason = "epoch_cap"
                    break
        write_json(folder / "training_status.json", {
            "completed_epochs": epoch + 1, "best_epoch": best_epoch,
            "best_validation_bce": best_loss, "stale_epochs": stale_epochs,
            "patience": args.patience, "stop_reason": stop_reason,
        })
        torch.save(model.state_dict(), folder / "last.pt")
        model.load_state_dict(torch.load(folder / "best.pt", map_location=args.device, weights_only=True))
        evaluation_split = "test" if test else "validation"
        evaluation_metrics, target, probability = run_epoch(model, loaders[2] if test else loaders[1], args.device)
        np.savez_compressed(folder / f"{evaluation_split}_predictions.npz", target=target, probability=probability)
        result = {"seed": seed, "evaluation_split": evaluation_split, "best_epoch": best_epoch,
                  "validation_bce": best_loss, **evaluation_metrics}
        write_json(folder / f"{evaluation_split}_metrics.json", result)
        results.append(result)
        write_json(args.output / "results.json", results)
        with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[key for key in result if key != "confusion_matrix"])
            writer.writeheader()
            writer.writerows({key: value for key, value in item.items() if key != "confusion_matrix"} for item in results)
    summary = {}
    for key in ("bce", "accuracy", "precision", "recall", "f1", "auroc", "average_precision"):
        values = [result[key] for result in results if result[key] is not None]
        summary[key] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None} if values else None
    write_json(args.output / "summary.json", {"seeds": args.seeds, "evaluation_split": "test" if test else "validation", "metrics": summary})


if __name__ == "__main__":
    main()
