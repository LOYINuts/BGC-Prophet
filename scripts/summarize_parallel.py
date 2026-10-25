"""Combine preserved seed 1301 with four separately trained legacy seeds."""
import csv
import json
from pathlib import Path
import statistics


def aggregate(folders, output):
    results = []
    reference = None
    compare_keys = ("dataset_sha256", "test_dataset_sha256", "batch_size", "learning_rate",
                    "epochs", "scheduler", "patience", "nhead", "encoder_layers")
    for seed, folder in folders.items():
        config = json.loads((folder / "config.json").read_text())
        status = json.loads((folder / "training_status.json").read_text())
        result = json.loads((folder / "test_metrics.json").read_text())
        expected = {key: config[key] for key in compare_keys}
        if reference is None:
            reference = expected
        if expected != reference or result["seed"] != seed or result["evaluation_split"] != "test":
            raise ValueError(f"Seed {seed} has inconsistent configuration or result")
        if status["completed_epochs"] != 200 or status["stop_reason"] != "epoch_cap":
            raise ValueError(f"Seed {seed} did not finish the legacy protocol")
        results.append(result)
    if sorted(folders) != list(range(1301, 1306)):
        raise ValueError("Exactly seeds 1301-1305 required")
    summary = {}
    for metric in ("bce", "accuracy", "precision", "recall", "f1", "auroc", "average_precision"):
        values = [result[metric] for result in results]
        summary[metric] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps({"seeds": sorted(folders), "evaluation_split": "test",
                                                    "metrics": summary,
                                                    "seed_directories": {str(k): str(v) for k, v in folders.items()}}, indent=2) + "\n")
    with (output / "results.csv").open("w", newline="") as handle:
        fields = [key for key in results[0] if key != "confusion_matrix"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    root = Path("runs/legacy_remaining_parallel")
    folders = {1301: Path("runs/detection_17112180_ng_esm2_legacy_complete/seed_1301")}
    folders.update({seed: root / f"seed_{seed}/seed_{seed}" for seed in range(1302, 1306)})
    aggregate(folders, root)
