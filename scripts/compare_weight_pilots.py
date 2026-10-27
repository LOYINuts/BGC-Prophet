"""Compare positive-weight pilots on validation only; never select on NG."""
import json
from pathlib import Path


def compare(root):
    records = []
    reference = None
    for weight in (1, 2, 4):
        folder = root / f"weight_{weight}/seed_1301"
        config = json.loads((folder / "config.json").read_text())
        split = json.loads((folder / "split.json").read_text())
        signature = {key: config[key] for key in ("dataset_sha256", "seed", "batch_size", "learning_rate",
                                                  "scheduler", "epochs", "patience", "nhead", "encoder_layers")}
        signature["split"] = split
        if reference is None:
            reference = signature
        if signature != reference or config["positive_weight"] != weight or config["evaluation_split"] != "validation":
            raise ValueError("Pilot configurations or data splits differ beyond positive weight")
        status = json.loads((folder / "training_status.json").read_text())
        metrics = json.loads((folder / "validation_metrics.json").read_text())
        records.append({"positive_weight": weight, "training_status": status, "validation": metrics})
    # AP measures ranking without selecting a threshold on NG; ties favor smaller weight.
    chosen = max(records, key=lambda row: (row["validation"]["average_precision"], -row["positive_weight"]))
    baseline = records[0]["validation"]["average_precision"]
    gain = chosen["validation"]["average_precision"] - baseline
    return {"pilots": records, "selection_metric": "validation average precision at minimum-validation-BCE checkpoint",
            "selected_weight": chosen["positive_weight"], "validation_ap_gain": gain,
            "next_step": "confirm selected weight on seeds 1302 and 1303 using validation only" if gain > 0 else "retain unweighted baseline",
            "ng_evaluated": False, "limitation": "single seed, row split and different GPU models; gains need replication"}


if __name__ == "__main__":
    root = Path("runs/positive_weight_pilot")
    report = compare(root)
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
