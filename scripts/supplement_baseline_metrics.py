"""Add independently verified MCC to fixed baseline without changing original artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score


def mcc_from_counts(matrix):
    (tn, fp), (fn, tp) = matrix
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else 0.0


def supplement(evidence, root=None):
    source = json.loads(evidence.read_text())
    records = []
    for row in source["seeds"]:
        result = row["test_metrics"]
        mcc = mcc_from_counts(result["confusion_matrix"])
        if root is not None:
            folder = root / Path(row["run_directory"]).relative_to(source["server_repository"])
            with np.load(folder / "test_predictions.npz", allow_pickle=False) as arrays:
                target = arrays["target"].ravel()
                prediction = arrays["probability"].ravel() >= 0.5
                assert confusion_matrix(target, prediction, labels=[0, 1]).tolist() == result["confusion_matrix"]
                assert abs(mcc - matthews_corrcoef(target, prediction)) < 1e-12
                assert abs(result["precision"] - precision_score(target, prediction, zero_division=0)) < 1e-12
                assert abs(result["recall"] - recall_score(target, prediction, zero_division=0)) < 1e-12
        records.append(dict(seed=row["seed"], precision=result["precision"], recall=result["recall"],
                            mcc=mcc, confusion_matrix=result["confusion_matrix"]))
    assert [row["seed"] for row in records] == list(range(1301, 1306))
    summary = {}
    for key in ("precision", "recall", "mcc"):
        values = [row[key] for row in records]
        summary[key] = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values))
    return dict(source_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                threshold=0.5, aggregation="per-seed metrics then mean and sample SD (ddof=1)",
                prediction_arrays_verified=root is not None, zero_denominator_mcc=0.0,
                seeds=records, summary=summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = supplement(args.evidence, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
