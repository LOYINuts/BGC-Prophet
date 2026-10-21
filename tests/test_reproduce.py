"""Small CPU checks; all CSVs and embeddings here are synthetic."""
import csv
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import lmdb
import numpy as np
import torch

from bgc_prophet.train.reproduce import metrics
from bgc_prophet.train.model import transformerEncoderNet


class ReproductionTests(unittest.TestCase):
    def test_metrics_use_raw_counts(self):
        result = metrics([0] * 8 + [1] * 2, [0.1] * 6 + [0.9] * 3 + [0.1])
        self.assertEqual(result["confusion_matrix"], [[6, 2], [1, 1]])
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.4)
        self.assertIsNone(metrics([0, 0], [0.1, 0.2])["auroc"])

    def test_seed_runs_and_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "lmdb"
            env = lmdb.open(str(database), map_size=10_000_000)
            with env.begin(write=True) as txn:
                for i in range(8):
                    value = {"mean_representations": {6: torch.full((320,), i / 8)}}
                    txn.put(f"gene{i}".encode(), pickle.dumps(value))
            env.close()
            dataset = root / "dataset.csv"
            with dataset.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "labels", "TDsentence", "TDlabels"])
                writer.writeheader()
                for row in range(10):
                    writer.writerow({"ID": f"row{row}", "labels": "NRP", "TDsentence": " ".join(f"gene{i}" for i in range(8)), "TDlabels": "0 0 0 0 1 1 1 1"})
            command = [sys.executable, "-m", "bgc_prophet.train.reproduce", "--dataset", str(dataset),
                       "--test-dataset", str(dataset), "--lmdb", str(database), "--test-lmdb", str(database),
                       "--output", str(root / "runs"), "--seeds", "1301", "1302", "--epochs", "1",
                       "--max-len", "8", "--batch-size", "8", "--device", "cpu"]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            summary = json.loads((root / "runs/summary.json").read_text())
            self.assertEqual(summary["seeds"], [1301, 1302])
            self.assertIsNotNone(summary["metrics"]["f1"]["sample_std"])
            for seed in (1301, 1302):
                folder = root / f"runs/seed_{seed}"
                split = json.loads((folder / "split.json").read_text())
                self.assertFalse(set(split["train"]) & set(split["validation"]))
                self.assertEqual(len(split["train"]), 8)
                state = torch.load(folder / "best.pt", weights_only=True)
                model = transformerEncoderNet(d_model=320, nhead=5, num_encoder_layers=2, dim_feedforward=1280)
                model.load_state_dict(state)
                with np.load(folder / "test_predictions.npz") as predictions:
                    self.assertEqual(predictions["probability"].shape, (10, 8))
            repeated = command.copy()
            repeated[repeated.index(str(root / "runs"))] = str(root / "repeat")
            completed = subprocess.run(repeated, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for seed in (1301, 1302):
                with np.load(root / f"runs/seed_{seed}/test_predictions.npz") as first, np.load(root / f"repeat/seed_{seed}/test_predictions.npz") as second:
                    np.testing.assert_array_equal(first["probability"], second["probability"])
            refused = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("Seed output exists", refused.stderr)
            checked = subprocess.run(command + ["--check-only"], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == "__main__":
    unittest.main()
