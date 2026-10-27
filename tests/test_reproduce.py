"""CPU checks use only synthetic CSVs, proteins and embeddings."""
import csv
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import lmdb
import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from bgc_prophet.train.reproduce import detection_loss, main, metrics
from bgc_prophet.train.model import transformerEncoderNet
from scripts.prepare_ng import windows


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
                    txn.put(f"gene{i}".encode(), pickle.dumps({"mean_representations": {6: torch.full((320,), i / 8)}}))
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
            for output in ("runs", "repeat"):
                invocation = command.copy()
                invocation[invocation.index(str(root / "runs"))] = str(root / output)
                completed = subprocess.run(invocation, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            summary = json.loads((root / "runs/summary.json").read_text())
            self.assertEqual(summary["seeds"], [1301, 1302])
            self.assertEqual(summary["evaluation_split"], "test")
            self.assertIsNotNone(summary["metrics"]["f1"]["sample_std"])
            for seed in (1301, 1302):
                folder = root / f"runs/seed_{seed}"
                split = json.loads((folder / "split.json").read_text())
                self.assertFalse(set(split["train"]) & set(split["validation"]))
                self.assertEqual(len(split["train"]), 8)
                state = torch.load(folder / "best.pt", weights_only=True)
                model = transformerEncoderNet(d_model=320, nhead=5, num_encoder_layers=2, dim_feedforward=1280)
                model.load_state_dict(state)
                with np.load(folder / "test_predictions.npz") as first, np.load(root / f"repeat/seed_{seed}/test_predictions.npz") as second:
                    self.assertEqual(first["probability"].shape, (10, 8))
                    np.testing.assert_array_equal(first["probability"], second["probability"])
            validation_only = command.copy()
            validation_only[validation_only.index(str(root / "runs"))] = str(root / "validation")
            option = validation_only.index("--test-lmdb")
            del validation_only[option:option + 2]
            completed = subprocess.run(validation_only, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((root / "validation/summary.json").read_text())
            self.assertEqual(report["evaluation_split"], "validation")
            self.assertFalse((root / "validation/seed_1301/test_metrics.json").exists())
            refused = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("Seed output exists", refused.stderr)
            checked = subprocess.run(command + ["--check-only"], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_early_stopping_and_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "lmdb"
            env = lmdb.open(str(database), map_size=1_000_000)
            with env.begin(write=True) as transaction:
                transaction.put(b"gene", pickle.dumps({"mean_representations": {6: torch.zeros(320)}}))
            env.close()
            dataset = root / "data.csv"
            with dataset.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["labels", "TDsentence", "TDlabels"])
                writer.writeheader()
                for _ in range(10):
                    writer.writerow({"labels": "NRP", "TDsentence": " ".join(["gene"] * 8),
                                     "TDlabels": "0 0 0 0 1 1 1 1"})
            validation_losses = iter([0.4, 0.4, 0.3, 0.3, 0.31])
            observed_rates = []
            observed_batches = []
            calls = []
            def fake_epoch(model, loader, device, optimizer=None, positive_weight=1.0):
                calls.append(optimizer is not None)
                if optimizer is not None:
                    observed_rates.append(optimizer.param_groups[0]["lr"])
                    observed_batches.append(loader.batch_size)
                    with torch.no_grad():
                        next(model.parameters()).fill_(len(observed_rates))
                    loss = 0.5
                else:
                    loss = next(validation_losses, 0.3)
                return {"bce": loss, **metrics([0, 1], [0.1, 0.9])}, np.array([[0, 1]]), np.array([[0.1, 0.9]])
            arguments = ["reproduce", "--dataset", str(dataset), "--lmdb", str(database),
                         "--device", "cpu", "--max-len", "8", "--seeds", "1301",
                         "--patience", "2", "--output", str(root / "run")]
            with patch.object(sys, "argv", arguments), patch("bgc_prophet.train.reproduce.run_epoch", side_effect=fake_epoch) as mocked_epoch:
                main()
                mocked_epoch.reset_mock()
            folder = root / "run/seed_1301"
            status = json.loads((folder / "training_status.json").read_text())
            self.assertEqual(status["completed_epochs"], 5)
            self.assertEqual(status["best_epoch"], 2)
            self.assertEqual(status["stale_epochs"], 2)
            self.assertEqual(status["stop_reason"], "early_stopping")
            self.assertEqual(observed_rates, [0.01] * 5)
            self.assertEqual(observed_batches, [64] * 5)
            self.assertEqual(calls, [True, False] * 5 + [False])
            best = torch.load(folder / "best.pt", weights_only=True)
            last = torch.load(folder / "last.pt", weights_only=True)
            first_key = next(iter(best))
            self.assertTrue(torch.all(best[first_key] == 3))
            self.assertTrue(torch.all(last[first_key] == 5))
            config = json.loads((folder / "config.json").read_text())
            self.assertIsNone(config["epochs"])
            self.assertEqual(config["patience"], 2)
            observed_rates.clear()
            calls.clear()
            # Constant validation loss must not stop a legacy run; decay starts after epoch 5.
            validation_losses = iter([0.4] * 8)
            legacy = arguments.copy()
            legacy[legacy.index(str(root / "run"))] = str(root / "legacy")
            legacy[legacy.index("--patience") + 1] = "0"
            legacy += ["--epochs", "8", "--scheduler", "original"]
            with patch.object(sys, "argv", legacy), patch("bgc_prophet.train.reproduce.run_epoch", side_effect=fake_epoch) as mocked_epoch:
                main()
                mocked_epoch.reset_mock()
            status = json.loads((root / "legacy/seed_1301/training_status.json").read_text())
            self.assertEqual(status["completed_epochs"], 8)
            self.assertEqual(status["stop_reason"], "epoch_cap")
            self.assertEqual(observed_rates[:6], [0.01] * 6)
            self.assertAlmostEqual(observed_rates[6], 0.01 * 0.95)
            self.assertAlmostEqual(observed_rates[7], 0.01 * 0.95 ** 2)

    def test_positive_weight_only_changes_positive_loss(self):
        probability = torch.tensor([0.2, 0.7], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        raw = torch.nn.functional.binary_cross_entropy(probability, target, reduction="none")
        torch.testing.assert_close(detection_loss(probability, target), raw.mean())
        weighted = detection_loss(probability, target, 4.0)
        torch.testing.assert_close(weighted, (raw[0] + 4 * raw[1]) / 2)
        base_gradient, = torch.autograd.grad(raw.mean(), probability, retain_graph=True)
        weighted_gradient, = torch.autograd.grad(weighted, probability)
        torch.testing.assert_close(weighted_gradient, base_gradient * torch.tensor([1.0, 4.0]))

    def test_parallel_aggregate(self):
        from scripts.summarize_parallel import aggregate
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folders = {}
            for seed in range(1301, 1306):
                folder = root / str(seed)
                folder.mkdir()
                folders[seed] = folder
                config = dict(dataset_sha256="train", test_dataset_sha256="test", batch_size=32,
                              learning_rate=0.0005, epochs=200, scheduler="original", patience=0,
                              nhead=5, encoder_layers=2)
                (folder / "config.json").write_text(json.dumps(config))
                (folder / "training_status.json").write_text(json.dumps(dict(completed_epochs=200, stop_reason="epoch_cap")))
                result = dict(seed=seed, evaluation_split="test", bce=0.1,
                              **metrics([0, 1], [0.1, 0.9]))
                (folder / "test_metrics.json").write_text(json.dumps(result))
            aggregate(folders, root / "summary")
            summary = json.loads((root / "summary/summary.json").read_text())
            self.assertEqual(summary["metrics"]["f1"], {"mean": 1.0, "sample_std": 0.0})
            config["batch_size"] = 64
            (folders[1305] / "config.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                aggregate(folders, root / "invalid")
            self.assertFalse((root / "invalid").exists())

    def test_weight_pilot_selection(self):
        from scripts.compare_weight_pilots import compare
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for weight, ap in ((1, 0.8), (2, 0.85), (4, 0.81)):
                folder = root / f"weight_{weight}/seed_1301"
                folder.mkdir(parents=True)
                config = dict(dataset_sha256="same", seed=1301, batch_size=32, learning_rate=0.0005,
                              scheduler="original", epochs=100, patience=20, nhead=5, encoder_layers=2,
                              positive_weight=weight, evaluation_split="validation")
                (folder / "config.json").write_text(json.dumps(config))
                (folder / "split.json").write_text(json.dumps({"train": [0], "validation": [1]}))
                (folder / "training_status.json").write_text(json.dumps({"completed_epochs": 30}))
                (folder / "validation_metrics.json").write_text(json.dumps({"average_precision": ap}))
            report = compare(root)
            self.assertEqual(report["selected_weight"], 2)
            self.assertFalse(report["ng_evaluated"])
            self.assertAlmostEqual(report["validation_ap_gain"], 0.05)
            (root / "weight_4/seed_1301/split.json").write_text(json.dumps({"train": [1], "validation": [0]}))
            with self.assertRaises(ValueError):
                compare(root)

    def test_prepare_ng(self):
        self.assertEqual(len(list(windows(list(range(256))))), 2)
        self.assertEqual(list(windows(list(range(129))))[-1], list(range(1, 129)))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = SeqRecord(Seq("ATG" * 30), id="contig", annotations={"molecule_type": "DNA"})
            record.features = [SeqFeature(SimpleLocation(i * 9, (i + 1) * 9), type="CDS",
                                          qualifiers={"locus_tag": [f"g{i}"], "translation": ["MMM"]})
                               for i in range(3)]
            SeqIO.write([record], root / "test.gbk", "genbank")
            (root / "labels.tsv").write_text("Locus Tag\tCluster ID\ng1\tcluster1\n")
            command = [sys.executable, "scripts/prepare_ng.py", "--genbank", str(root / "test.gbk"),
                       "--annotations", str(root / "labels.tsv"), "--output", str(root / "prepared")]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (root / "prepared/dataset.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(len(row["TDsentence"].split()), 128)
            self.assertEqual(row["TDlabels"].split()[:6], ["0", "1", "0", "0", "1", "0"])
            self.assertEqual(json.loads((root / "prepared/provenance.json").read_text())["unique_genes"], 3)
            proteins = list(SeqIO.parse(root / "prepared/proteins.faa", "fasta"))
            self.assertEqual([protein.id for protein in proteins], ["g0", "g1", "g2"])
            self.assertEqual(str(proteins[0].seq), "MMM")
            self.assertFalse((root / "prepared/lmdb").exists())
            (root / "labels.tsv").write_text("Locus Tag\tCluster ID\ng1\tcluster1\nabsent\tcluster2\n")
            command[-1] = str(root / "partial_annotations")
            partial = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(partial.returncode, 0, partial.stderr)
            manifest = json.loads((root / "partial_annotations/provenance.json").read_text())
            self.assertEqual(manifest["annotation_tags_absent_from_genbank"], ["absent"])
            self.assertEqual(manifest["annotated_genes_in_genbank"], 1)
            (root / "labels.tsv").write_text("Locus Tag\tCluster ID\nabsent\tcluster1\n")
            command[-1] = str(root / "invalid")
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("No annotated genes found in GenBank", failed.stderr)
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
