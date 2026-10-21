"""CPU checks use only synthetic CSVs, proteins and embeddings."""
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
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

from bgc_prophet.train.reproduce import metrics
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
