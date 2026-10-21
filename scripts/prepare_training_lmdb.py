"""Convert existing ESM2 layer-6 Parquet embeddings to training LMDB.

Run with an existing uv-managed environment containing pyarrow, lmdb and torch.
Keys must exactly match CSV identifiers; missing keys abort before output creation.
"""
import argparse
import csv
import json
from pathlib import Path
import pickle

import lmdb
import numpy as np
import pyarrow.parquet as pq
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-lmdb", type=Path, required=True,
                        help="Compare sample values with original ESM2 LMDB before converting")
    args = parser.parse_args()
    with args.dataset.open() as handle:
        wanted = {gene for row in csv.DictReader(handle) for gene in row["TDsentence"].split()}
    keys = pq.read_table(args.parquet, columns=["gene_id"])["gene_id"].to_pylist()
    missing = wanted - set(keys)
    if missing or len(keys) != len(set(keys)):
        raise ValueError(f"Missing genes: {len(missing)}, first: {sorted(missing)[:10]}; duplicate source keys: {len(keys)-len(set(keys))}")
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = lmdb.open(str(args.reference_lmdb), readonly=True, lock=False)
    checked = 0
    source = pq.ParquetFile(args.parquet)
    with reference.begin() as transaction:
        for batch in source.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                if row["gene_id"] not in wanted:
                    continue
                raw = transaction.get(row["gene_id"].encode("ascii"))
                if raw is None:
                    raise ValueError(f"Sample missing from reference: {row['gene_id']}")
                expected = pickle.loads(raw)["mean_representations"][6].cpu().numpy()
                np.testing.assert_array_equal(np.asarray(row["embedding"], dtype=np.float32), expected)
                checked += 1
                if checked == 32:
                    break
            if checked == 32:
                break
    reference.close()
    args.output.mkdir(parents=True)
    environment = lmdb.open(str(args.output), map_size=16 * 1024**3)
    written = 0
    try:
        for batch in source.iter_batches(batch_size=4096):
            with environment.begin(write=True) as transaction:
                for row in batch.to_pylist():
                    key = row["gene_id"]
                    if key not in wanted:
                        continue
                    embedding = torch.tensor(row["embedding"], dtype=torch.float32)
                    if embedding.shape != (320,) or not torch.isfinite(embedding).all():
                        raise ValueError(f"Invalid embedding: {key}")
                    transaction.put(key.encode("ascii"), pickle.dumps({"mean_representations": {6: embedding}}))
                    written += 1
            print(f"Converted {written}/{len(wanted)}", flush=True)
        if written != len(wanted):
            raise ValueError(f"Expected {len(wanted)} genes, wrote {written}")
        environment.sync()
        (args.output / "provenance.json").write_text(json.dumps({
            "source": str(args.parquet.resolve()), "dataset": str(args.dataset.resolve()),
            "reference_lmdb": str(args.reference_lmdb.resolve()),
            "reference_samples_exactly_matched": checked, "genes": written,
        }, indent=2) + "\n")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
