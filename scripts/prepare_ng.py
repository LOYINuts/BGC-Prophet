"""Prepare our NG GenBank and labels for fresh ESM2 encoding and evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def windows(genes, length=128):
    if not genes:
        raise ValueError("Contig has no CDS genes")
    if len(genes) < length:
        yield (genes * ((length + len(genes) - 1) // len(genes)))[:length]
        return
    for start in range(0, len(genes), length):
        yield genes[start:start + length] if start + length <= len(genes) else genes[-length:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genbank", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.annotations.open() as handle:
        annotations = {row["Locus Tag"].strip(): row["Cluster ID"].strip()
                       for row in csv.DictReader(handle, delimiter="\t") if row["Locus Tag"].strip()}
    genes, rows, proteins = set(), [], []
    for record in SeqIO.parse(args.genbank, "genbank"):
        contig = []
        for feature in record.features:
            if feature.type != "CDS":
                continue
            key = feature.qualifiers.get("locus_tag", [""])[0]
            translation = feature.qualifiers.get("translation", [""])[0]
            if not key or key in genes or not translation:
                raise ValueError(f"Missing/duplicate CDS tag or missing protein sequence: {key}")
            genes.add(key)
            contig.append(key)
            proteins.append(SeqRecord(Seq(translation), id=key, description=""))
        for index, window in enumerate(windows(contig)):
            labels = [int(bool(annotations.get(key))) for key in window]
            rows.append({"ID": f"{record.id}_{index}", "labels": "Unknown",
                         "TDsentence": " ".join(window), "TDlabels": " ".join(map(str, labels))})
    missing_annotations = set(annotations) - genes
    if not genes or not genes.intersection(annotations):
        raise ValueError("No annotated genes found in GenBank")
    if missing_annotations:
        print(f"Annotation tags absent from GenBank: {sorted(missing_annotations)}", flush=True)
    args.output.mkdir(parents=True)
    SeqIO.write(proteins, args.output / "proteins.faa", "fasta")
    with (args.output / "dataset.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "labels", "TDsentence", "TDlabels"])
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"source_files": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (args.genbank, args.annotations)},
                "unique_genes": len(genes), "windows": len(rows), "annotated_genes": len(annotations),
                "annotated_genes_in_genbank": len(set(annotations) & genes),
                "annotation_tags_absent_from_genbank": sorted(missing_annotations),
                "protocol": "GenBank CDS order per contig; stride 128; final window last 128; short contigs repeated",
                "labels": "locus tag membership in 291BGC.csv; unmatched CDS negative",
                "encoding_required": "fresh esm2_t6_8M_UR50D layer 6 mean; truncate proteins to 1022 residues",
                "metrics_unit": "gene positions in windows, including overlapping contig tails"}
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
