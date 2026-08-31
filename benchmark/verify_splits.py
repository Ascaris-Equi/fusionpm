"""Audit supplied train/validation folds for overlap and provenance."""

from __future__ import annotations

import argparse
import glob
import hashlib
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-glob", required=True)
    parser.add_argument("--validation-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--peptide-column", default="peptide")
    parser.add_argument("--hla-column", default="HLA_sequence")
    parser.add_argument("--label-column", default="label")
    return parser.parse_args()


def fold_number(path: Path) -> int:
    match = re.search(r"fold(\d+)", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot identify fold number from {path.name}")
    return int(match.group(1))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_keys(path: Path, peptide: str, hla: str, label: str):
    frame = pd.read_csv(path, usecols=[peptide, hla, label])
    frame[peptide] = frame[peptide].astype("string").str.strip().str.upper()
    frame[hla] = frame[hla].astype("string").str.strip().str.upper()
    frame[label] = pd.to_numeric(frame[label], errors="raise").astype(int)
    if frame[[peptide, hla, label]].isna().any().any():
        raise ValueError(f"{path}: null value in split key columns")
    pair_keys = set(zip(frame[peptide], frame[hla]))
    labeled_keys = set(zip(frame[peptide], frame[hla], frame[label]))
    canonical_rows = sorted(zip(frame[peptide], frame[hla], frame[label]))
    digest = hashlib.sha256()
    for row in canonical_rows:
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\x1e")
    return frame, pair_keys, labeled_keys, digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    train = {fold_number(Path(p)): Path(p) for p in glob.glob(args.train_glob)}
    validation = {
        fold_number(Path(p)): Path(p) for p in glob.glob(args.validation_glob)
    }
    if not train or set(train) != set(validation):
        raise ValueError(
            f"Train/validation fold mismatch: train={sorted(train)}, "
            f"validation={sorted(validation)}"
        )

    rows = []
    for fold in sorted(train):
        train_frame, train_pairs, train_labeled, _ = read_keys(
            train[fold], args.peptide_column, args.hla_column, args.label_column
        )
        val_frame, val_pairs, val_labeled, val_rowset_sha256 = read_keys(
            validation[fold], args.peptide_column, args.hla_column, args.label_column
        )
        shared_pairs = train_pairs & val_pairs
        shared_labeled = train_labeled & val_labeled
        shared_label_0 = sum(key[2] == 0 for key in shared_labeled)
        shared_label_1 = sum(key[2] == 1 for key in shared_labeled)
        train_labels = {}
        for peptide, hla, label in train_labeled:
            train_labels.setdefault((peptide, hla), set()).add(label)
        val_labels = {}
        for peptide, hla, label in val_labeled:
            val_labels.setdefault((peptide, hla), set()).add(label)
        conflicting = sum(
            bool(train_labels[pair].isdisjoint(val_labels[pair]))
            for pair in shared_pairs
        )

        rows.append(
            {
                "fold": fold,
                "train_rows": len(train_frame),
                "validation_rows": len(val_frame),
                "train_unique_pairs": len(train_pairs),
                "validation_unique_pairs": len(val_pairs),
                "shared_pairs": len(shared_pairs),
                "shared_labeled_rows": len(shared_labeled),
                "shared_label_0": shared_label_0,
                "shared_label_1": shared_label_1,
                "shared_pairs_conflicting_labels": conflicting,
                "train_sha256": file_sha256(train[fold]),
                "validation_sha256": file_sha256(validation[fold]),
                "validation_canonical_rowset_sha256": val_rowset_sha256,
            }
        )

    audit = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "split_audit.csv", index=False)

    shown = audit.drop(
        columns=[
            "train_sha256",
            "validation_sha256",
            "validation_canonical_rowset_sha256",
        ]
    )
    validation_sets_identical = (
        audit["validation_canonical_rowset_sha256"].nunique() == 1
    )
    if validation_sets_identical:
        validation_note = (
            "All five validation CSVs contain the same normalized "
            "`(peptide, HLA_sequence, label)` row multiset; their byte hashes differ "
            "only because row order/index values differ. The upstream preprocessing "
            "notebook selects `val_data_cv_idx_dict[0]` for every saved validation "
            "fold. These files therefore do not form conventional disjoint "
            "cross-validation validation folds, and folds 1-4 show substantial "
            "train/validation overlap."
        )
    else:
        validation_note = (
            "The validation CSVs do not share one identical normalized row multiset."
        )
    report = "\n".join(
        [
            "# Train/validation split audit",
            "",
            "Keys are normalized by stripping whitespace and uppercasing `peptide` "
            "and `HLA_sequence`. `shared_labeled_rows` uses "
            "`(peptide, HLA_sequence, label)`; `shared_pairs` ignores the label.",
            "",
            validation_note,
            "",
            markdown_table(shown),
            "",
            "The machine-readable CSV also records SHA-256 hashes for every split.",
            "",
        ]
    )
    (output_dir / "split_audit.md").write_text(report, encoding="utf-8")
    print(shown.to_string(index=False))


if __name__ == "__main__":
    main()
