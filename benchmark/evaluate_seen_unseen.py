"""Evaluate Fusion-pM on training-seen and training-unseen test examples.

The primary comparison uses the exact normalized (peptide, HLA_sequence)
pair as the overlap key. A five-way novelty breakdown is also emitted so
that exact-pair overlap is not confused with peptide-only or HLA-only
coverage.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


METRIC_COLUMNS = [
    "grouping",
    "group",
    "n_total",
    "n_unique_pairs",
    "n_evaluable",
    "n_positive",
    "positive_rate",
    "auc",
    "aupr",
    "accuracy",
    "mcc",
    "f1",
    "sensitivity",
    "specificity",
    "precision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare performance for training-seen and unseen test rows."
    )
    parser.add_argument(
        "--train-glob",
        required=True,
        help="Glob for training split CSVs, e.g. dataset/train_data_fold*.csv",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Fusion-pM prediction CSV containing labels and scores",
    )
    parser.add_argument(
        "--test-input",
        default=None,
        help="Optional raw test CSV to include separately in the data manifest",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--peptide-column", default="peptide")
    parser.add_argument("--hla-column", default="HLA_sequence")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--write-annotated",
        action="store_true",
        help="Also write row-level predictions with overlap annotations",
    )
    return parser.parse_args()


def normalized(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def require_columns(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_metric(function, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(function(y_true, y_score))
    except (ValueError, IndexError):
        return float("nan")


def evaluate_group(
    frame: pd.DataFrame,
    grouping: str,
    group: str,
    label_column: str,
    score_column: str,
    threshold: float,
) -> dict[str, object]:
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    scores = pd.to_numeric(frame[score_column], errors="coerce")
    valid = labels.isin([0, 1]) & scores.notna()
    y_true = labels.loc[valid].astype(int).to_numpy()
    y_score = scores.loc[valid].astype(float).to_numpy()
    y_pred = (y_score > threshold).astype(int)

    result: dict[str, object] = {
        "grouping": grouping,
        "group": group,
        "n_total": int(len(frame)),
        "n_unique_pairs": int(frame["_normalized_pair"].nunique()),
        "n_evaluable": int(valid.sum()),
        "n_positive": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "auc": safe_metric(roc_auc_score, y_true, y_score),
        "aupr": safe_metric(average_precision_score, y_true, y_score),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(y_true) else float("nan"),
    }

    if len(y_true):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        precision = tp / (tp + fp) if tp + fp else float("nan")
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity
            else 0.0
        )
    else:
        sensitivity = specificity = precision = f1 = float("nan")

    result.update(
        {
            "f1": float(f1),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
        }
    )
    return result


def markdown_table(metrics: pd.DataFrame) -> str:
    shown = metrics[
        [
            "grouping",
            "group",
            "n_evaluable",
            "n_unique_pairs",
            "n_positive",
            "positive_rate",
            "auc",
            "aupr",
            "accuracy",
            "mcc",
            "f1",
        ]
    ].copy()
    for column in ["positive_rate", "auc", "aupr", "accuracy", "mcc", "f1"]:
        shown[column] = shown[column].map(
            lambda value: "NA" if pd.isna(value) else f"{value:.4f}"
        )
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    train_paths = [Path(path) for path in sorted(glob.glob(args.train_glob))]
    if not train_paths:
        raise FileNotFoundError(f"No files matched --train-glob: {args.train_glob}")

    prediction_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    peptide_set: set[str] = set()
    hla_set: set[str] = set()
    pair_set: set[tuple[str, str]] = set()
    manifest_rows: list[dict[str, object]] = []

    for path in train_paths:
        frame = pd.read_csv(path, usecols=[args.peptide_column, args.hla_column])
        require_columns(path, frame, [args.peptide_column, args.hla_column])
        peptides = normalized(frame[args.peptide_column])
        hlas = normalized(frame[args.hla_column])
        valid = peptides.notna() & hlas.notna()
        peptide_set.update(peptides.loc[valid].tolist())
        hla_set.update(hlas.loc[valid].tolist())
        pair_set.update(zip(peptides.loc[valid], hlas.loc[valid]))
        manifest_rows.append(
            {
                "role": "training_split",
                "file": path.name,
                "bytes": path.stat().st_size,
                "rows": int(len(frame)),
                "sha256": sha256(path),
            }
        )

    predictions = pd.read_csv(prediction_path)
    require_columns(
        prediction_path,
        predictions,
        [
            args.peptide_column,
            args.hla_column,
            args.label_column,
            args.score_column,
        ],
    )
    manifest_rows.append(
        {
            "role": "test_predictions",
            "file": prediction_path.name,
            "bytes": prediction_path.stat().st_size,
            "rows": int(len(predictions)),
            "sha256": sha256(prediction_path),
        }
    )

    peptides = normalized(predictions[args.peptide_column])
    hlas = normalized(predictions[args.hla_column])
    peptide_seen = peptides.isin(peptide_set)
    hla_seen = hlas.isin(hla_set)
    pair_seen = pd.Series(
        [pair_ in pair_set for pair_ in zip(peptides, hlas)],
        index=predictions.index,
        dtype=bool,
    )

    predictions["exact_pair_overlap"] = np.where(
        pair_seen, "exact_pair_seen", "exact_pair_unseen"
    )
    predictions["_normalized_pair"] = list(zip(peptides, hlas))
    predictions["novelty_group"] = np.select(
        [
            pair_seen,
            peptide_seen & hla_seen,
            peptide_seen & ~hla_seen,
            ~peptide_seen & hla_seen,
        ],
        [
            "exact_pair_seen",
            "both_components_seen_new_pair",
            "peptide_seen_hla_unseen",
            "peptide_unseen_hla_seen",
        ],
        default="both_unseen",
    )

    rows = [
        evaluate_group(
            predictions,
            "overall",
            "all",
            args.label_column,
            args.score_column,
            args.threshold,
        )
    ]
    group_values = {
        "exact_pair_overlap": ["exact_pair_unseen", "exact_pair_seen"],
        "novelty_group": [
            "peptide_unseen_hla_seen",
            "both_components_seen_new_pair",
            "exact_pair_seen",
            "peptide_seen_hla_unseen",
            "both_unseen",
        ],
    }
    for grouping, groups in group_values.items():
        for group in groups:
            group_frame = predictions.loc[predictions[grouping] == group]
            rows.append(
                evaluate_group(
                    group_frame,
                    grouping,
                    str(group),
                    args.label_column,
                    args.score_column,
                    args.threshold,
                )
            )

    metrics = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    metrics.to_csv(output_dir / "seen_unseen_metrics.csv", index=False)

    summary = {
        "definition": {
            "normalization": "strip whitespace and uppercase both key columns",
            "primary_overlap_key": [args.peptide_column, args.hla_column],
            "training_reference": "union of all files matched by --train-glob",
            "decision_rule": f"predicted positive when {args.score_column} > {args.threshold}",
        },
        "training_files": [path.name for path in train_paths],
        "unique_training_peptides": len(peptide_set),
        "unique_training_hlas": len(hla_set),
        "unique_training_pairs": len(pair_set),
        "test_rows": int(len(predictions)),
        "unique_test_pairs": int(predictions["_normalized_pair"].nunique()),
        "duplicate_test_rows_by_pair": int(
            len(predictions) - predictions["_normalized_pair"].nunique()
        ),
        "test_rows_by_exact_pair_overlap": {
            str(key): int(value)
            for key, value in predictions["exact_pair_overlap"].value_counts().items()
        },
        "test_rows_by_novelty_group": {
            str(key): int(value)
            for key, value in predictions["novelty_group"].value_counts().items()
        },
    }

    if args.test_input:
        test_input_path = Path(args.test_input)
        test_input = pd.read_csv(test_input_path)
        manifest_rows.insert(
            len(train_paths),
            {
                "role": "independent_test",
                "file": test_input_path.name,
                "bytes": test_input_path.stat().st_size,
                "rows": int(len(test_input)),
                "sha256": sha256(test_input_path),
            },
        )
    pd.DataFrame(manifest_rows).to_csv(output_dir / "dataset_manifest.csv", index=False)
    (output_dir / "seen_unseen_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = "\n".join(
        [
            "# Training-overlap performance",
            "",
            "Rows are normalized by stripping whitespace and uppercasing both `peptide` and "
            "`HLA_sequence`. The primary comparison treats an exact normalized "
            "`(peptide, HLA_sequence)` pair as seen when it occurs in the union of all "
            "training-fold CSVs. The five-way breakdown separates exact-pair overlap from "
            "component-level coverage.",
            "",
            markdown_table(metrics),
            "",
            f"Decision threshold: `{args.score_column} > {args.threshold}`.",
            "",
            "Machine-readable outputs: `seen_unseen_metrics.csv`, "
            "`seen_unseen_summary.json`, and `dataset_manifest.csv`.",
            "",
        ]
    )
    (output_dir / "README.md").write_text(report, encoding="utf-8")

    if args.write_annotated:
        predictions.drop(columns=["_normalized_pair"]).to_csv(
            output_dir / "independent_predictions_annotated.csv", index=False
        )

    print(metrics.to_string(index=False))
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
