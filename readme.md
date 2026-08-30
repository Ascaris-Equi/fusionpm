# Fusion-pM

Fusion-pM is a deep learning–based service for Class I HLA–peptide binding
prediction and immunogenicity-related analysis. The model integrates HLA
Class I sequences with peptide sequences and uses **cross-attention** and
**masked-residue learning** to support HLA–peptide binding prediction,
candidate peptide ranking, and NetMHCpan-style affinity reporting.

**License notice:** Fusion-pM is publicly available for
**non-commercial research use only**. Commercial use requires prior written
permission.

**Clinical notice:** Fusion-pM is a computational research tool. It is not
intended for clinical diagnosis, treatment selection, or standalone medical
decision-making.

---

## Overview

Fusion-pM integrates HLA Class I pseudo-sequences with peptide sequences of
8 to 14 amino acids. On top of a TransPHLA-style dual self-attention
backbone, Fusion-pM adds:

- a **Cross-Attention bridge** between the peptide and HLA encoders
  (bidirectional);
- a **Masked Language Modeling** auxiliary head on peptide residues for
  regularization during training;
- a **NetMHCpan-style affinity transform** on the output, reported together
  with `SB / WB / NB` binder classes.

The model uses attention-based mechanisms to help identify informative
HLA and peptide regions, including peptide anchor residues and HLA
binding-groove-related positions.

## Key Features

- **HLA–peptide binding prediction.** Predicts Class I HLA–peptide
  binding-related scores.
- **Peptide candidate ranking.** Produces ranked peptide candidates to
  support neoantigen-oriented research workflows.
- **NetMHCpan-style IC50 reporting.** Reports a ranking-friendly
  pseudo-affinity `IC50_nM = 50000^(1 − score)` together with SB / WB / NB
  classes.
- **5-fold ensemble inference.** Default inference averages five
  cross-validation folds; an optional fast mode uses the best single fold.
- **Pretrained model files.** Includes five pretrained `.pkl` model files
  for direct inference.

## Repository Contents

| Path | Description |
|---|---|
| `README.md` | Project overview and quick start |
| `model.py` | Model architecture (backbone + cross-attention + MLM head) |
| `train.py` | 5-fold training script |
| `infer.py` | CSV-in / CSV-out inference (ensemble or fast mode) |
| `weights/model_fold[0-4].pkl` | Pretrained 5-fold weights |
| `weights/vocab_dict.npy` | Vocabulary dictionary |
| `weights/best_fold.json` | Best-fold index used by `--fast` |
| `dataset/` | User-provided data folder (not tracked in git) |
| `benchmark/` | Manuscript benchmark commands, provenance, and result tables |

## Installation

Clone the repository:

```
git clone https://github.com/Ascaris-Equi/fusionpm.git
cd fusionpm
```

Create a Python environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```
python -m pip install --upgrade pip
python -m pip install torch numpy pandas scikit-learn
```

## Quick Start

Run inference on a CSV file (5-fold ensemble, default):

```
python infer.py --input dataset/independent_set.csv --output preds.csv
```

Single best-fold mode (≈ 4× faster, slightly lower accuracy):

```
python infer.py --input dataset/independent_set.csv --output preds.csv --fast
```

## Input Format

Each input CSV must contain a `peptide` column and either an `HLA_sequence`
column or an `HLA` allele column.

| column | required | content |
|---|---|---|
| `peptide` | yes | 8–14 amino acids, single-letter codes, no spaces |
| `HLA_sequence` | yes (or `HLA`) | Class-I pseudo-sequence, up to 34 amino acids |
| `HLA` | optional | Allele name (e.g. `HLA-A*02:01`), resolved via `dataset/common_hla.csv` |
| `label` | optional | 0 / 1 binding label (only used for training and offline evaluation) |
| `id` | optional | Passed through to output |

When `HLA_sequence` is absent, Fusion-pM looks up the pseudo-sequence from
`dataset/common_hla.csv` (`HLA` / `allele` → `HLA_sequence`).

## Output Format

Each output CSV contains:

```
id, peptide, HLA_sequence,
score, IC50_nM, binder_class, pred_label, rank, n_models, status
```

| column | meaning |
|---|---|
| `score` | Mean softmax-positive probability across `n_models` folds |
| `IC50_nM` | NetMHCpan-style affinity transform: `50000^(1 − score)` |
| `binder_class` | `SB` (< 50 nM), `WB` (< 500 nM), `NB` (≥ 500 nM) |
| `pred_label` | `int(score > threshold)`; default threshold 0.5 |
| `rank` | Dense rank within the same HLA_sequence (1 = best) |
| `n_models` | Number of fold weights actually used for the row |
| `status` | `ok`, or short tag (`bad-pep-len(7)`, `bad-hla-aa(X)`, ...) |

Invalid rows are kept with `score = NaN`; the run does not abort.

`IC50_nM` is a ranking-friendly pseudo-affinity derived from the
binary-classifier confidence via the NetMHCpan transform. It is **not** a
quantitative IC50 trained on regression labels.

## Inference CLI

| flag | default | meaning |
|---|---|---|
| `--input` | — (required) | Input CSV path |
| `--output` | `<input>.pred.csv` | Output CSV path |
| `--batch_size` | 4096 | Inference batch size |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |
| `--threshold` | 0.5 | `pred_label` cutoff |
| `--top_k` | 0 | Keep only top-N peptides per HLA_sequence (0 = all) |
| `--fast` | off | Use only the best single fold (per `weights/best_fold.json`) |
| `--dry_run` | off | Validate the input CSV only; no inference |

## Training

Place the dataset files under `./dataset/`:

```
dataset/
├── train_data_fold0.csv ... train_data_fold4.csv
├── val_data_fold0.csv   ... val_data_fold4.csv
├── independent_set.csv
├── external_set.csv
└── common_hla.csv          (optional, for HLA-allele → pseudo-seq lookup)
```

Then run 5-fold training:

```
python train.py
```

| flag | default | meaning |
|---|---|---|
| `--folds 0,2` | all | Comma-separated list of fold indices |
| `--epochs N` | 50 | Epochs per fold |
| `--batch_size` | 4096 | Tuned for a 24–32 GB GPU |
| `--num_workers` | 6 | Tuned for a 6c/12t CPU |
| `--lr` | 1e-3 | Adam learning rate |
| `--mask_rate` | 0.15 | MLM mask rate on peptide tokens |
| `--mlm_w` | 0.1 | MLM auxiliary loss weight |
| `--force` | off | Retrain even if a fold weight already exists |
| `--device` | `auto` | `auto` / `cuda` / `cpu` |

Training uses TF32 + bfloat16 AMP automatically on Ampere/Ada/Blackwell.
At the end of each fold, the script prints TransPHLA-style metrics on the
validation, `independent_set`, and `external_set` splits — **only model
weights are written to disk** (`weights/model_fold[0-4].pkl`). The fold
with the highest validation `avg(auc + acc + mcc + f1) / 4` is recorded in
`weights/best_fold.json` and is used by `infer.py --fast`.

## Reproducing the manuscript benchmark

The iScience manuscript benchmark evaluates all 171,438 independent-test rows
with the released five-fold ensemble. Exact commands and configuration are in
[`benchmark/README.md`](benchmark/README.md), frozen input provenance and
SHA-256 values are in
[`benchmark/DATA_PROVENANCE.md`](benchmark/DATA_PROVENANCE.md), baseline and
web-server versions are in
[`benchmark/BASELINE_PROVENANCE.md`](benchmark/BASELINE_PROVENANCE.md), and the
GitHub-rendered seen/unseen result is in
[`benchmark/results/README.md`](benchmark/results/README.md).

Quick check (a few minutes on a recent GPU):

```
python train.py --folds 0 --epochs 3
python infer.py --input dataset/independent_set.csv --output ind_pred.csv --fast
```

For full manuscript-level reproducibility, the following materials are
typically also required:

- training, validation, and test splits;
- processed benchmark datasets;
- HLA allele or pseudo-sequence tables;
- random seeds;
- baseline model outputs;
- metric calculation scripts;
- source data for figures and tables.

## Limitations

Fusion-pM is intended for computational HLA–peptide binding prediction and
candidate prioritization. Important limitations:

- Binding prediction alone does not prove T-cell immunogenicity.
- Predicted scores require experimental validation.
- The model should not be used as a standalone clinical decision-making
  tool.
- Performance may vary across HLA alleles, peptide lengths, datasets, and
  experimental settings.
- Attention visualizations can support interpretation but should not be
  treated as direct mechanistic proof.
- `IC50_nM` is derived from the classifier score, not from a quantitative
  regression target.

## Authors

- Jiahao Ma, BayVax Biotech Limited
- Hongzong Li, BayVax Biotech Limited
- Yaojun Yu, Second Affiliated Hospital of Wenzhou Medical University
- Xiaoping Su, Wenzhou Medical University
- Zhenzhai Cai, Second Affiliated Hospital of Wenzhou Medical University
- Ye-Fan Hu, BayVax Biotech Limited
- Yifan Chen, Hong Kong Baptist University
- Jian-Dong Huang, The University of Hong Kong

## Acknowledgments

Supported by the National Key Research and Development Program of China,
the Health and Medical Research Fund, and other investors and sponsors.

## License

Fusion-pM is distributed for non-commercial research use under the
PolyForm Noncommercial License 1.0.0. Commercial use requires prior written
permission. For commercial licensing, contact: **fusionpm@bayvaxbio.com**

## Contact

For questions, feedback, or commercial licensing, contact:
**fusionpm@bayvaxbio.com**
