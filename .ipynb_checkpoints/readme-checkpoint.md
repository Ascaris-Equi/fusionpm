# Fusion-pM

Fusion-pM is a deep learning-based service for Class I HLA-peptide binding
prediction and immunogenicity-related analysis. The model integrates HLA
Class I sequences with peptide sequences and uses **cross-attention** and
**masked-residue learning** to support HLA-peptide binding prediction,
candidate peptide ranking, and NetMHCpan-style IC50 reporting.

**License notice:** Fusion-pM is publicly available for non-commercial
research use only. Commercial use requires prior written permission
(see `LICENSE`, `NOTICE`).

**Clinical notice:** Fusion-pM is a computational research tool. It is not
intended for clinical diagnosis, treatment selection, or standalone medical
decision-making.

## Overview

Fusion-pM integrates Class I HLA pseudo-sequences (≤34 AA) with peptide
sequences of 8 to 14 amino acids. On top of the TransPHLA-style dual
self-attention backbone, Fusion-pM adds:

- **Cross-Attention bridge** between peptide and HLA encoders (bidirectional)
- **Masked Language Modeling** auxiliary head on peptide residues
  (regularization during training)
- **NetMHCpan-style IC50** transform on the output: `IC50_nM = 50000^(1−score)`,
  with SB / WB / NB binder classes

## Repository layout

fusionpm/
├── dataset/ # put your data here
│ ├── train_data_fold[0-4].csv
│ ├── val_data_fold[0-4].csv
│ ├── independent_set.csv
│ ├── external_set.csv
│ └── common_hla.csv
├── weights/ # created/filled by train.py
│ ├── model_fold[0-4].pkl
│ ├── vocab_dict.npy
│ └── best_fold.json
├── model.py # architecture (backbone + cross-attn + MLM head)
├── train.py # 5-fold trainer
├── infer.py # ensemble / fast-mode CSV inference
└── README.md


## Installation

git clone https://github.com/Ascaris-Equi/fusionpm.git
cd fusionpm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pandas scikit-learn
n1ql


## Data format

Each CSV must contain three columns (TransPHLA-compatible; an extra leading
index column is tolerated):

| column         | content                                                |
|----------------|--------------------------------------------------------|
| `peptide`      | 8–14 amino acids, single-letter codes                  |
| `HLA_sequence` | Class-I pseudo-sequence, up to 34 amino acids          |
| `label`        | 0 / 1 binding label                                    |

`dataset/common_hla.csv` (optional) maps allele names (`HLA-A*02:01`, ...) to
pseudo-sequences and is used by `infer.py` when the input CSV has an `HLA`
column instead of `HLA_sequence`.

## Training (5-fold)

python train.py
gherkin


Useful flags:

| flag             | default | meaning                                       |
|------------------|---------|-----------------------------------------------|
| `--folds 0,2`    | all     | comma-separated list of fold indices          |
| `--epochs N`     | 50      | epochs per fold                               |
| `--batch_size`   | 4096    | tuned for 24–32 GB GPU (e.g. RTX 5090)        |
| `--num_workers`  | 6       | tuned for 6c/12t CPU (e.g. Ryzen 5600X)       |
| `--lr`           | 1e-3    | Adam learning rate                            |
| `--mask_rate`    | 0.15    | MLM mask rate on peptide tokens               |
| `--mlm_w`        | 0.1     | MLM auxiliary loss weight                     |
| `--force`        | off     | retrain even if a fold weight already exists  |
| `--device`       | auto    | `auto`/`cuda`/`cpu`                           |

`train.py` uses TF32 + bfloat16 AMP automatically on Ampere/Ada/Blackwell.
At the end of each fold it prints TransPHLA-style metrics on `val`,
`independent_set` and `external_set` (no extra files written; **only model
weights are saved** to `weights/`).

The fold with the highest validation `avg(auc+acc+mcc+f1)/4` is recorded in
`weights/best_fold.json` for the inference `--fast` mode.

## Inference

Default — ensemble across all folds:

python infer.py --input examples/input.csv --output out.csv


Fast mode — single best fold only (≈4× faster):

python infer.py --input examples/input.csv --output out.csv --fast
gherkin


CLI flags:

| flag           | default | meaning                                          |
|----------------|---------|--------------------------------------------------|
| `--input`      | (req.)  | CSV path                                         |
| `--output`     | auto    | output CSV path                                  |
| `--batch_size` | 4096    |                                                  |
| `--device`     | auto    | `auto` / `cuda` / `cpu`                          |
| `--threshold`  | 0.5     | `pred_label` cutoff                              |
| `--top_k N`    | 0       | keep only top-N peptides per HLA_sequence (0=all)|
| `--fast`       | off     | single-fold inference                            |
| `--dry_run`    | off     | input validation only                            |

### Input columns

`peptide` (required) + one of `HLA_sequence` or `HLA` (allele name, looked up
via `dataset/common_hla.csv`). Optional `id` is passed through.

### Output columns

id, peptide, HLA_sequence,
score, IC50_nM, binder_class, pred_label, rank, n_models, status
scheme


- `score` — mean softmax-positive probability across `n_models` folds.
- `IC50_nM` — NetMHCpan-style affinity transform `50000^(1 − score)`.
  Reported as a **ranking-friendly pseudo-affinity** derived from the
  binary-classifier confidence; this is **not** a quantitative IC50 trained on
  regression labels.
- `binder_class` — `SB` (IC50 < 50 nM), `WB` (< 500 nM), `NB` (≥ 500 nM).
- `rank` — dense rank within the same HLA_sequence (1 = best).
- `status` — `ok` or short tag (`bad-pep-len(7)`, `bad-hla-aa(X)`, ...).
  Invalid rows are kept with `score = NaN`; the run does not abort.

## Reproducibility quick-check

python train.py --folds 0 --epochs 3 # ~minutes on RTX 5090
python infer.py --input dataset/independent_set.csv --output ind_pred.csv --fast


## Limitations

- Binding-prediction confidence does not prove T-cell immunogenicity.
- IC50 here is a transform of the classifier score, not a regression target.
- Performance varies across HLA alleles, peptide lengths, datasets, and
  experimental settings.
- Attention visualizations support interpretation, not mechanistic proof.

## Authors

Jiahao Ma · Hongzong Li · Xiaoping Su · Zhenzhai Cai · Ye-Fan Hu ·
Yifan Chen · Jian-Dong Huang

## Contact

`fusionpm@bayvaxbio.com`