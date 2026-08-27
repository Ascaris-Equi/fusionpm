# Data provenance and split-generation status

## Frozen pHLA inputs used here

The model-ready files used for the revision analysis are the supplied
`train_data_fold[0-4].csv`, `val_data_fold[0-4].csv`, and
`independent_set.csv`. Their exact byte hashes and row counts are recorded in
`results/dataset_manifest.csv` and `results/split_audit.csv`.

These filenames and schemas originate from the TransPHLA data release. The
pinned upstream repository is:

- repository: <https://github.com/a96123155/TransPHLA-AOMP>
- commit: `3ed2260292934170757507a71e645d0bcadfc44b`
- upstream preprocessing notebook:
  `Dataset/2_Deduplication of positive and negative samples_different data sets.ipynb`

The HPL-APMS-pHLA data documentation independently identifies its
`train_data_fold4.csv`, `val_data_fold4.csv`, and independent data as being
consistent with the TransPHLA release:

- repository: <https://github.com/Jiadong001/HPL-APMS-pHLA>
- commit: `64f3e97177a85eb0f7c7e85cf1e6eb938833c78a`
- archived data DOI: <https://doi.org/10.6084/m9.figshare.28863005>

## What this repository reproduces

Fusion-pM exactly records and reproduces the model-side preprocessing applied
to the frozen CSV inputs:

1. `train.py::load_csv` reads either indexed or standard CSV files;
2. `encode_df` strips and uppercases sequences;
3. peptides are right-padded/truncated to 15 residues;
4. HLA pseudo-sequences are right-padded/truncated to 34 residues;
5. rows containing characters absent from the fixed 20-amino-acid vocabulary
   are skipped;
6. labels are converted to integer binary targets.

`benchmark/verify_splits.py` records file hashes and audits exact
train/validation pair overlap. `benchmark/evaluate_seen_unseen.py` records the
independent-test overlap analysis.

## Submission blocker: original five-fold generation

The current Fusion-pM repository receives the five fold CSV pairs as frozen
inputs; it does not yet regenerate them from the raw TransPHLA source tables.
Therefore, the repository must not claim that it reproduces the original
five-fold split-generation step.

To satisfy the editor's request literally, add one of the following before
submission:

1. the original split-generation notebook/script plus all required raw inputs
   and a command that reproduces the recorded SHA-256 hashes; or
2. a versioned data-repository DOI containing the frozen splits, with the
   manuscript and response letter explicitly stating that Fusion-pM uses the
   published TransPHLA splits unchanged.

The audit also shows substantial exact positive-pair overlap between the
provided train and validation files for folds 1–4. The fold construction and
intended pairing should be checked against the original source before these
folds are described as independent validation partitions.
