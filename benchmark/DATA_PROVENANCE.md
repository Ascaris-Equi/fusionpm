# Data provenance and split-generation status

## Frozen pHLA inputs used here

The model-ready files used for the revision analysis are the supplied
`train_data_fold[0-4].csv`, `val_data_fold[0-4].csv`, and
`independent_set.csv`. The HLA lookup is included for inference inputs that
provide allele names instead of pseudo-sequences.

These filenames and schemas originate from the TransPHLA data release. The
pinned upstream repository is:

- repository: <https://github.com/a96123155/TransPHLA-AOMP>
- commit: `3ed2260292934170757507a71e645d0bcadfc44b`
- upstream preprocessing notebook:
  `Dataset/2_Deduplication of positive and negative samples_different data sets.ipynb`
- retrieval date for the frozen local inputs: 2026-08-27

The HPL-APMS-pHLA data documentation independently identifies its
`train_data_fold4.csv`, `val_data_fold4.csv`, and independent data as being
consistent with the TransPHLA release:

- repository: <https://github.com/Jiadong001/HPL-APMS-pHLA>
- commit: `64f3e97177a85eb0f7c7e85cf1e6eb938833c78a`
- archived data DOI: <https://doi.org/10.6084/m9.figshare.28863005>

## Frozen input manifest

The following table is the byte-level contract for the manuscript benchmark.
The TransPHLA commit records source lineage; the SHA-256 values identify the
exact frozen files used in the reported analysis.

| File | Role | Source snapshot | Retrieved | Rows | Bytes | SHA-256 |
| --- | --- | --- | --- | ---: | ---: | --- |
| `train_data_fold0.csv` | training split 0 | TransPHLA `3ed2260292934170757507a71e645d0bcadfc44b` | 2026-08-27 | 574658 | 39397753 | `b48f7a297cc7af091df44c1e9faa4efaea34fe268224b0c2f45c62843b3b829d` |
| `train_data_fold1.csv` | training split 1 | TransPHLA `3ed2260292934170757507a71e645d0bcadfc44b` | 2026-08-27 | 574658 | 39397753 | `928326253e30c6db9b7cebdb025250579ee1ca6eff657c30639d93009213f71a` |
| `train_data_fold2.csv` | training split 2 | TransPHLA `3ed2260292934170757507a71e645d0bcadfc44b` | 2026-08-27 | 574658 | 39397753 | `f2789090ea4d87fde9048ee437296cf1862ad5d3376196235a5fdd75ae79fa81` |
| `train_data_fold3.csv` | training split 3 | TransPHLA `3ed2260292934170757507a71e645d0bcadfc44b` | 2026-08-27 | 574658 | 39397753 | `762ce7dc9cbe0e8cd2f81181ee18170b178b3e9081086f2c4b0b62236e519cc8` |
| `train_data_fold4.csv` | training split 4 | TransPHLA `3ed2260292934170757507a71e645d0bcadfc44b` | 2026-08-27 | 574696 | 39400265 | `1cbb046f990c359325f49d1d6624c192ee82bbf3e40b45407264427d441351de` |
| `val_data_fold0.csv` | validation split 0 | TransPHLA `3ed2260292934170757507a71e645d0bcadfc44b` | 2026-08-27 | 143674 | 9766763 | `c5fa1f948a90cf8756159a5bdb2bc6d0413ca2263a120096966c99abdbb57e22` |
| `val_data_fold1.csv` | validation split 1 | TransPHLA `3ed2260292934170757507a71e645d0bcadfc44b` | 2026-08-27 | 143674 | 9766763 | `ed4f19c57f4294212811c3b06c6b0faa3a426b4b7d89c7f7893b8b6335f0ca5a` |
| `val_data_fold2.csv` | validation split 2 | TransPHLA `3ed2260292934170757507a71e645d0bcadfc44b` | 2026-08-27 | 143674 | 9766763 | `373d138b603c1aff8b0d9e020ce7c501ca1b654dcfeaebb7fc043e961622b7ec` |
| `val_data_fold3.csv` | validation split 3 | TransPHLA `3ed2260292934170757507a71e645d0bcadfc44b` | 2026-08-27 | 143674 | 9766763 | `e691cacd911be3f20591ac6a8ff3da848510fc483e5272188495772df98b8dd4` |
| `val_data_fold4.csv` | validation split 4 | TransPHLA `3ed2260292934170757507a71e645d0bcadfc44b` | 2026-08-27 | 143674 | 9766763 | `975c142363804b13ef747de454f1967e8f37e19354263e80cc0f8d1fa9f1de90` |
| `independent_set.csv` | independent test | TransPHLA `3ed2260292934170757507a71e645d0bcadfc44b` | 2026-08-27 | 171438 | 11673010 | `3abbf5264a3a6cf0787c542d8b26ec95e1357e15ad658d9b8ad05d98301da1b9` |
| `common_hla.csv` | optional allele lookup | TransPHLA `3ed2260292934170757507a71e645d0bcadfc44b` | 2026-08-27 | 112 | 5281 | `b4ee3ec46cfaf0a50a0a9300e1e36fe84b0d58b1fe7700f6c2299499b73d9bbe` |

The independent test file contains 171,438 rows: 85,876 positives and
85,562 negatives (positive rate 0.500916). After stripping whitespace and
uppercasing `peptide` and `HLA_sequence`, it contains 171,330 unique pairs,
108 repeated pair rows, and no pair with conflicting labels.

The five training CSVs contain 2,873,328 rows in total. This sum counts rows
again when they occur in multiple released training folds and therefore is
not a count of unique training examples. It does not support a 515,149-row
description of these released model-ready files.

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

## Released validation-split behavior

The supplied TransPHLA-derived CSVs are frozen benchmark inputs rather than
newly generated splits. Byte-level comparison against the pinned upstream
snapshot found all 12 listed CSVs identical to their upstream files.

The five validation files have different byte hashes because their index and
row order differ, but they contain the same normalized
`(peptide, HLA_sequence, label)` row multiset. The pinned upstream notebook
uses `val_data_cv_idx_dict[0]` when saving every `val_data_fold*.csv`, rather
than indexing that dictionary by the current fold. Consequently, folds 1-4
have extensive train/validation pair overlap, as quantified in
[`results/split_audit.md`](results/split_audit.md).

These files are therefore not represented as conventional disjoint
five-fold cross-validation validation folds. Inference results average the
probabilities from five released checkpoints; they are not claimed as an
unbiased five-fold cross-validation estimate. The source snapshot, retrieval
date, dimensions, and SHA-256 manifest above still make the released inputs
independently verifiable byte for byte.
