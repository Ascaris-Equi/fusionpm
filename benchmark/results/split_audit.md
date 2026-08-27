# Train/validation split audit

Keys are normalized by stripping whitespace and uppercasing `peptide` and `HLA_sequence`. `shared_labeled_rows` uses `(peptide, HLA_sequence, label)`; `shared_pairs` ignores the label.

| fold | train_rows | validation_rows | train_unique_pairs | validation_unique_pairs | shared_pairs | shared_labeled_rows | shared_label_0 | shared_label_1 | shared_pairs_conflicting_labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 574658 | 143674 | 573576 | 143619 | 563 | 563 | 3 | 560 | 0 |
| 1 | 574658 | 143674 | 573610 | 143619 | 71859 | 71859 | 77 | 71782 | 0 |
| 2 | 574658 | 143674 | 573557 | 143619 | 71857 | 71857 | 75 | 71782 | 0 |
| 3 | 574658 | 143674 | 573558 | 143619 | 71848 | 71848 | 66 | 71782 | 0 |
| 4 | 574696 | 143674 | 573614 | 143619 | 71864 | 71864 | 82 | 71782 | 0 |

The machine-readable CSV also records SHA-256 hashes for every split.
