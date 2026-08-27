# Training-overlap performance

Rows are normalized by stripping whitespace and uppercasing both `peptide` and `HLA_sequence`. The primary comparison treats an exact normalized `(peptide, HLA_sequence)` pair as seen when it occurs in the union of all training-fold CSVs. The five-way breakdown separates exact-pair overlap from component-level coverage.

| grouping | group | n_evaluable | n_unique_pairs | n_positive | positive_rate | auc | aupr | accuracy | mcc | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | all | 171438 | 171330 | 85876 | 0.5009 | 0.9860 | 0.9860 | 0.9444 | 0.8888 | 0.9446 |
| exact_pair_overlap | exact_pair_unseen | 170596 | 170488 | 85043 | 0.4985 | 0.9859 | 0.9858 | 0.9442 | 0.8883 | 0.9441 |
| exact_pair_overlap | exact_pair_seen | 842 | 842 | 833 | 0.9893 | 0.9763 | 0.9997 | 0.9917 | 0.6696 | 0.9958 |
| novelty_group | peptide_unseen_hla_seen | 110742 | 110718 | 37314 | 0.3369 | 0.9827 | 0.9701 | 0.9378 | 0.8625 | 0.9094 |
| novelty_group | both_components_seen_new_pair | 59854 | 59770 | 47729 | 0.7974 | 0.9866 | 0.9961 | 0.9559 | 0.8686 | 0.9720 |
| novelty_group | exact_pair_seen | 842 | 842 | 833 | 0.9893 | 0.9763 | 0.9997 | 0.9917 | 0.6696 | 0.9958 |
| novelty_group | peptide_seen_hla_unseen | 0 | 0 | 0 | NA | NA | NA | NA | NA | NA |
| novelty_group | both_unseen | 0 | 0 | 0 | NA | NA | NA | NA | NA | NA |

Decision threshold: `score > 0.5`.

Machine-readable outputs: `seen_unseen_metrics.csv`, `seen_unseen_summary.json`, and `dataset_manifest.csv`.
