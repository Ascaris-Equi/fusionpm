# Baseline provenance

This inventory is limited to the **15 baselines named in the Fusion-pM
manuscript**. The `source` field distinguishes published provenance from the
one author-run comparator:

- 13 rows are transcribed from the immutable Figure 4 source workbook of
  Chu et al. (2022);
- HLAthena is a published method, but it is **not present** in that workbook
  and is therefore linked to its own article;
- HLA-Inception is the only baseline marked `local`.

Fusion-pM is the proposed model and is not included as a baseline row. The
machine-readable [`baseline_versions.csv`](baseline_versions.csv) is
authoritative. Its metric vectors are ordered as
`AUC;Accuracy;MCC;F1` and each Chu-derived row identifies the exact worksheet
and cell range.

Published Chu source: Chu et al., *Nature Machine Intelligence* (2022), DOI
[`10.1038/s42256-022-00459-7`](https://doi.org/10.1038/s42256-022-00459-7),
with the Figure 4 workbook pinned at TransPHLA-AOMP commit
[`3ed2260`](https://github.com/a96123155/TransPHLA-AOMP/blob/3ed2260292934170757507a71e645d0bcadfc44b/Source%20data-Figure%204.xlsx).

| Baseline | Exact release/version statement | Source | Workbook location or reference | AUC / Accuracy / MCC / F1 |
| --- | --- | --- | --- | --- |
| ACME | Chu Figure 4 result release; upstream model version not stated | published | `(c) Independent - Unmatchable!P1:P5` | 0.810841 / 0.705530 / 0.447862 / 0.635174 |
| ANN | Chu Figure 4 result release; upstream server version not stated | published | `(c) Independent - Unmatchable!B1:B5` | 0.932007 / 0.779802 / 0.613394 / 0.724008 |
| Anthem | Chu Figure 4 result release; upstream model version not stated | published | `(c) Independent - Unmatchable!N1:N5` | 0.977069 / 0.923851 / 0.847720 / 0.924256 |
| DeepNetBim | Chu Figure 4 result release; upstream model version not stated | published | `(c) Independent - Unmatchable!R1:R5` | 0.638353 / 0.558279 / 0.142117 / 0.388080 |
| HLAthena | MSi predictor described by Sarkizova et al.; server version not stated | published | [HLAthena article](https://doi.org/10.1038/s41587-019-0322-9); no Chu Figure 4 row | — |
| IEDB_consensus | Workbook label `Consensus`; upstream server version not stated | published | `(c) Independent - Unmatchable!F1:F5` | 0.931086 / 0.788322 / 0.626864 / 0.737543 |
| NetMHCcons | Chu Figure 4 result release; upstream server version not stated | published | `(c) Independent - Unmatchable!H1:H5` | 0.952890 / 0.789359 / 0.629417 / 0.738655 |
| NetMHCpan_BA | NetMHCpan 4.1 BA output reported by Chu et al. | published | `(a) Independent - Matchable!B1:B5` | 0.955066 / 0.801666 / 0.649185 / 0.757434 |
| NetMHCpan_EL | NetMHCpan 4.1 EL output reported by Chu et al. | published | `(a) Independent - Matchable!C1:C5` | 0.955796 / 0.795115 / 0.643276 / 0.744978 |
| NetMHCstabpan | NetMHCstabpan 1.0 output reported by Chu et al. | published | `(a) Independent - Matchable!D1:D5` | 0.915652 / 0.789685 / 0.621612 / 0.743866 |
| PickPocket | PickPocket 1.1 output reported by Chu et al. | published | `(a) Independent - Matchable!E1:E5` | 0.924055 / 0.702044 / 0.492774 / 0.584021 |
| SMM | Chu Figure 4 result release; upstream server version not stated | published | `(c) Independent - Unmatchable!J1:J5` | 0.912403 / 0.788691 / 0.606827 / 0.751184 |
| SMMPMBEC | Chu Figure 4 result release; upstream server version not stated | published | `(c) Independent - Unmatchable!L1:L5` | 0.915624 / 0.788739 / 0.609004 / 0.749755 |
| TransPHLA | TransPHLA-AOMP commit `3ed2260292934170757507a71e645d0bcadfc44b` | published | `(a) Independent - Matchable!F1:F5` | 0.978880 / 0.930844 / 0.861688 / 0.930927 |
| HLA-Inception | Zenodo `10.5281/zenodo.10516431`, published 2024-01-16; no version field | local | Author-run Fusion-pM comparison | — |

The words “matchable” and “unmatchable” are Chu workbook panel names, not two
fixed, complementary test subsets. Methods in the unmatchable sheet have
different evaluable row counts, so those sheet labels must not be interpreted
as two fixed, complementary subsets of the 171,438-row independent set.

HLA-Inception's official archived release, peptide-mode output convention,
supported lengths, and allele mapping are recorded in
[`HLA_INCEPTION_PROVENANCE.md`](HLA_INCEPTION_PROVENANCE.md).
