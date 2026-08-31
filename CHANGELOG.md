# Changelog

## v1.0.3-iscience - 2026-08-31

- Corrected the benchmark inventory to the 15 manuscript baselines: ACME,
  ANN, Anthem, DeepNetBim, HLAthena, IEDB_consensus, NetMHCcons,
  NetMHCpan_BA, NetMHCpan_EL, NetMHCstabpan, PickPocket, SMM, SMMPMBEC,
  TransPHLA, and HLA-Inception.
- Added exact Chu Figure 4 worksheet/cell provenance and published metric
  vectors for the 13 methods actually present in that workbook; HLAthena now
  points to its own publication.
- Replaced the inaccurate HLA-Inception `v1`/current-HEAD pairing with its
  dated Zenodo record and documented verified official tool behavior.
- Added validation files to the generated dataset manifest and documented
  that the five released validation CSVs contain the same normalized row
  multiset.
- Described inference as a five-checkpoint probability average rather than an
  unbiased five-fold cross-validation estimate.
- Added exact checkpoint hashes, the 1,890,455 trainable-parameter count
  (separated from 960,513 serialized buffer elements), and the full PolyForm
  Noncommercial License 1.0.0 text.
- Added Zenodo release metadata with the complete author list and the
  PolyForm Noncommercial license identifier; README citations use the stable
  concept DOI `10.5281/zenodo.22178340`.
