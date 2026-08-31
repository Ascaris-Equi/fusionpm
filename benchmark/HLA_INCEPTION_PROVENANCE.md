# HLA-Inception provenance

## Recovered author-run record

The original local evaluation was recovered from the repository's
[`hla-inception`](https://github.com/Ascaris-Equi/fusionpm/tree/a85603a5616808aaa46c6e614f21837d258374be/hla-inception)
branch and is pinned here to commit
`a85603a5616808aaa46c6e614f21837d258374be`. The successful run began on
2026-05-20 at 19:49:09 and completed at 19:56:21. The earlier empty cached
outputs were detected and deleted before that run; the final log reports all
171,438 independent-set rows joined, zero missing predictions, and zero failed
alleles.

| Artifact | Pinned record |
| --- | --- |
| Run script | [`hla-inception/1.py`](https://github.com/Ascaris-Equi/fusionpm/blob/a85603a5616808aaa46c6e614f21837d258374be/hla-inception/1.py), Git blob `07c1412dc0f9b060fe2f500b305032ff5580da88` |
| Run log | [`all_results.txt`](https://github.com/Ascaris-Equi/fusionpm/blob/a85603a5616808aaa46c6e614f21837d258374be/hla-inception/hla_inception_run/all_results.txt), Git blob `e82cefb6a66e4fdb38f20631bf4dc9c0bfc8c1e5` |
| Input | `independent_set.csv`, 171,438 rows, SHA-256 `3abbf5264a3a6cf0787c542d8b26ec95e1357e15ad658d9b8ad05d98301da1b9` |
| Consolidated predictions | [`raw_predictions.csv`](https://github.com/Ascaris-Equi/fusionpm/blob/a85603a5616808aaa46c6e614f21837d258374be/hla-inception/hla_inception_run/results/independent/raw_predictions.csv), 171,438 rows, SHA-256/LFS OID `ce52e876a5236c6d6fc9eb174624c33fd8553ff00b3917f23d53935d5dd690f9` |
| Metrics | [`metrics.json`](https://github.com/Ascaris-Equi/fusionpm/blob/a85603a5616808aaa46c6e614f21837d258374be/hla-inception/hla_inception_run/metrics.json), SHA-256/LFS OID `a3e281069afb20e52eadbedc5f06f03f8635264ad6799a43ad5e980f678a3376` |

The branch intentionally remains separate from `main`: it contains duplicated
upstream source, model weights, and many per-allele outputs. The immutable
commit and object hashes above provide the run record without making the main
release archive unnecessarily large.

## Exact implementation and command

- Official repository: <https://github.com/eawilson-CompBio/HLA-Inception>
- Upstream source commit used by the recovered snapshot:
  `38fdbcbb2d1d35a42b4390b81c63e989c4453335`.
- Archived release: <https://doi.org/10.5281/zenodo.10516431>, published
  2024-01-16. The Zenodo record itself does not provide a version string.
- A file-level comparison against upstream commit `38fdbc...` found the 18
  shared files content-identical. One test FASTA is stored through Git LFS in
  the author-run branch but its underlying SHA-256 is identical; only macOS
  metadata files differ. This supports the commit assignment independently of
  the present-day upstream HEAD.
- The executable was compiled locally with Go 1.13.8. The run log records Git
  2.25.1 and Git LFS 2.9.2. An executable hash was not retained, so the source
  commit and archived prediction objects are the stable identifiers.

For each allele, the run script executed the following command with
`HI_PRED_PATH` set to the pinned HLA-Inception source directory:

```bash
HI_PRED_PATH=<HLA-Inception-dir> \
  <HLA-Inception-dir>/hla-inception \
  -i results/independent/<allele>/peps.in \
  -P 1 \
  -a <allele> \
  -o results/independent/<allele>/pred.out
```

Peptide mode returns the columns `RawScore`, `RawScorePercentile`,
`LengthCorrectedScore`, and `LengthCorrectedScorePercentile`. The run script
selected the final numeric field, `LengthCorrectedScorePercentile`, as
`hla_inception_score`. The independent set contains lengths 8-14; all 171,438
rows passed validation. HLA-Inception peptide mode also supports length 15.

## Metric definitions and recovered values

- ROC-AUC and AUPR use the unmodified `LengthCorrectedScorePercentile`; larger
  values denote stronger predicted binding.
- Thresholded metrics use
  `normalized_score = (score - min(score)) / (max(score) - min(score))` and
  predict positive when `normalized_score >= 0.5`.
- In the archived independent-set output, the minimum is 0.0 and the maximum
  is 99.8, so the exact raw-percentile cutoff is `score >= 49.9`.
- The official FASTA-mode default threshold of 99.5 was **not** used for this
  peptide-mode comparison.

| Metric | Value |
| --- | ---: |
| n | 171,438 |
| positives / negatives | 85,876 / 85,562 |
| ROC-AUC | 0.9458812899950538 |
| AUPR | 0.951698565374432 |
| Accuracy | 0.7554626162227744 |
| MCC | 0.5649714378332985 |
| F1 | 0.7988658225904727 |

## Allele mapping audit

Fusion-pM allele names map to the HLA-Inception CLI form by removing the
`HLA-` prefix and replacing `*` with `_`, for example
`HLA-A*02:01 -> A_02:01`. All 112 alleles in the frozen Fusion-pM lookup can
be submitted, so no rows or alleles were excluded. Five invoke the tool's
nearest-neighbor fallback:

| Input allele | HLA-Inception fallback | Alignment score |
| --- | --- | ---: |
| `HLA-B*73:01` | `B_39:50` | 0.08 |
| `HLA-B*45:06` | `B_45:01` | 0.09 |
| `HLA-C*17:01` | `C_05:206` | 0.09 |
| `HLA-A*24:06` | `A_24:13` | 0.79 |
| `HLA-A*02:50` | `A_02:122` | 0.97 |

The per-allele raw files are archived as
`results/independent/<allele>/pred.out`; the consolidated table is
`results/independent/raw_predictions.csv`. These records close the previously
unknown command, score, threshold, length-handling, allele-mapping, exclusion,
and output-file fields for the local baseline.
