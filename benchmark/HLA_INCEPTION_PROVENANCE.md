# HLA-Inception provenance

## Verified release information

- Official repository: <https://github.com/eawilson-CompBio/HLA-Inception>
- Archived release: <https://doi.org/10.5281/zenodo.10516431>, published
  2024-01-16. The Zenodo record does not provide a version string.
- The current repository HEAD is not used as a surrogate version for the
  Zenodo archive because source changes occurred after the archive date.
- The official peptide-mode command form is:

  ```bash
  ./hla-inception -i <peptide-file> -P 1 -a <allele> -o <output-file>
  ```

- Peptide mode supports lengths 8-15 and returns every input prediction. Its
  output includes `RawScore`, `RawScorePercentile`,
  `LengthCorrectedScore`, and `LengthCorrectedScorePercentile`.
- In the official output convention, a larger percentile denotes stronger
  predicted binding. The documented default `99.5` threshold applies to the
  FASTA scanning mode; peptide mode itself does not apply a binary threshold.

## Allele mapping audit

Fusion-pM allele names map to the HLA-Inception CLI form by removing the
`HLA-` prefix and replacing `*` with `_`, for example
`HLA-A*02:01 -> A_02:01`. All 112 alleles in the frozen Fusion-pM lookup can
be submitted to the official alignment table. Five invoke the tool's
nearest-neighbor fallback:

| Input allele | HLA-Inception fallback | Alignment score |
| --- | --- | ---: |
| `HLA-B*73:01` | `B_39:50` | 0.08 |
| `HLA-B*45:06` | `B_45:01` | 0.09 |
| `HLA-C*17:01` | `C_05:206` | 0.09 |
| `HLA-A*24:06` | `A_24:13` | 0.79 |
| `HLA-A*02:50` | `A_02:122` | 0.97 |

## Run-record fields

For a thresholded comparison, record the exact executable/archive, full
command, binary percentile threshold, continuous score used for ROC-AUC,
peptide-length handling, allele mapping, and raw-output filename together with
the result table. The official FASTA-mode default must not be silently applied
to peptide mode, which returns unthresholded scores.
