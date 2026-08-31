# Released model provenance

The repository contains five ordinary Git blobs, not Git LFS pointer files.
Each checkpoint strict-loads into the released `FusionPM` architecture.

| Checkpoint | File bytes | SHA-256 |
| --- | ---: | --- |
| `weights/model_fold0.pkl` | 11,425,471 | `e6e52bfed8848a62a2db95951cccf3d356915e2967f0aa70d35d702b0f3cdf60` |
| `weights/model_fold1.pkl` | 11,425,471 | `7e3d3a4ecf68aae553e9d9fbd97d1d989909cd5fe3824958a8d94f7e234bae71` |
| `weights/model_fold2.pkl` | 11,425,471 | `b9927e1b31207d94a17650cd7728033457dd648df31675ca8ce79055d7372de0` |
| `weights/model_fold3.pkl` | 11,425,471 | `4a78a3040c039d3c047d9e7441060db26d595d8b4f1dc1b9d94bf5594ba00cff` |
| `weights/model_fold4.pkl` | 11,425,471 | `1a54eca27ced66f68d8da7733f70901b990f38533d4ef5a683d13fa1b6981d8f` |

The exact trainable parameter count is **1,890,455**. The serialized state
dictionary contains 2,850,968 tensor elements in total: 1,890,455 parameters
plus 960,513 non-trainable buffer elements, principally fixed positional
encodings. Those tensors occupy 11,403,876 bytes; serialization metadata
accounts for the small difference from each checkpoint file size. Manuscript
tables should use 1,890,455 as the parameter count rather than inferring
2.85 M from checkpoint size or retaining a 1.75 M estimate.

Training uses base seed `19961231` and calls `set_seed(19961231 + fold)` for
fold index 0-4. The inference CLI averages positive-class probabilities from
the five released checkpoints by default. Because the released validation
CSVs are not five disjoint folds, this operation is described as a
five-checkpoint probability average rather than an unbiased cross-validation
estimate.

The model code implements cross-attention in both directions:
peptide-to-HLA (`pep_to_hla`) and HLA-to-peptide (`hla_to_pep`).
