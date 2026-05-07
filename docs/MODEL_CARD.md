# Model Card for Fusion-pM

## Model name

Fusion-pM

## Model type

Deep learning model for Class I HLA-peptide binding-related prediction.

## Intended use

Fusion-pM is intended for non-commercial research use in:

- HLA-peptide binding prediction
- peptide candidate prioritization
- neoantigen-oriented computational analysis
- exploratory immunoinformatics research
- attention-based model interpretation

## Out-of-scope use

Fusion-pM should not be used as a standalone tool for:

- clinical diagnosis
- treatment selection
- vaccine efficacy prediction
- regulatory decision-making
- patient-level medical decision-making
- commercial use without separate written permission

## Inputs

Fusion-pM accepts:

- full-length HLA Class I sequences or HLA pseudo-sequences
- peptide sequences of 8 to 14 amino acids
- multiple peptides, one per line

## Outputs

Depending on the workflow, outputs may include:

- HLA-peptide binding-related prediction scores
- ranked peptide candidates
- attention-based visualizations
- result tables

## Training data

Training data details should be documented in future releases, including:

- source datasets
- inclusion and exclusion criteria
- preprocessing pipeline
- train, validation, and test splits
- HLA allele coverage
- peptide length distribution
- positive and negative sample definitions

## Evaluation

Evaluation details should be documented in future releases, including:

- benchmark datasets
- metrics
- baseline models
- confidence intervals or statistical tests where appropriate
- source data for figures and tables

## Limitations

Important limitations include:

- Binding prediction does not prove immunogenicity.
- Attention scores are interpretability aids, not direct biological evidence.
- Performance may vary by HLA allele, peptide length, dataset, and experimental context.
- External validation is required before applying the model to new biological or clinical settings.
- The model should not be used as a standalone clinical decision-making tool.

## Ethical and clinical considerations

Fusion-pM is a research tool.

Users should avoid uploading or distributing sensitive patient data, personally identifiable information, or controlled clinical data unless they have proper authorization and comply with applicable regulations.

## License

Fusion-pM is provided for non-commercial research use under the PolyForm Noncommercial License 1.0.0 unless otherwise stated.

Commercial use requires prior written permission.

Contact:

```text
fusionpm@bayvaxbio.com
```
