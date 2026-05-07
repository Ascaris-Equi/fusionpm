# Reproducibility Guide

This document describes how to install and run Fusion-pM for basic reproducibility.

## Environment

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Web Service

Run:

```bash
python gradio_app.py
```

Or use the helper script:

```bash
bash scripts/run_webapp.sh
```

## Input Format

Fusion-pM accepts:

- HLA Class I full-length sequences or pseudo-sequences
- peptide sequences of 8 to 14 amino acids
- multiple peptide sequences, one per line

Example files are available in:

```text
examples/example_hla.txt
examples/example_peptides.txt
```

## Model Weights

The repository includes pretrained `.pkl` model files.

For long-term reproducibility, future releases should ideally provide versioned model weights through GitHub Releases, Zenodo, or another archival storage service.

## Recommended Reproducibility Materials

For full manuscript-level reproducibility, the following materials should be provided:

- training, validation, and test splits
- processed benchmark datasets
- HLA allele or pseudo-sequence tables
- pretrained model weights
- random seeds
- metric calculation scripts
- baseline model outputs
- source data for figures and tables

## Limitations

Fusion-pM provides computational HLA-peptide binding-related predictions.

The predicted scores should not be interpreted as direct proof of T-cell immunogenicity, vaccine efficacy, clinical response, or disease protection without experimental validation.
