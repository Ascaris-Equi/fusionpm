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

## Run the web service

```bash
python gradio_app.py
```

Or:

```bash
bash scripts/run_webapp.sh
```

## Example inputs

Example files:

```text
examples/example_hla.txt
examples/example_peptides.txt
```

## Model files

The repository includes pretrained model files:

```text
model_fold_0.pkl
model_fold_1.pkl
model_layer1_multihead9_fold0.pkl
model_layer1_multihead9_fold1.pkl
model_layer1_multihead9_fold2.pkl
model_layer1_multihead9_fold3.pkl
model_layer1_multihead9_fold4.pkl
regmodel_fold_1.pkl
```

## Basic code check

A basic syntax check can be run with:

```bash
python -m py_compile config.py data_utils.py evaluate.py gradio_app.py model.py test.py train.py train_eval.py
```

## Materials recommended for full reproducibility

For full manuscript-level reproducibility, the following materials should be provided or archived:

| Item | Status |
| --- | --- |
| Source code | Included |
| Pretrained model weights | Included |
| Example inputs | Included |
| Python dependencies | Included in `requirements.txt` |
| Training dataset | To be documented |
| Validation dataset | To be documented |
| Test dataset | To be documented |
| Dataset split files | To be documented |
| HLA allele or pseudo-sequence table | To be documented |
| Random seeds | To be documented |
| Baseline outputs | To be documented |
| Metric calculation scripts | To be documented |
| Source data for figures and tables | To be documented |

## Recommended archival practice

For stable releases, consider archiving:

- source code release tag
- model weights
- processed datasets
- benchmark outputs
- documentation
- citation metadata

Possible archival locations include GitHub Releases, Zenodo, Figshare, or an institutional repository.

## Limitations

Fusion-pM provides computational predictions for research use.

Predicted scores should not be interpreted as direct evidence of T-cell immunogenicity, vaccine efficacy, clinical response, or disease protection without experimental validation.
