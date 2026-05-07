# Usage Guide

This guide describes how to run Fusion-pM locally and prepare inputs.

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Run the web application

```bash
python gradio_app.py
```

Or:

```bash
bash scripts/run_webapp.sh
```

Open the local Gradio URL printed in the terminal.

## 3. Prepare inputs

### HLA sequence

Use a full-length HLA Class I sequence or a pseudo-sequence.

Requirements:

- single-letter amino acid code
- no spaces
- avoid invalid characters

### Peptide sequence

Use peptide sequences of 8 to 14 amino acids.

For multiple peptides, provide one sequence per line:

```text
SIINFEKL
GILGFVFTL
NLVPMVATV
```

## 4. Interpret outputs

Fusion-pM outputs computational prediction scores and ranked candidate peptides.

The results are intended for research prioritization. They should not be interpreted as proof of:

- T-cell activation
- immunogenicity
- vaccine efficacy
- clinical response
- disease protection

Experimental validation is required.

## 5. Troubleshooting

### Dependency installation fails

Try upgrading `pip` first:

```bash
python -m pip install --upgrade pip
```

Then reinstall:

```bash
python -m pip install -r requirements.txt
```

### Gradio URL does not open

Check whether the application is still running in the terminal.

If running on a remote server, make sure the server port is accessible.

### Model file not found

Check that the pretrained `.pkl` files are present in the repository root.

### Invalid sequence errors

Check that input sequences contain only valid amino acid single-letter codes.
