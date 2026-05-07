# Fusion-pM

[![Python CI](https://github.com/Ascaris-Equi/fusionpm/actions/workflows/python-ci.yml/badge.svg)](https://github.com/Ascaris-Equi/fusionpm/actions/workflows/python-ci.yml)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Use](https://img.shields.io/badge/use-non--commercial%20research-orange)
![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)

Fusion-pM is a deep learning-based web service for Class I HLA-peptide binding prediction and immunogenicity-related analysis.

The model integrates HLA Class I sequences with peptide sequences and uses cross-attention and mask learning to support HLA-peptide binding prediction, candidate peptide ranking, and attention-based visualization.

> **License notice:** Fusion-pM is publicly available for **non-commercial research use only**. Commercial use requires prior written permission.
>
> **Clinical notice:** Fusion-pM is a computational research tool. It is not intended for clinical diagnosis, treatment selection, or standalone medical decision-making.

## Overview

Fusion-pM integrates full-length HLA Class I sequences or HLA pseudo-sequences with peptide sequences of 8 to 14 amino acids.

The model uses attention-based mechanisms to help identify informative HLA and peptide regions, including peptide anchor residues and HLA binding-groove-related positions.

## Key Features

- **HLA-peptide binding prediction**
  Predicts Class I HLA-peptide binding-related scores.

- **Peptide candidate ranking**
  Produces ranked peptide candidates to support neoantigen-oriented research workflows.

- **Attention-based visualization**
  Provides attention information for inspecting potentially important HLA-peptide regions.

- **Web interface**
  Includes a Gradio-based web application for local interactive use.

- **Pretrained model files**
  Includes pretrained `.pkl` model files for prediction workflows.

## Repository Contents

| Path | Description |
| --- | --- |
| `README.md` | Project overview and quick start |
| `requirements.txt` | Python dependencies |
| `gradio_app.py` | Gradio web application |
| `model.py` | Model architecture |
| `data_utils.py` | Data processing utilities |
| `config.py` | Configuration |
| `evaluate.py` | Evaluation utilities |
| `train.py` | Training script |
| `train_eval.py` | Training and evaluation script |
| `test.py` | Test or example script |
| `model_*.pkl`, `regmodel_*.pkl` | Pretrained model files |
| `vocab_dict.npy` | Vocabulary dictionary |
| `examples/` | Example input files |
| `scripts/` | Helper scripts |
| `docs/` | Documentation |

## Installation

Clone the repository:

```bash
git clone https://github.com/Ascaris-Equi/fusionpm.git
cd fusionpm
```

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Start

Run the Gradio web application:

```bash
python gradio_app.py
```

Alternatively:

```bash
bash scripts/run_webapp.sh
```

Then open the local Gradio URL shown in the terminal.

## Input Format

### HLA sequence

Input either:

- a full-length HLA Class I sequence
- an HLA pseudo-sequence

Use single-letter amino acid codes without spaces.

### Peptide sequence

Input peptide sequences of 8 to 14 amino acids.

Multiple peptides can be provided with one peptide per line.

Longer sequences may be segmented depending on the web-service workflow. Users should verify segmentation behavior before interpreting results.

## Example Input

Example HLA sequence:

```text
MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFFTSVSRPGR...
```

Example peptide:

```text
SIINFEKL
```

Multiple peptides:

```text
SIINFEKL
GILGFVFTL
NLVPMVATV
```

Example files are available in:

```text
examples/example_hla.txt
examples/example_peptides.txt
```

## Output

Fusion-pM returns HLA-peptide binding-related predictions and ranked peptide candidates.

Depending on the workflow, outputs may include:

- prediction scores
- ranked peptide tables
- attention-based visualizations
- downloadable result files

Predicted scores should be interpreted as computational prioritization results, not as direct evidence of T-cell immunogenicity or clinical efficacy.

## Documentation

Additional documentation:

- [Usage Guide](docs/USAGE.md)
- [Reproducibility Guide](docs/REPRODUCIBILITY.md)
- [Model Card](docs/MODEL_CARD.md)
- [License Policy](docs/LICENSE_POLICY.md)
- [Commercial Use](docs/COMMERCIAL_USE.md)

## Reproducibility

The repository currently includes pretrained `.pkl` model files.

For full manuscript-level reproducibility, users may also need access to:

- training, validation, and test splits
- processed benchmark datasets
- HLA allele or pseudo-sequence tables
- random seeds
- baseline model outputs
- metric calculation scripts
- source data for figures and tables

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Citation

If you use Fusion-pM, please cite this repository and the related manuscript if available.

Citation metadata is provided in:

```text
CITATION.cff
```

## License

Fusion-pM is distributed for non-commercial research use under the PolyForm Noncommercial License 1.0.0.

See:

- [LICENSE](LICENSE)
- [NOTICE](NOTICE)
- [docs/LICENSE_POLICY.md](docs/LICENSE_POLICY.md)
- [docs/COMMERCIAL_USE.md](docs/COMMERCIAL_USE.md)

Commercial use requires prior written permission.

For commercial licensing, contact:

```text
fusionpm@bayvaxbio.com
```

## Limitations

Fusion-pM is intended for computational HLA-peptide binding prediction and candidate prioritization.

Important limitations:

- Binding prediction alone does not prove T-cell immunogenicity.
- Predicted scores require experimental validation.
- The model should not be used as a standalone clinical decision-making tool.
- Performance may vary across HLA alleles, peptide lengths, datasets, and experimental settings.
- Attention visualizations can support interpretation but should not be treated as direct mechanistic proof.

## Authors

- Jiahao Ma, BayVax Biotech Limited
- Hongzong Li, BayVax Biotech Limited
- Xiaoping Su, Wenzhou Medical University
- Zhenzhai Cai, Second Affiliated Hospital of Wenzhou Medical University
- Ye-Fan Hu, BayVax Biotech Limited
- Yifan Chen, Hong Kong Baptist University
- Jian-Dong Huang, The University of Hong Kong

## Acknowledgments

Supported by the National Key Research and Development Program of China, the Health and Medical Research Fund, and other investors and sponsors.

## Contact

For questions, feedback, or commercial licensing, contact:

```text
fusionpm@bayvaxbio.com
```
