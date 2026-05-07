# Fusion-pM

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-To%20be%20specified-lightgrey)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Platform](https://img.shields.io/badge/platform-Gradio-orange)

Fusion-pM is a deep learning-based web service for Class I HLA-peptide binding prediction and immunogenicity-related analysis.

The model integrates HLA Class I sequences with peptide sequences and uses cross-attention and mask learning to support peptide-HLA binding prediction, candidate peptide ranking, and interactive visualization.

## Background

Fusion-pM integrates full-length HLA sequences or pseudo-sequences with peptides of 8 to 14 amino acids.

The model uses cross-attention mechanisms to focus on informative regions, such as peptide anchor residues and HLA binding-groove-related regions. This helps make predictions more interpretable.

## Key Features

- **HLA-peptide binding prediction**  
  Predicts Class I HLA-peptide binding-related scores.

- **Ranked outputs**  
  Generates prioritized peptide candidate lists.

- **Interactive visualization**  
  Provides attention-based visualization for inspecting important HLA-peptide regions.

- **Broad compatibility**  
  Designed as a web service and tested on common browsers and operating systems.

## Installation

Clone the repository:

```bash
git clone https://github.com/Ascaris-Equi/fusionpm.git
cd fusionpm
```

Create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

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

### HLA Sequence

Input either:

- a full-length HLA Class I sequence
- or an HLA pseudo-sequence

Use single-letter amino acid code without spaces.

### Peptide Sequence

Input peptide sequences of 8 to 14 amino acids.

Multiple peptides can be provided with one peptide per line.

Longer sequences may be automatically segmented depending on the web-service workflow.

## Example Input

HLA sequence:

```text
MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFFTSVSRPGR...
```

Peptide sequence:

```text
SIINFEKL
```

Multiple peptides:

```text
SIINFEKL
GILGFVFTL
NLVPMVATV
```

Example files:

```text
examples/example_hla.txt
examples/example_peptides.txt
```

## Repository Structure

```text
fusionpm/
├── README.md
├── requirements.txt
├── config.py
├── data_utils.py
├── evaluate.py
├── gradio_app.py
├── model.py
├── train.py
├── train_eval.py
├── test.py
├── model_fold_0.pkl
├── model_fold_1.pkl
├── model_layer1_multihead9_fold0.pkl
├── model_layer1_multihead9_fold1.pkl
├── model_layer1_multihead9_fold2.pkl
├── model_layer1_multihead9_fold3.pkl
├── model_layer1_multihead9_fold4.pkl
├── regmodel_fold_1.pkl
├── vocab_dict.npy
├── docs/
├── examples/
└── scripts/
```

## Reproducibility

See:

```text
docs/REPRODUCIBILITY.md
```

The repository currently includes pretrained `.pkl` model files.

For full reproducibility, future releases should provide dataset splits, benchmark datasets, baseline outputs, metric scripts, random seeds, and versioned model weights.

## Limitations

Fusion-pM is intended for computational peptide-HLA-I binding prediction and candidate prioritization.

Important limitations:

- Binding prediction alone does not prove T-cell immunogenicity.
- Predicted scores require experimental validation.
- Fusion-pM should not be used as a standalone clinical decision-making tool.
- Performance may vary across HLA alleles, peptide lengths, datasets, and experimental settings.

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

For questions or feedback, please contact:

```text
fusionpm@bayvaxbio.com
```

If you find Fusion-pM helpful, please consider starring the repository.
