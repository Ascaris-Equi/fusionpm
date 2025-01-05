# fusionpm
Fusion-pM is a deep learning-based web service that predicts Class I HLA-peptide binding for immunogenicity analysis. With cross-attention and mask learning, it surpasses current baselines, offering interactive visualizations and ranking data to aid neoantigen-based cancer vaccine research.
---

## Table of Contents
1. [Background](#background)
2. [Key Features](#key-features)
3. [Usage Guide](#usage-guide)
4. [Input Example](#input-example)
5. [Installation and Deployment](#installation-and-deployment)
6. [Authors](#authors)
7. [Acknowledgments](#acknowledgments)
8. [Contact](#contact)

---

## Background
Fusion-pM integrates full-length HLA sequences with peptides (8–14 amino acids) to predict binding affinity and immunogenicity. By utilizing cross-attention mechanisms, the model can focus on critical regions—such as anchor residues in peptides and the MHC binding groove—making the predictions both accurate and explainable.

---

## Key Features
- **High Accuracy**  
  Advanced transformer-based architectures and mask learning for state-of-the-art performance.
- **Interactive Visualization**  
  Provides attention maps that highlight key contact residues between HLA and peptide.
- **Broad Compatibility**  
  Free to use, no login required, tested on major browsers (Chrome, Firefox, Edge) and operating systems.
- **Ranked Outputs**  
  Generates a prioritized list of candidate peptides, aiding in neoantigen-focused research.

---

## Usage Guide
1. **HLA Sequence**  
   - Full-length HLA Class I sequence (single-letter code, no spaces)  
   - Or a pseudo-sequence

2. **Peptide Sequence**  
   - Amino acid sequence for peptides (8–14 residues)  
   - Longer sequences will be automatically segmented

**Output**  
The service ranks peptide binding affinity from highest to lowest, offering interactive plots and detailed tables for further analysis.

---

## Input Example
```text
HLA Sequence:
MAVMAPRTLVLLLSGALALTQTWAGSHSMRYFFTSVSRPGR... (Full or partial)

Peptide Sequence:
SIINFEKL
or
MACTGPSLPSAFDILGAAGQDKLLYLKHKLKTPRPGCQGQDLLHAMVLLKLGQETEARISLEALKADAVARLVARQWAGVDSTEDPEEPPDVSWAVARLYHLLAEEKLCPASLRDVAYQEAVRTLSSRDDHRLGELQDEARNRCGWDIAGDPGSIRTLQSNLGCLPPSSALPSGTRSLPRPIDGVSDWSQGCSLRSTGSPASLASNLEISQSPTMPFLSLHRSPHGPSKLCDDPQASLVPEPVPGGCQEPEEMSWPPSGEIASPPELPSSPPPGLPEVAPDATSTGLPDTPAAPETSTNYPVECTEGSAGPQSLPLPILEPVKNPCSVKDQTPLQLSVEDTTSPNTKPCPPTPTTPETSPPPPPPPPSSTPCSAHLTPSSLFPSSLESSSEQKFYNFVILHARADEHIALRVREKLEALGVPDGATFCEDFQVPGRGELSCLQDAIDHSAFIILLLTSNFDCRLSLHQVNQAMMSNLTRQGSPDCVIPFLPLESSPAQLSSDTASLLSGLVRLDEHSQIFARKVANTFKPHRLQARKAMWRKEQDTRALREQSQHLDGERMQAAALNAAYSAYLQSYLSYQAQMEQLQVAFGSHMSFGTGAPYGARMPFGGQVPLGAPPPFPTWPGCPQPPPLHAWQAGTPPPPSPQPAAFPQSLPFPQSPAFPTASPAPPQSPGLQPLIIHHAQMVQLGLNNHMWNQRGSQAPEDKTQEAE
(You can input multiple peptides, one per line.)
```
Installation and Deployment

(To be add)
Authors

    Jiahao Ma, BayVax Biotech Limited
    Hongzong Li, BayVax Biotech Limited
    Xiaoping Su, Wenzhou Medical University
    Zhenzhai Cai, Second Affiliated Hospital of Wenzhou Medical University
    Ye-Fan Hu, BayVax Biotech Limited
    Yifan Chen, Hong Kong Baptist University
    Jian-Dong Huang, The University of Hong Kong

Acknowledgments

Supported by the National Key Research and Development Program of China (2021YFA0910700), the Health and Medical Research Fund, and other investors and sponsors.
Contact

If you have any questions or feedback, please email us at fusionpm@bayvaxbio.com.

If you find Fusion-pM helpful, please give us a Star!
Happy researching!

