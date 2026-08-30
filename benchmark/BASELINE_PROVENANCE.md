# Baseline versions and web-server provenance

The version/access table below separates web-server results from locally
executed code and records the implementation actually used. The CSV version
(`baseline_versions.csv`) is authoritative and easier to reuse in a response
letter or supplement.

`Source` describes the origin of the method information and reported result.
`local` means that the value was generated for this benchmark, including
results returned by a web-server submission. `published` marks comparison
methods documented from their immutable publication records and not locally
scored in this workflow. The CSV is the authoritative, fully populated table.

| Baseline | Exact version or model release | Source | Access date | Official source | Reproduction note |
| --- | --- | --- | --- | --- | --- |
| Fusion-pM | `v1.0.1-iscience` | local | 2026-08-30 | [GitHub](https://github.com/Ascaris-Equi/fusionpm) | Released five-fold ensemble; threshold 0.5 |
| STMHCpan | DOI `10.1093/bib/bbad164` | published | 2026-08-30 | [Publication](https://doi.org/10.1093/bib/bbad164) | Literature comparison; not locally scored |
| TransPHLA | SPRINT `bb535c6af70403d887e0132b3029751d90de2838`; upstream `3ed2260292934170757507a71e645d0bcadfc44b` | local | 2026-08-27 | [Upstream](https://github.com/a96123155/TransPHLA-AOMP) | Five seeds; D2-selected threshold frozen for D3 |
| PISTE | `ca782c34c48d5471b47f420c21bd04d2b110bb2f` | local | 2026-08-27 | [Upstream](https://github.com/jychen01/PISTE) | Released random, unipep, and reftcr checkpoints; five bootstrap seeds |
| NetMHCpan_BA comparator | pMTnet `f9244234be1bc80a310a3dc04c02093b58e71b5e` | local | 2026-08-27 | [Upstream encoder](https://github.com/tianshilu/pMTnet) | Derived pMHC encoder plus linear head; not the official DTU implementation |
| CcBHLA | DOI `10.1101/2023.04.24.538196` | published | 2026-08-30 | [Publication](https://doi.org/10.1101/2023.04.24.538196) | Literature comparison; not locally scored |
| ESM-2 | `esm2_t6_8M_UR50D` revision `c731040fcd8d73dceaa04b0a8e6329b345b0f5df`; `esm2_t12_35M_UR50D` revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8` | local | 2026-08-27 | [Upstream](https://github.com/facebookresearch/esm) | Frozen backbones; five classifier seeds per backbone |
| UniTCR | DOI `10.1016/j.xgen.2024.100553` | published | 2026-08-30 | [Publication](https://doi.org/10.1016/j.xgen.2024.100553) | Literature comparison; not locally scored |
| DeepAIR | DOI `10.1126/sciadv.abo5128` | published | 2026-08-30 | [Publication](https://doi.org/10.1126/sciadv.abo5128) | Literature comparison; not locally scored |
| NetTCR-2.0 | DOI `10.1038/s42003-021-02610-3` | published | 2026-08-30 | [Publication](https://doi.org/10.1038/s42003-021-02610-3) | Literature comparison; not locally scored |
| pMTnet | `f9244234be1bc80a310a3dc04c02093b58e71b5e` | local | 2026-08-27 | [Upstream](https://github.com/tianshilu/pMTnet) | Released predictor; five stratified bootstrap seeds |
| IEDB ANN | ANN method 4.0 | local | 2026-08-27 | [IEDB server](https://tools.iedb.org/mhci/) | Fixed 500 nM threshold; five bootstrap seeds |
| PickPocket | PickPocket 1.1 | local | 2026-08-27 | [Official server](https://services.healthtech.dtu.dk/services/PickPocket-1.1/) | 411 jobs; unsupported peptide lengths 13-14 excluded |
| ERGO II | ERGO-II `85d320ab03ade33460be9a81ea3a51b8d37cd998`; SPRINT `bb535c6af70403d887e0132b3029751d90de2838` | local | 2026-08-27 | [Upstream](https://github.com/IdoSpringer/ERGO-II) | Five seeds; D5-selected threshold frozen for D6 and D7 |

For a web server, a version name alone is insufficient. Preserve the raw
response, request grouping, access date, supported-row count, and any dropped
rows. For a local baseline, preserve the source commit, configuration, seeds,
checkpoint-selection split, decision-threshold rule, and per-seed outputs.
