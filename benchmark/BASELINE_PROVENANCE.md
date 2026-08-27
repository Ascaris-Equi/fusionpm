# Baseline versions and web-server provenance

The version/access table below separates web-server results from locally
executed code and records the implementation actually used. The CSV version
(`baseline_versions.csv`) is authoritative and easier to reuse in a response
letter or supplement.

| Baseline | Implementation | Exact version or model release | Access date | Official source | Reproduction note |
| --- | --- | --- | --- | --- | --- |
| Fusion-pM | This repository | `8d386dd6fabb6a3a554bccfbc660aaf5b10f891c` | 2026-08-27 | [GitHub](https://github.com/Ascaris-Equi/fusionpm) | Released five-fold ensemble; threshold 0.5 |
| PickPocket | DTU Health Tech webserver | PickPocket 1.1 | 2026-08-27 | [Official server](https://services.healthtech.dtu.dk/services/PickPocket-1.1/) | 411 archived jobs; lengths 13–14 are unsupported and excluded from its metrics |
| TransPHLA | SPRINT adapter | Benchmark snapshot `960ccddb555deacea513cd9fa0f5d701e031f35c` | 2026-08-27 | [Upstream](https://github.com/a96123155/TransPHLA-AOMP) | Five seeds; threshold selected on D2 and frozen for D3 |
| pMTnet pMHC encoder + linear head | Frozen upstream encoder plus trained 60-to-1 head | pMTnet `f9244234be1bc80a310a3dc04c02093b58e71b5e` | 2026-08-27 | [Upstream](https://github.com/tianshilu/pMTnet) | This derived comparator is not official NetMHCpan and must not be labeled as NetMHCpan |
| ESM-2 | Frozen ESM-2 plus trained classifier | `esm2_t12_35M_UR50D`, revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8` | 2026-08-27 | [Upstream](https://github.com/facebookresearch/esm) | One seed; frozen backbone |

For a web server, a version name alone is insufficient. Preserve the raw
response, request grouping, access date, supported-row count, and any dropped
rows. For a local baseline, preserve the source commit, configuration, seeds,
checkpoint-selection split, decision-threshold rule, and per-seed outputs.
