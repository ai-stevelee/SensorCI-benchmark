# SensorCI — Anonymous Submission

> **When LLMs Read Your Sensors: A Contextual Integrity Benchmark for Privacy in Industrial Time-Series**
> *Anonymous Authors — NeurIPS 2026 Datasets & Benchmarks Track submission*

This repository contains the full reproducibility artefact for the SensorCI benchmark.

## Repository layout

```
submission/
├── paper/
│   ├── draft_paper_v1.tex          # main paper (compile with NeurIPS 2026 style)
│   └── figures/                    # PNG fallbacks (paper builds figures via TikZ)
│
├── sensorci/                       # Python package (importable)
│   ├── core/                       # dataset, persona, segment, role abstractions
│   ├── datasets/                   # 5 dataset loaders (TEP, C-MAPSS, AMPDs2, PAMAP2, MIT-BIH)
│   ├── defenses/                   # 12 defense baselines (D1–D12)
│   ├── attacks/                    # 8 attacks (A0 frontier-LLM, A1–A7 specialised)
│   ├── encodings/                  # raw_stats / sax / compact_numeric encoders
│   ├── judge/                      # LLM-as-Judge generator + voter (curation)
│   ├── api_pool/                   # multi-key concurrent API dispatcher (anonymous proxy)
│   ├── metrics/                    # S_P1, S_P2, S_P3-CI, S_P4 + composites
│   └── viz/                        # plotting utilities
│
├── scripts/                        # all runnable entrypoints
│   ├── 00_check_env.py             # validate API keys + dependencies
│   ├── 01_download_tep.py          # download TEP dataset
│   ├── 51_download_datasets.py     # download remaining 4 datasets
│   ├── a0_attack/                  # frontier-LLM attack pipeline (Stage A0)
│   │   ├── 10_pilot_a0.py
│   │   ├── 11_pilot_algebra.py
│   │   ├── 12_ablation_a0.py
│   │   └── 13_contamination_probe.py
│   ├── curation/                   # multi-LLM voting (queries + sharing matrix)
│   │   ├── 14_judge_robustness.py
│   │   ├── 15_coverage_check.py
│   │   ├── 20_judge_query_bank.py
│   │   └── 21_judge_role_matrix.py
│   ├── main_eval/                  # the headline evaluation
│   │   ├── 30_main_evaluation.py   # main 3,257-cell loop
│   │   ├── 40_aggregate_results.py # bootstrap, Wilcoxon, Cliff's δ
│   │   ├── 41_make_paper_artifacts.py  # build figures + tables
│   │   └── 50_launch_full_experiment.py # one-shot orchestrator
│   ├── rebuttal_q1/                # Q1: SP1^supp scope-restricted utility
│   │   ├── run_q1_supp.py
│   │   └── run_q3_t234_generality.sh   # Q3: T2/T4 on AMPDs2 + C-MAPSS (server)
│   ├── rebuttal_q4/                # Q4: PAMAP2 T3 sensitivity ablations
│   │   └── run_q4_t3_ablation.py
│   ├── gap_fills/                  # missing-cell fill scripts
│   │   └── run_cmapss_a0_gapfill.sh
│   └── figures/                    # standalone figure regeneration scripts
│       ├── _regen_pareto.py
│       ├── _regen_figs.py          # radar grid (PNG fallback)
│       └── _fit_C1C2.py            # Theorem 2 OLS fits per dataset
│
├── configs/
│   ├── paper_full.yaml             # config for full reproduction
│   └── pilot_tep.yaml              # smaller pilot config
│
├── results/                        # released aggregated outputs
│   ├── aggregated_summary.json     # bootstrap CIs, Wilcoxon, Cliff's δ, per-defense scores
│   ├── clean_merged_final.jsonl    # 3,257 evaluation cells (one row per cell)
│   └── paper_artifacts/            # tables/, figures/ used in the paper
│       ├── tables/
│       └── figures/
│
├── docs/
│   ├── HOWTO_REPRODUCE.md          # step-by-step reproduction
│   └── EXTENSION.md                # adding new datasets/defenses
│
├── pyproject.toml                  # dependencies + package metadata
├── .env.example                    # API key template (DO NOT commit your .env)
├── .gitignore
└── README.md                       # this file
```

## Quick start

### 1. Install

```bash
pip install -e .                          # editable install of the sensorci/ package
cp .env.example .env                       # then fill in your API keys
python scripts/00_check_env.py             # verify environment
```

### 2. Download datasets

```bash
python scripts/01_download_tep.py          # ~150 MB
python scripts/51_download_datasets.py     # ~3 GB total (CMAPSS + AMPDs2 + PAMAP2 + MIT-BIH)
```

### 3. Reproduce headline result (no API keys, T1-only)

```bash
python scripts/main_eval/30_main_evaluation.py \
    --datasets tep cmapss ampds2 pamap2 mitbih \
    --defenses d1_raw d2_gaussian_dp d3_laplace_dp d4_dp_ae \
               d5_affine d6_per_seg_affine d7_ope d8_doppelganger \
               d9_kanon d10_fe_ip d11_vfae d12_ib \
    --attacks a1_distinguish a2_linear_recon a3_neural_recon \
              a4_ci_linkage a5_stat_tests a6_algebra a7_mia_ts \
    --tiers T1 \
    --seeds 42 43 44 45 46 \
    --max_workers 32 --out_dir results/

python scripts/main_eval/40_aggregate_results.py results/
python scripts/main_eval/41_make_paper_artifacts.py results/
```

### 4. (Optional) Add A0 frontier-LLM attack

Set `OPENAI_API_KEYS`, `ANTHROPIC_API_KEYS`, `GOOGLE_API_KEYS` in `.env`, then add `a0_frontier` to the `--attacks` list above. Cost: ~$80–$150 for the full 257 A0 cells.

Three replication tiers documented in `docs/HOWTO_REPRODUCE.md`:
- **T1-only, no API**: ~4–6 GPU-hours, $0
- **T1+T2–T4, no API**: add ≤1 GPU-hour for cached tier aggregations
- **Full + A0**: add $100–$150 in API spend, ~5h wall-time at 64-way concurrency

## Reproducing the paper figures

```bash
# Pareto frontier (cross-dataset)
python scripts/figures/_regen_pareto.py

# Per-defense Quadrilemma radar (12 panels)
python scripts/figures/_regen_figs.py

# Theorem 2 OLS fits (C_1, C_2 per dataset)
python scripts/figures/_fit_C1C2.py
```

The paper TeX (`paper/draft_paper_v1.tex`) draws both Pareto and radar figures natively in TikZ; the standalone PNG regeneration scripts above are an alternative for non-LaTeX consumers.

## Datasets

All five datasets are publicly available (see `docs/HOWTO_REPRODUCE.md` §Datasets for direct links). The SensorCI curation layer (persona schema, sharing matrices Φ_D, query banks, audit logs) is released under the MIT License; raw signals are referenced via their original sources to preserve upstream licensing.

| Dataset            | Persona unit  | License             | Direct download                                     |
|--------------------|---------------|---------------------|-----------------------------------------------------|
| Tennessee Eastman  | Sim. run      | Public domain       | Harvard Dataverse (Rieth ext.)                      |
| NASA C-MAPSS       | Engine unit   | Public domain       | NASA PCoE                                           |
| AMPDs2             | Household     | CC BY 4.0           | https://dataverse.harvard.edu/dataset.xhtml?id=ampds2 |
| PAMAP2             | Subject       | UCI public          | UCI ML Repository                                   |
| MIT-BIH Arrhythmia | Patient       | ODC-By v1.0         | PhysioNet                                           |

## License

- **Curation layer** (this repo): MIT License — see `LICENSE`
- **Upstream raw signals**: respective dataset licenses (see table above)

## Citation

> *Citation placeholder — Anonymous authors, NeurIPS 2026 Datasets & Benchmarks Track submission. Will be updated with the published reference upon acceptance.*

## Anonymity statement

This repository is prepared for double-blind review. No author or institutional identifying information is included. The `sensorci/api_pool/` module routes calls through a generic API hub configurable via environment variables; the implementation is portable to any OpenAI-compatible provider (a list of public providers is provided in `.env.example`).
