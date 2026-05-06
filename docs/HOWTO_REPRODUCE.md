# How to reproduce the SensorCI paper

This document walks you through reproducing the headline numbers in the paper at three replication tiers, ordered by cost.

## Hardware requirements

| Tier                  | GPU                  | CPU       | RAM   | Disk  | Wall-time   | API cost     |
|-----------------------|----------------------|-----------|-------|-------|-------------|--------------|
| T1-only, no API       | 1× RTX-class         | 32 cores  | 32 GB | 20 GB | 4–6 h       | $0           |
| T1+T2–T4, no API      | 1× RTX-class         | 32 cores  | 32 GB | 20 GB | 5–7 h       | $0           |
| Full + A0             | 1× RTX-class         | 32 cores  | 32 GB | 20 GB | ~5 h        | $100–$150    |

`A_0` runs in parallel via 100-API-key concurrent dispatcher (max 64 in-flight).

## Step 1 — install

```bash
git clone <this-repo>
cd submission
pip install -e .                            # editable install
python scripts/00_check_env.py              # verify
```

`00_check_env.py` checks: Python ≥3.11, PyTorch with CUDA, dataset directory, API keys (if A0 enabled).

## Step 2 — download datasets

```bash
python scripts/01_download_tep.py           # TEP only (~150 MB)
python scripts/51_download_datasets.py      # all 5 datasets (~3 GB)
```

Downloads land in `data/` (configurable via `SENSORCI_DATA_DIR` env var).

| Dataset           | Direct source                                                |
|-------------------|--------------------------------------------------------------|
| Tennessee Eastman | https://dataverse.harvard.edu/dataset.xhtml?id=tep_extended  |
| NASA C-MAPSS      | https://www.nasa.gov/intelligent-systems-division/datasets   |
| AMPDs2            | https://dataverse.harvard.edu/dataset.xhtml?id=ampds2        |
| PAMAP2            | https://archive.ics.uci.edu/dataset/231/pamap2               |
| MIT-BIH           | https://physionet.org/content/mitdb/1.0.0/                   |

## Step 3a — reproduce T1 composite (no API, ~5 h)

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
    --max_workers 32 \
    --out_dir results/

python scripts/main_eval/40_aggregate_results.py results/
python scripts/main_eval/41_make_paper_artifacts.py results/
```

Expected outputs in `results/`:
- `clean_merged_final.jsonl` — 3,257 evaluation cells (12 defenses × 5 datasets × 5 seeds × 7 attacks ≈ 2,100 cells, plus T2/T3/T4 derived rows)
- `aggregated_summary.json` — bootstrap 95% CIs, paired Wilcoxon p-values, Cliff's δ, Friedman χ² = 30.10 (p = 0.0015)
- `paper_artifacts/tables/composite_T1.tex` — Table 5 in paper

Match: VFAE-TS (D11) ranked 1st with HM = 0.77; OPE/FE-IP at HM = 0.00.

## Step 3b — add T2/T3/T4 compositional tiers (no API, +1 h)

Add `--tiers T1 T2 T3 T4` to the `30_main_evaluation.py` command. T2/T3/T4 reuse cached A4-CI features; the additional ~1 h is dominated by aggregation.

## Step 3c — full reproduction with A0 frontier-LLM attack

Set in `.env`:
```
OPENAI_API_KEYS=sk-xxx,sk-yyy,...   # comma-separated for multi-key concurrency
ANTHROPIC_API_KEYS=sk-ant-xxx,...
GOOGLE_API_KEYS=AIza-xxx,...
```

Add `a0_frontier` to `--attacks`:
```bash
python scripts/main_eval/30_main_evaluation.py \
    --attacks a0_frontier a1_distinguish ... \
    --max_workers 64 \
    --out_dir results/
```

Cost ≈ $100–$150 for 257 A0 cells × 50 prompts × 3 models = 38,550 calls.

## Q-rebuttal experiments

### Q1: scope-restricted SP1^supp (OPE / FE-IP utility within native scope)
```bash
python scripts/rebuttal_q1/run_q1_supp.py
```
~25 min, no API. Outputs `results/q1_supp_summary.json`.

### Q3: T2/T4 generality on AMPDs2 + C-MAPSS (server)
```bash
bash scripts/rebuttal_q1/run_q3_t234_generality.sh
```
~6 GPU-h on a server. Pending; stay tuned to camera-ready.

### Q4: PAMAP2 T3 sensitivity ablations
```bash
python scripts/rebuttal_q4/run_q4_t3_ablation.py
```
~9 min, single CPU + 1 GPU. Outputs `results/q4_t3_ablation_summary.json`. All 180 cells return max-adv = 0 (intrinsic small-N null on PAMAP2).

## Reproducing paper figures

```bash
# TikZ-based (compile via the paper TeX directly)
cd paper/
pdflatex draft_paper_v1.tex      # twice for cross-refs

# OR — standalone PNG regeneration:
python scripts/figures/_regen_pareto.py     # Pareto frontier (Fig. 1)
python scripts/figures/_regen_figs.py       # 12-panel radar (App. J)
python scripts/figures/_fit_C1C2.py         # Theorem 2 OLS fits (App. A.1.2)
```

## Troubleshooting

- **CUDA mismatch**: `pyproject.toml` pins PyTorch 2.x. If your CUDA differs, override the install via `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
- **API rate limits**: lower `max_workers` to 32 or 16 in `30_main_evaluation.py`.
- **synthetic-data fallback**: should never trigger; the loader raises `RuntimeError` if signal std < 1e-3 to prevent silent fallback.
