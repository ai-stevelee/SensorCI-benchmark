#!/usr/bin/env bash
# =============================================================================
# Q3: T2/T3/T4 generality — extend compositional analysis from PAMAP2-only
# to AMPDs2 and C-MAPSS (two non-HAR domains: smart-meter, turbofan engine).
# =============================================================================
#
# Reviewer concern: "compositional tiers T2/T4 being primarily demonstrated on
# PAMAP2 calls for at least one additional non-HAR dataset at T2/T4."
#
# This script extends T2 + T3 + T4 to AMPDs2 (smart-meter) and C-MAPSS
# (turbofan engine). It does NOT modify any T1 result.
#
# Cells: 12 defenses x 3 tiers x 2 datasets x 5 seeds = 360 new evaluation cells.
# Compute:  ~6 GPU-hours on a single RTX-class GPU (CPU-only also works, ~12h).
# API:      $0  (T2/T3/T4 do NOT use frontier LLM; they use local A4-CI features.)
#
# OUTPUT
#   results/main_eval_q3_<TIMESTAMP>_ampds2_cmapss.jsonl
# (To merge into the master clean_merged_final.jsonl, see RUN_Q3_T234_GENERALITY.md)
# =============================================================================

set -e

# Pin GPU 0 (override via CUDA_VISIBLE_DEVICES env var)
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# Where to write outputs (defaults to ./results inside code/)
: "${OUT_DIR:=results}"

cd "$(dirname "$0")/.."  # cd to code/ root

echo "=== Q3 generality: T2/T3/T4 on AMPDs2 + C-MAPSS ==="
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Out:  $OUT_DIR/"
echo "Started: $(date)"
echo

# Single combined run  (12 defenses x 3 tiers x 2 datasets x 5 seeds = 360 cells)
python scripts/30_main_evaluation.py \
  --datasets ampds2 cmapss \
  --defenses d1_raw d2_gaussian_dp d3_laplace_dp d4_dp_ae \
             d5_affine d6_per_seg_affine d7_ope d8_doppelganger \
             d9_kanon d10_fe_ip d11_vfae d12_ib \
  --attacks t2_cross_segment t3_multi_recipient t4_longitudinal \
  --tiers T2 T3 T4 \
  --seeds 42 43 44 45 46 \
  --max_workers 32 \
  --out_dir "$OUT_DIR"

echo
echo "=== Done at $(date) ==="
echo "Result file(s):"
ls -lh "$OUT_DIR"/main_eval_*ampds2*cmapss*.jsonl 2>/dev/null \
  || ls -lh "$OUT_DIR"/main_eval_$(date +%Y%m%d)*.jsonl 2>/dev/null
