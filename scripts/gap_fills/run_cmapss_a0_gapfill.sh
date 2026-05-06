#!/usr/bin/env bash
# =============================================================================
# C-MAPSS A0 gap-fill — fill the 43 missing A0_frontier cells on C-MAPSS.
# =============================================================================
#
# Reviewer concern: "43 cells on C-MAPSS remain incomplete."
#
# Currently in clean_merged_final.jsonl (3,257 cells):
#   - C-MAPSS a0_frontier = 17 / 60 cells
#       d1_raw  : 5 / 5  ✅
#       d2_g_dp : 5 / 5  ✅
#       d3_l_dp : 5 / 5  ✅
#       d4_dp_ae: 2 / 5  (need seeds 44, 45, 46)
#       d5..d12 : 0 / 5  (all 8 defenses, all 5 seeds missing = 40 cells)
#   - Total missing: 3 + 40 = 43 cells
#
# This script invokes ONLY the missing (defense, seed) combinations.
# AMPDs2 / PAMAP2 / MIT-BIH / TEP A0 results stay untouched.
#
# Cells: 43
# API:   ~6,450 calls (43 × 50 prompts × 3 LLMs)
# Cost:  ~$5–10 (mixed-provider estimate)
# Time:  ~30–60 min @ 64-way concurrency
# =============================================================================

set -e

cd "$(dirname "$0")/.."  # cd to code/ root

# Optional override
: "${OUT_DIR:=results}"

# 1) Ensure .env has API keys
if [ ! -f .env ]; then
  echo "ERROR: code/.env not found. Copy .env.example to .env and fill API keys."
  exit 1
fi

# Validate at least one of the keys is set
if ! grep -qE '^(OPENAI_API_KEYS|AZURE_OPENAI_KEY|ANTHROPIC_API_KEYS|GOOGLE_API_KEYS)=[^[:space:]]+' .env; then
  echo "ERROR: At least one API key must be set in code/.env."
  echo "Looking for OPENAI_API_KEYS / AZURE_OPENAI_KEY / ANTHROPIC_API_KEYS / GOOGLE_API_KEYS"
  exit 1
fi

echo "=== C-MAPSS A0 gap-fill ==="
echo "Out: $OUT_DIR/"
echo "Started: $(date)"
echo

# Run 1: d4_dp_ae for seeds 44, 45, 46  (3 cells)
echo "--- d4_dp_ae seeds 44/45/46 (3 cells) ---"
python scripts/30_main_evaluation.py \
  --datasets cmapss \
  --defenses d4_dp_ae \
  --attacks a0_frontier \
  --tiers T1 \
  --seeds 44 45 46 \
  --n_samples 50 \
  --max_workers 64 \
  --out_dir "$OUT_DIR"

# Run 2: d5..d12 for all 5 seeds  (40 cells)
echo
echo "--- d5..d12 all seeds (40 cells) ---"
python scripts/30_main_evaluation.py \
  --datasets cmapss \
  --defenses d5_affine d6_per_seg_affine d7_ope d8_doppelganger \
             d9_kanon d10_fe_ip d11_vfae d12_ib \
  --attacks a0_frontier \
  --tiers T1 \
  --seeds 42 43 44 45 46 \
  --n_samples 50 \
  --max_workers 64 \
  --out_dir "$OUT_DIR"

echo
echo "=== Done at $(date) ==="
echo "Result file(s):"
ls -lh "$OUT_DIR"/main_eval_*cmapss*.jsonl 2>/dev/null \
  || ls -lh "$OUT_DIR"/main_eval_$(date +%Y%m%d)*.jsonl 2>/dev/null
