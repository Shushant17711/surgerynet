#!/usr/bin/env bash
# scripts/reproduce.sh -- end-to-end reproduction (design doc §11's own
# week-5-gate discipline): run the MWPM validation gate first and abort
# immediately if it fails, then run E2 through E9 via experiments/run_all.py
# (aggregating into results/main_table.md) and the leave-one-out ablations
# via experiments/run_ablations.py (results/ABLATIONS.md). This is exactly
# how results/main_table.md and results/ABLATIONS.md as checked into this
# repo were produced -- rerunning this script with the same arguments
# regenerates the same numbers up to Monte Carlo noise (seeds are fixed
# per-run but not pinned across machines/library versions).
#
# Real publication-scale runs take a long time (design doc §9, weeks
# 6-11) -- this script sequences the pipeline correctly, it does not make
# the underlying Monte Carlo/training fast. Pass --shots/--epochs/
# --num-seeds through to `experiments.run_all` to control scale (its own
# CLI defaults are deliberately smoke-test-sized, matching this project's
# test suite -- running this script with NO extra args runs that small,
# fast smoke-test scale, not the scale below). experiments/run_all.py also
# now defaults to the *tuned* hyperparameters from a real search
# (results/HPARAM_SEARCH.md: hidden_dim=64, num_layers=6,
# conv_type=transformer, heads=2, lr=3e-4, weight_decay=1e-4), which GPU
# support made practical to search for at all -- see LIMITATIONS.md.
#
# The results/main_table.md and results/ABLATIONS.md actually checked
# into this repo were produced with:
#   scripts/reproduce.sh --distance 3 --k 1 --p 0.002 \
#     --shots 30000 --epochs 20 --num-seeds 8
# (the ablation stage below is scaled to match).
#
# Usage:
#   scripts/reproduce.sh              # run the full pipeline at smoke-test scale
#   scripts/reproduce.sh --check      # verify every script/path and both
#                                      # Python environments exist, without
#                                      # running anything
#   scripts/reproduce.sh --shots 30000 --epochs 20 --num-seeds 8   # the real scale above
#                                      # (only affects run_all.py's stages;
#                                      # rerun run_ablations.py separately
#                                      # with its own --train-shots/
#                                      # --test-shots/--epochs to scale
#                                      # the ablation study too)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
TQEC_PYTHON="$PROJECT_ROOT/.venv-tqec/bin/python"

REQUIRED_FILES=(
  "circuits/validate.py"
  "circuits/tqec_backend.py"
  "experiments/e2_memory_baseline.py"
  "experiments/e3_zeroshot.py"
  "experiments/e4_localization.py"
  "experiments/e5_surgery_trained.py"
  "experiments/e6_config_randomization.py"
  "experiments/e7_timelike.py"
  "experiments/e8_latency.py"
  "experiments/e9_ablations.py"
  "experiments/run_all.py"
  "experiments/run_ablations.py"
  "experiments/report.py"
)

check() {
  local ok=1
  if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "MISSING: $VENV_PYTHON (main venv not set up -- see requirements.txt)"
    ok=0
  fi
  if [[ ! -x "$TQEC_PYTHON" ]]; then
    echo "MISSING: $TQEC_PYTHON (see circuits/tqec_backend.py's module docstring for setup)"
    ok=0
  fi
  for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$PROJECT_ROOT/$f" ]]; then
      echo "MISSING: $f"
      ok=0
    fi
  done
  if [[ "$ok" -eq 1 ]]; then
    echo "check OK: all scripts and both Python environments are present"
    return 0
  fi
  return 1
}

if [[ "${1:-}" == "--check" ]]; then
  check
  exit $?
fi

check

echo "=== [gate] Week-5 validation gate (design doc §5.1/§9/§11) ==="
"$VENV_PYTHON" circuits/validate.py

echo "=== [E2-E9] Full experiment pipeline ==="
"$VENV_PYTHON" -m experiments.run_all "$@"

echo "=== [ablations] Leave-one-out ablation study ==="
# Separate results file from run_all.py's own E9 stage -- writing to the
# same file would silently merge the 5 "component removed" variants into
# main_table.md's single E9 summary row (a real bug caught by inspecting
# the raw jsonl after a full run and finding 6 different configs mixed
# into what should have been one).
#
# p=0.002 (not 0.008): 0.008 is far above this ~15-round surgery circuit's
# actual threshold (~0.0035, confirmed with a real p-sweep in
# results/THRESHOLD_SWEEP.md) -- raw observable flip rate at 0.008 was
# ~0.49, indistinguishable from a coin flip, so nothing could have learned
# anything regardless of training budget. See LIMITATIONS.md.
# train/test-shots=30000, epochs=20, num-seeds=8 (not 3000/8/3): the scale
# used for the results/ files currently checked into this repo, run on an
# RTX 4060 after experiments/hparam_search.py tuned the hyperparameters
# below (results/HPARAM_SEARCH.md) -- GPU support made this scale-up
# practical where it wasn't on CPU. Smaller values still work for a quick
# local check; they just produce noisier per-variant estimates.
"$VENV_PYTHON" -m experiments.run_ablations \
  --k 1 --p 0.002 --train-shots 30000 --test-shots 30000 --epochs 20 --num-seeds 8 \
  --hidden-dim 64 --conv-type transformer --heads 2 --lr 3e-4 --weight-decay 1e-4 \
  --results-path results/e9_ablations_loo.jsonl --report-path results/ABLATIONS.md
