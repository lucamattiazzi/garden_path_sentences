#!/usr/bin/env bash
#
# End-to-end GPS pipeline for one model folder.
#
#   bash cloud/run_all.sh [MODEL_DIR] [NULL_SAMPLES]
#   bash cloud/run_all.sh gemma3_4b            # local
#   bash cloud/run_all.sh gemma3_27b 60        # default
#
# Backend-agnostic: get_device() picks cuda in the cloud and mps on a Mac.
#
# With AUTO_CANDIDATES=1 (default) the head-level steps 03/04/06/12 are chained
# automatically — cloud/pick_candidates.py reads the 02/03/05/06 sweeps and
# selects the same kind of candidates a human would (most negative effect_drop
# for the surprise heads, largest vc_gps_inflation for the comprehension layers
# and heads, all filtered on control_inflation). Step 06 is the direct
# reanalysis-head hunt: a per-head sweep on the comprehension metric inside the
# top step-05 layers, independent of the Δ-surprisal track. Set
# AUTO_CANDIDATES=0 to run only the argument-free core and get the manual
# follow-up instructions instead.
#
# Knobs: TOP_LAYERS=3  TOP_HEADS=4  TOP_VCLAYERS=2  MAX_CONTROL=0.5  NULL_SAMPLES_VC=300
set -euo pipefail

MODEL_DIR="${1:-gemma3_27b}"
NULL_SAMPLES="${2:-60}"

PY="${PYTHON_RUN:-uv run python}"                       # step scripts need the deps
HERE="$(cd "$(dirname "$0")" && pwd)"
PICK="python3 ${HERE}/pick_candidates.py"              # stdlib only, no venv needed
AUTO="${AUTO_CANDIDATES:-1}"
TOP_LAYERS="${TOP_LAYERS:-3}"
TOP_HEADS="${TOP_HEADS:-4}"
TOP_VCLAYERS="${TOP_VCLAYERS:-2}"
MAX_CONTROL="${MAX_CONTROL:-0.5}"

cd "${HERE}/.."   # repo root
LOGS="${MODEL_DIR}/logs"
[ -d "${MODEL_DIR}/scripts" ] || { echo "no such model folder: ${MODEL_DIR}" >&2; exit 1; }

hdr() { echo; echo "==================== $1 ===================="; }
step() {
  hdr "$1"
  if [ "${SKIP_DONE:-0}" = "1" ]; then
    local out="$1"
    case "$1" in
      03_head_sweep|06_predict_head_sweep) out="$1_L${2:-}" ;;
    esac
    if [ -f "${LOGS}/${out}.json" ]; then
      echo "already done (${LOGS}/${out}.json) — skipping"
      return
    fi
  fi
  ${PY} "${MODEL_DIR}/scripts/$1.py" "${@:2}"
}

echo "### GPS pipeline · ${MODEL_DIR} · auto-candidates=${AUTO} ###"
date -u +"start %Y-%m-%dT%H:%M:%SZ"

# ── surprise track ─────────────────────────────────────────────────────────
step 01_baseline
step 02_layer_sweep

HEADS=""
if [ "$AUTO" = "1" ]; then
  LAYERS=$($PICK layers "${LOGS}/02_layer_sweep.json" --top "$TOP_LAYERS" --max-control "$MAX_CONTROL")
  echo ">> candidate layers: ${LAYERS:-<none>}"
  for L in $LAYERS; do step "03_head_sweep" "$L"; done
  if [ -n "$LAYERS" ]; then
    FILES=""; for L in $LAYERS; do FILES="$FILES ${LOGS}/03_head_sweep_L${L}.json"; done
    HEADS=$($PICK heads $FILES --top "$TOP_HEADS" --max-control "$MAX_CONTROL")
    echo ">> candidate heads: ${HEADS:-<none>}"
    [ -n "$HEADS" ] && step "04_predict" $HEADS
  fi
fi

# ── comprehension track + diagnostics ──────────────────────────────────────
step 05_predict_layer_sweep

# Direct hunt for reanalysis heads: head sweep on the comprehension metric
# inside the layers step 05 flagged, no detour through the Δ-surprisal track.
VCLAYERS=""
VCHEADS=""
if [ "$AUTO" = "1" ]; then
  VCLAYERS=$($PICK vclayers "${LOGS}/05_predict_layer_sweep.json" --top "$TOP_VCLAYERS" --max-control "$MAX_CONTROL")
  echo ">> comprehension layers: ${VCLAYERS:-<none>}"
  for L in $VCLAYERS; do step "06_predict_head_sweep" "$L"; done
  if [ -n "$VCLAYERS" ]; then
    FILES=""; for L in $VCLAYERS; do FILES="$FILES ${LOGS}/06_predict_head_sweep_L${L}.json"; done
    VCHEADS=$($PICK vcheads $FILES --top "$TOP_HEADS" --max-control "$MAX_CONTROL")
    echo ">> comprehension heads: ${VCHEADS:-<none>}"
  fi
fi

step 07_lexical_baseline
step 08_novel_lexical_baseline
step 09_novel_predict
step 10_attention_patterns
step 11_patching
step 13_null_distribution "$NULL_SAMPLES"
step 14_null_distribution_vc "${NULL_SAMPLES_VC:-300}"

# ── mean-ablation replication (needs surprise heads + comprehension layers) ─
if [ "$AUTO" = "1" ] && [ -n "$HEADS" ]; then
  step "12_mean_ablation" $HEADS ${VCLAYERS:+-- $VCLAYERS}
fi

date -u +"end %Y-%m-%dT%H:%M:%SZ"
echo; echo "### done — results in ${LOGS}/*.json ###"

if [ "$AUTO" != "1" ]; then
  cat <<EOF

AUTO_CANDIDATES=0 → the head-level steps were skipped. Run them by hand:
  uv run python ${MODEL_DIR}/scripts/03_head_sweep.py <LAYER>        # from logs/02
  uv run python ${MODEL_DIR}/scripts/04_predict.py <L:H> ...         # from the 03 outputs
  uv run python ${MODEL_DIR}/scripts/06_predict_head_sweep.py <LAYER>  # from logs/05
  uv run python ${MODEL_DIR}/scripts/12_mean_ablation.py <L:H> ... -- <LAYER> ...
EOF
fi
