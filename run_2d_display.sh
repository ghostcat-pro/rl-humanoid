#!/usr/bin/env bash
# Evaluation script with xvfb for visual rendering and video recording
# Usage examples:
#   ./run_2d_display.sh                       # latest SD3 checkpoint
#   ./run_2d_display.sh --use-final-model     # latest SD3 final_model.zip
#   ./run_2d_display.sh --env-id Humanoid-v5
#   ./run_2d_display.sh --run outputs/2025-10-28/21-33-51

set -euo pipefail

ENV_ID="Walker2d-v5"
USE_FINAL_MODEL=false
DETERMINISTIC=true
RUN_DIR_OVERRIDE=""
VIDEO_DIR="./videos"
EPISODES=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-id) ENV_ID="$2"; shift 2;;
    --use-final-model) USE_FINAL_MODEL=true; shift;;
    --no-deterministic) DETERMINISTIC=false; shift;;
    --run) RUN_DIR_OVERRIDE="$2"; shift 2;;
    --video-dir) VIDEO_DIR="$2"; shift 2;;
    --episodes) EPISODES="$2"; shift 2;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

# Prefer venv python
PY=".venv/bin/python"; [[ -x "$PY" ]] || PY="python3"

# Helper: pick the newest SD3 run directory under outputs/
pick_latest_sd3_run() {
  local root="outputs"
  mapfile -t candidates < <(
    find "$root" -type f \( -name "final_model.zip" -o -name "model_*.zip" \) -printf "%T@ %h %p\n" 2>/dev/null \
    | sort -nr | awk '{print $2}' | uniq
  )
  for d in "${candidates[@]}"; do
    if [[ -f "$d/policy.pt" ]]; then
      continue
    fi
    echo "$d"
    return 0
  done
  return 1
}

# Resolve run directory
if [[ -n "$RUN_DIR_OVERRIDE" ]]; then
  RUN_DIR="$RUN_DIR_OVERRIDE"
  [[ -d "$RUN_DIR" ]] || { echo "ERROR: --run '$RUN_DIR' is not a directory"; exit 1; }
else
  RUN_DIR="$(pick_latest_sd3_run || true)"
  [[ -n "${RUN_DIR:-}" ]] || { echo "ERROR: No SD3 run found under ./outputs"; exit 1; }
fi

# Resolve model + vecnorm
if $USE_FINAL_MODEL; then
  MODEL="$RUN_DIR/final_model.zip"
  [[ -f "$MODEL" ]] || { echo "ERROR: $MODEL not found."; exit 1; }
  VEC="$RUN_DIR/vecnormalize_final.pkl"
else
  MODEL=$(find "$RUN_DIR" -maxdepth 1 -type f -name "model_*.zip" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n1 | awk '{print $2}')
  if [[ -z "${MODEL:-}" ]]; then
    MODEL="$RUN_DIR/final_model.zip"
    [[ -f "$MODEL" ]] || { echo "ERROR: No model_*.zip or final_model.zip in $RUN_DIR"; exit 1; }
  fi
  BASENAME=$(basename "$MODEL" .zip)
  if [[ "$BASENAME" =~ model_([0-9]+)$ ]]; then
    STEPS="${BASH_REMATCH[1]}"
    CAND="$RUN_DIR/vecnormalize_${STEPS}.pkl"
    if [[ -f "$CAND" ]]; then VEC="$CAND"; else
      VEC=$(find "$RUN_DIR" -maxdepth 1 -type f -name "vecnormalize_*.pkl" -printf "%T@ %p\n" | sort -nr | head -n1 | awk '{print $2}')
    fi
  else
    VEC="$RUN_DIR/vecnormalize_final.pkl"
  fi
fi

echo "Run dir: $RUN_DIR"
echo "Model:   $MODEL"
if [[ -n "${VEC:-}" && -f "${VEC:-/dev/null}" ]]; then
  echo "VecNorm: $VEC"
else
  echo "WARNING: No VecNormalize file found alongside model."
  VEC=""
fi

# Check xvfb
if ! command -v xvfb-run &> /dev/null; then
  echo "ERROR: xvfb-run not found. Install with: sudo apt-get install xvfb"
  exit 1
fi

# Build args and run with xvfb
ARGS=("evaluate_headless.py" "--env_id" "$ENV_ID" "--model_path" "$MODEL" "--episodes" "$EPISODES")
[[ -n "$VEC" ]] && ARGS+=("--vecnorm_path" "$VEC")
ARGS+=("--render" "--video_dir" "$VIDEO_DIR")
$DETERMINISTIC && ARGS+=("--deterministic")

mkdir -p "$VIDEO_DIR"

echo -e "\n>>> Launching evaluation with video recording..."
echo "Videos will be saved to: $VIDEO_DIR"
echo "Command: xvfb-run -a -s \"-screen 0 1400x900x24\" $PY ${ARGS[*]}"

# Set MUJOCO_GL before running
export MUJOCO_GL=glfw

exec xvfb-run -a -s "-screen 0 1400x900x24" "$PY" "${ARGS[@]}"
