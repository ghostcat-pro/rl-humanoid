#!/usr/bin/env bash
# Evaluate SD3 (Stable-Baselines3) models stored under ./outputs
# Usage examples:
#   ./run_2d.sh                       # latest SD3 checkpoint
#   ./run_2d.sh --use-final-model     # latest SD3 final_model.zip
#   ./run_2d.sh --env-id Walker2d-v5
#   ./run_2d.sh --run outputs/2025-10-28/11-33-29   # force a specific run dir
#   ./run_2d.sh --no-render --no-deterministic

set -euo pipefail

ENV_ID="Walker2d-v5"
USE_FINAL_MODEL=false
DETERMINISTIC=true
RENDER=true
RUN_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-id) ENV_ID="$2"; shift 2;;
    --use-final-model) USE_FINAL_MODEL=true; shift;;
    --no-deterministic) DETERMINISTIC=false; shift;;
    --no-render) RENDER=false; shift;;
    --run) RUN_DIR_OVERRIDE="$2"; shift 2;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

# Prefer venv python
PY=".venv/bin/python"; [[ -x "$PY" ]] || PY="python3"

# Helper: pick the newest SD3 run directory under outputs/
pick_latest_sd3_run() {
  # A directory is considered an SD3 run if:
  #  - it contains final_model.zip OR model_*.zip
  #  - it does NOT contain TorchRL artifacts (policy.pt)
  # We’ll sort candidates by mtime of their model zip.
  local root="outputs"
  mapfile -t candidates < <(
    find "$root" -type f \( -name "final_model.zip" -o -name "model_*.zip" \) -printf "%T@ %h %p\n" 2>/dev/null \
    | sort -nr | awk '{print $2}' | uniq
  )
  for d in "${candidates[@]}"; do
    if [[ -f "$d/policy.pt" ]]; then
      # looks like a TorchRL run, skip
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
  [[ -n "${RUN_DIR:-}" ]] || { echo "ERROR: No SD3 run found under ./outputs (final_model.zip or model_*.zip)."; exit 1; }
fi

# Resolve model + vecnorm
if $USE_FINAL_MODEL; then
  MODEL="$RUN_DIR/final_model.zip"
  [[ -f "$MODEL" ]] || { echo "ERROR: $MODEL not found."; exit 1; }
  VEC="$RUN_DIR/vecnormalize_final.pkl"
else
  # pick newest model_*.zip inside this run dir
  MODEL=$(find "$RUN_DIR" -maxdepth 1 -type f -name "model_*.zip" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n1 | awk '{print $2}')
  if [[ -z "${MODEL:-}" ]]; then
    # fallback to final if no model_*.zip
    MODEL="$RUN_DIR/final_model.zip"
    [[ -f "$MODEL" ]] || { echo "ERROR: No model_*.zip or final_model.zip in $RUN_DIR"; exit 1; }
  fi
  # Try to match vecnormalize_<steps>.pkl
  BASENAME=$(basename "$MODEL" .zip)
  if [[ "$BASENAME" =~ model_([0-9]+)$ ]]; then
    STEPS="${BASH_REMATCH[1]}"
    CAND="$RUN_DIR/vecnormalize_${STEPS}.pkl"
    if [[ -f "$CAND" ]]; then VEC="$CAND"; else
      VEC=$(find "$RUN_DIR" -maxdepth 1 -type f -name "vecnormalize_*.pkl" -printf "%T@ %p\n" | sort -nr | head -n1 | awk '{print $2}')
    fi
  else
    # if we fell back to final_model.zip
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

# Optional sanity: if vecnorm exists, assert it looks like SD3 (has observation_space with shape)
if [[ -n "$VEC" ]]; then
  "$PY" - <<'PYCODE' "$VEC" || { echo "ERROR: VecNormalize file does not look like SB3 vecnorm."; exit 1; }
import sys, pickle
p=sys.argv[1]
with open(p, "rb") as f:
    obj = pickle.load(f)
shape = getattr(getattr(obj, "observation_space", None), "shape", None)
print("VecNormalize obs shape:", shape)
PYCODE
fi

# Build args and run
ARGS=("evaluate.py" "--env_id" "$ENV_ID" "--model_path" "$MODEL")
[[ -n "$VEC" ]] && ARGS+=("--vecnorm_path" "$VEC")
$RENDER && ARGS+=("--render")
$DETERMINISTIC && ARGS+=("--deterministic")

echo -e "\n>>> Launching evaluation..."
echo "$PY ${ARGS[*]}"
exec "$PY" "${ARGS[@]}"
