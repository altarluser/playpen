#!/bin/bash
#SBATCH -J playpen-eval-moe
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --chdir=/home/bbmaltar/im/playpen
#SBATCH -o /home/bbmaltar/im/playpen/logs/%x-%j.out
#SBATCH -e /home/bbmaltar/im/playpen/logs/%x-%j.err

set -euo pipefail

REPO_DIR="/home/bbmaltar/im/playpen"
SCRATCH_ROOT="/scratch/usr/bbmaltar"
HF_CACHE_ROOT="$SCRATCH_ROOT/hf-cache"
REGISTRY_OUT_DIR="$SCRATCH_ROOT/playpen-model-registries"
RESULTS_DIR="${RESULTS_DIR:-$SCRATCH_ROOT/playpen-eval/moe-only/$SLURM_JOB_ID}"

# Eval knobs (override via sbatch --export=...)
REGISTRY_PATH="${REGISTRY_PATH:-}"
MODEL_NAME="${MODEL_NAME:-}"
EVAL_SUITE="${EVAL_SUITE:-static}"
EVAL_GAME="${EVAL_GAME:-}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-}"
EVAL_SKIP_GAMEPLAY="${EVAL_SKIP_GAMEPLAY:-0}"
MOE_EVAL_ROUTING_LOGS="${MOE_EVAL_ROUTING_LOGS:-1}"
MOE_EVAL_ROUTING_FLUSH="${MOE_EVAL_ROUTING_FLUSH:-8}"

mkdir -p "$REPO_DIR/logs" "$HF_CACHE_ROOT" "$RESULTS_DIR"
cd "$REPO_DIR"

if ! type module >/dev/null 2>&1; then
  source /etc/profile || true
fi

module purge || true
module load slurm
module load NHRZIBenv
module load sw.a100.el9 || module load sw.clx.el9 || true
module load anaconda3/2023.09

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate playpen

if [ -n "${SLURM_STEP_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS%%,*}"
elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS%%,*}"
fi

export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT"
export NLTK_DATA="${NLTK_DATA:-$HF_CACHE_ROOT/nltk_data}"

: "${HF_HUB_OFFLINE:=0}"
: "${HF_DATASETS_OFFLINE:=0}"
export HF_HUB_OFFLINE HF_DATASETS_OFFLINE

touch "$REPO_DIR/clembench.log"
cat > "$REPO_DIR/game_registry.json" <<'JSON'
[{"benchmark_path": "/home/bbmaltar/im/playpen/clembench"}]
JSON

if [ -z "$REGISTRY_PATH" ]; then
  REGISTRY_PATH="$(ls -1t "$REGISTRY_OUT_DIR"/model_registry.*.sft.*.json | head -n1 || true)"
fi
if [ -z "$REGISTRY_PATH" ]; then
  echo "ERROR: Could not resolve REGISTRY_PATH (and no snapshots found in $REGISTRY_OUT_DIR)." >&2
  exit 2
fi

if [ -z "$MODEL_NAME" ]; then
  MODEL_NAME="$(
    REGISTRY_PATH="$REGISTRY_PATH" python - <<'PY'
import json
import os
from pathlib import Path

registry_path = Path(os.environ["REGISTRY_PATH"])
payload = json.loads(registry_path.read_text(encoding="utf-8"))
entries = payload if isinstance(payload, list) else [payload]

names = []
for item in entries:
    if not isinstance(item, dict):
        continue
    name = item.get("model_name")
    cfg = item.get("model_config") or {}
    if isinstance(name, str) and cfg.get("moe_enabled"):
        names.append(name)

preferred = [n for n in names if n.endswith("-sft-moe")]
pick = preferred[0] if preferred else (names[0] if names else "")
print(pick)
PY
  )"
fi

if [ -z "$MODEL_NAME" ]; then
  echo "ERROR: MODEL_NAME not provided and no moe_enabled model found in $REGISTRY_PATH." >&2
  exit 3
fi

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(python -V 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "HF cache: $HF_CACHE_ROOT"
echo "Registry: $REGISTRY_PATH"
echo "Model: $MODEL_NAME"
echo "Results: $RESULTS_DIR"
nvidia-smi || true

cmd=(python -m playpen.cli eval "$MODEL_NAME" --suite "$EVAL_SUITE" --model-registry "$REGISTRY_PATH" -r "$RESULTS_DIR")
if [ -n "$EVAL_MAX_TOKENS" ]; then
  cmd+=( -L "$EVAL_MAX_TOKENS" )
fi
if [ -n "$EVAL_GAME" ]; then
  cmd+=( -g "$EVAL_GAME" )
fi
if [ "$EVAL_SKIP_GAMEPLAY" = "1" ]; then
  cmd+=( --skip_gameplay )
fi

echo "Running eval: ${cmd[*]}"
if [ "$MOE_EVAL_ROUTING_LOGS" = "1" ]; then
  PLAYPEN_MOE_EVAL_LOG_DIR="$RESULTS_DIR/moe-routing" \
  PLAYPEN_MOE_EVAL_FLUSH_EVERY="$MOE_EVAL_ROUTING_FLUSH" \
  "${cmd[@]}"
else
  "${cmd[@]}"
fi

echo "Done."
