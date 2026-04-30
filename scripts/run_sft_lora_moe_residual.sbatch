#!/bin/bash
#SBATCH -J playpen-sft-moe-res
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --chdir=/home/bbmaltar/im/playpen
#SBATCH -o /home/bbmaltar/im/playpen/logs/%x-%j.out
#SBATCH -e /home/bbmaltar/im/playpen/logs/%x-%j.err

set -euo pipefail

REPO_DIR="/home/bbmaltar/im/playpen"
BASE_REGISTRY="$REPO_DIR/model_registry.json"
SCRATCH_ROOT="/scratch/usr/bbmaltar"
HF_CACHE_ROOT="$SCRATCH_ROOT/hf-cache"
LEARNER_MODEL="${LEARNER_MODEL:-llama3-8b}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-moe-residual-v1}"
ADAPTER_BASE_ROOT="${ADAPTER_BASE_ROOT:-$SCRATCH_ROOT/playpen-adapters}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$ADAPTER_BASE_ROOT/${LEARNER_MODEL}-adapters}"
REGISTRY_OUT_DIR="${REGISTRY_OUT_DIR:-$SCRATCH_ROOT/playpen-model-registries/$EXPERIMENT_TAG}"
EVAL_ROOT="${EVAL_ROOT:-$SCRATCH_ROOT/playpen-eval/$EXPERIMENT_TAG/$SLURM_JOB_ID}"
RUNTIME_REGISTRY="$REGISTRY_OUT_DIR/model_registry.runtime.${SLURM_JOB_ID}.json"

# Training/eval knobs (override via sbatch --export=...)
TRAINER_SCRIPT="${TRAINER_SCRIPT:-examples/trl/sft_trainer_lora_moe_residual.py}"
MOE_CONFIG_PATH="${MOE_CONFIG_PATH:-$REPO_DIR/examples/trl/sft_trainer_lora_moe_residual.yaml}"
TRAIN_TARGETS="${TRAIN_TARGETS:-all-games}"
PER_GAME_MAX_SAMPLES="${PER_GAME_MAX_SAMPLES:-620}"
EVAL_SUITE="${EVAL_SUITE:-all}"
EVAL_GAME="${EVAL_GAME:-}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-}"
EVAL_SKIP_GAMEPLAY="${EVAL_SKIP_GAMEPLAY:-0}"
PLAYPEN_EVAL_INSTANCES_PATH="${PLAYPEN_EVAL_INSTANCES_PATH:-/scratch/usr/bbmaltar/datasets/playpen-data-v3/instances-validation}"
PLAYPEN_EVAL_INSTANCES_STATIC_PATH="${PLAYPEN_EVAL_INSTANCES_STATIC_PATH:-/scratch/usr/bbmaltar/datasets/playpen-data-v3/instances-static-validation}"
MOE_EVAL_ROUTING_LOGS="${MOE_EVAL_ROUTING_LOGS:-1}"
MOE_EVAL_ROUTING_FLUSH="${MOE_EVAL_ROUTING_FLUSH:-8}"
PLAYPEN_MERGE_PREFER_MANUAL="${PLAYPEN_MERGE_PREFER_MANUAL:-weight_averaging,task_arithmetic}"
PLAYPEN_MERGE_ALLOW_FALLBACK="${PLAYPEN_MERGE_ALLOW_FALLBACK:-1}"
PLAYPEN_GENERATION_TIMEOUT_SECONDS="${PLAYPEN_GENERATION_TIMEOUT_SECONDS:-300}"
PLAYPEN_INSTANCE_CALL_TIMEOUT_SECONDS="${PLAYPEN_INSTANCE_CALL_TIMEOUT_SECONDS:-300}"
PLAYPEN_INSTANCE_HARD_TIMEOUT_SECONDS="${PLAYPEN_INSTANCE_HARD_TIMEOUT_SECONDS:-300}"
PLAYPEN_WANDB="${PLAYPEN_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-llama3-sft-moe-residual}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DIR="${WANDB_DIR:-$SCRATCH_ROOT/wandb}"

mkdir -p "$REPO_DIR/logs" "$HF_CACHE_ROOT" "$ADAPTER_BASE_ROOT" "$ADAPTER_ROOT" "$REGISTRY_OUT_DIR" "$EVAL_ROOT" "$WANDB_DIR"
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

auto_fix_cuda_visible_devices() {
  if python - <<'PY' >/dev/null 2>&1
import torch, sys
sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
  then
    return 0
  fi
  local raw="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-}}"
  local first="${raw%%,*}"
  local idx=""
  if [ -n "$first" ]; then
    case "$first" in
      GPU-*|MIG-*) idx="$first" ;;
      *) idx="$(printf '%s' "$first" | sed -nE 's/[^0-9]*([0-9]+).*/\1/p')" ;;
    esac
  fi
  local candidates=()
  [ -n "$idx" ] && candidates+=("$idx")
  [ -n "$first" ] && candidates+=("$first")
  candidates+=("0")
  local cand
  for cand in "${candidates[@]}"; do
    if CUDA_VISIBLE_DEVICES="$cand" python - <<'PY' >/dev/null 2>&1
import torch, sys
sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
    then
      export CUDA_VISIBLE_DEVICES="$cand"
      return 0
    fi
  done
  echo "ERROR: Could not find a working CUDA_VISIBLE_DEVICES mapping." >&2
  return 1
}
auto_fix_cuda_visible_devices

export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT"
export NLTK_DATA="${NLTK_DATA:-$HF_CACHE_ROOT/nltk_data}"

: "${HF_HUB_OFFLINE:=1}"
: "${HF_DATASETS_OFFLINE:=1}"
HF_LOCAL_SNAPSHOT_FALLBACK="${HF_LOCAL_SNAPSHOT_FALLBACK:-/scratch/usr/bbmaltar/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
export HF_HUB_OFFLINE HF_DATASETS_OFFLINE
export HF_LOCAL_SNAPSHOT_FALLBACK

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(python -V 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "HF cache: $HF_CACHE_ROOT"
echo "NLTK_DATA=$NLTK_DATA"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
nvidia-smi || true
python - <<'PY'
import os
try:
    import torch
    print(
        f"torch.cuda.is_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()} "
        f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"torch CUDA probe failed: {exc}")
PY

# Avoid duplicate game discovery from both cwd + game_registry.json.
touch "$REPO_DIR/clembench.log"
rm -f "$REPO_DIR/game_registry.json"

mkdir -p "$REPO_DIR/models"
if [ -e "$REPO_DIR/models/sft+lora" ] && [ ! -L "$REPO_DIR/models/sft+lora" ]; then
  echo "ERROR: $REPO_DIR/models/sft+lora exists and is not a symlink. Move it manually, then rerun." >&2
  exit 2
fi
ln -sfn "$ADAPTER_BASE_ROOT" "$REPO_DIR/models/sft+lora"

export PLAYPEN_TRAIN_TARGETS="$TRAIN_TARGETS"
export PLAYPEN_PER_GAME_MAX_SAMPLES="$PER_GAME_MAX_SAMPLES"
export PLAYPEN_MODEL_REGISTRY_DIR="$REGISTRY_OUT_DIR"
export PLAYPEN_SOURCE_MODEL_REGISTRY="$RUNTIME_REGISTRY"
export PLAYPEN_SFT_MOE_RESIDUAL_CONFIG="$MOE_CONFIG_PATH"
export PLAYPEN_ADAPTER_ROOT="$ADAPTER_ROOT"
export PLAYPEN_MERGE_PREFER_MANUAL
export PLAYPEN_MERGE_ALLOW_FALLBACK
export PLAYPEN_GENERATION_TIMEOUT_SECONDS
export PLAYPEN_INSTANCE_CALL_TIMEOUT_SECONDS
export PLAYPEN_INSTANCE_HARD_TIMEOUT_SECONDS
export PLAYPEN_WANDB
export WANDB_PROJECT
export WANDB_MODE
export WANDB_DIR
if [ -n "$WANDB_ENTITY" ]; then
  export WANDB_ENTITY
fi

BASE_REGISTRY="$BASE_REGISTRY" \
RUNTIME_REGISTRY="$RUNTIME_REGISTRY" \
LEARNER_MODEL="$LEARNER_MODEL" \
HF_CACHE_ROOT="$HF_CACHE_ROOT" \
python - <<'PY'
import json
import os
from pathlib import Path

base_registry = Path(os.environ["BASE_REGISTRY"])
runtime_registry = Path(os.environ["RUNTIME_REGISTRY"])
learner_model = os.environ["LEARNER_MODEL"]
hf_cache_root = Path(os.environ["HF_CACHE_ROOT"])
explicit_snapshot = (os.getenv("HF_LOCAL_SNAPSHOT_PATH") or "").strip()
fallback_snapshot = (os.getenv("HF_LOCAL_SNAPSHOT_FALLBACK") or "").strip()

payload = json.loads(base_registry.read_text(encoding="utf-8"))
entries = payload if isinstance(payload, list) else [payload]

def resolve_snapshot(repo_id: str) -> str:
    if explicit_snapshot:
        p = Path(explicit_snapshot).expanduser()
        if p.exists() and p.is_dir():
            return str(p)
    if fallback_snapshot:
        p = Path(fallback_snapshot).expanduser()
        if p.exists() and p.is_dir():
            return str(p)
    candidates = [repo_id]
    try:
        org, model = repo_id.split("/", 1)
        variants = [model]
        if model.startswith("Meta-"):
            variants.append(model[len("Meta-"):])
        elif not model.startswith("meta-"):
            variants.append("Meta-" + model)
        if model.startswith("meta-"):
            variants.append(model[len("meta-"):])
        variants = [v for i, v in enumerate(variants) if v and v not in variants[:i]]
        candidates.extend([f"{org}/{v}" for v in variants])
    except Exception:
        pass
    for cand in candidates:
        pattern = f"models--{cand.replace('/', '--')}/snapshots/*"
        snaps = sorted([p for p in hf_cache_root.glob(pattern) if p.is_dir()])
        if snaps:
            return str(snaps[-1])
    return ""

updated = False
for entry in entries:
    if not isinstance(entry, dict):
        continue
    if str(entry.get("model_name")) != learner_model:
        continue
    hf_id = entry.get("huggingface_id")
    if not isinstance(hf_id, str) or not hf_id:
        continue
    if Path(hf_id).expanduser().exists():
        updated = True
        break
    if "/" not in hf_id:
        continue
    snap = resolve_snapshot(hf_id)
    if not snap:
        continue
    entry["huggingface_id"] = snap
    model_config = entry.get("model_config")
    if not isinstance(model_config, dict):
        model_config = {}
    model_config["requires_api_key"] = False
    entry["model_config"] = model_config
    updated = True
    break

runtime_registry.write_text(json.dumps(entries, indent=2), encoding="utf-8")
print(f"Runtime registry written: {runtime_registry}")
if updated:
    print(f"Updated {learner_model} to local snapshot path for offline-safe loading.")
else:
    if str(os.getenv("HF_HUB_OFFLINE", "0")).strip() == "1":
        raise SystemExit(
            f"ERROR: HF_HUB_OFFLINE=1 and no local snapshot found for {learner_model}. "
            "Set HF_LOCAL_SNAPSHOT_PATH to your local Llama snapshot path."
        )
    print(
        f"Warning: could not resolve local snapshot for {learner_model}; "
        f"will keep original huggingface_id from {base_registry}."
    )
PY

echo "=== Start residual MoE training ==="
echo "EXPERIMENT_TAG=$EXPERIMENT_TAG"
echo "LEARNER_MODEL=$LEARNER_MODEL"
echo "TRAINER_SCRIPT=$TRAINER_SCRIPT"
echo "MOE_CONFIG_PATH=$MOE_CONFIG_PATH"
echo "PER_GAME_MAX_SAMPLES=$PER_GAME_MAX_SAMPLES"
echo "ADAPTER_BASE_ROOT=$ADAPTER_BASE_ROOT"
echo "ADAPTER_ROOT=$ADAPTER_ROOT"
echo "PLAYPEN_WANDB=$PLAYPEN_WANDB WANDB_PROJECT=$WANDB_PROJECT WANDB_MODE=$WANDB_MODE"
python -m playpen.cli run "$TRAINER_SCRIPT" -l "$LEARNER_MODEL"

LATEST_REGISTRY="$(ls -1t "$REGISTRY_OUT_DIR"/model_registry.*.sft.*.json | head -n1 || true)"
LATEST_REGISTRY="$(printf '%s' "$LATEST_REGISTRY" | tr -d '\r\n')"
if [ -z "$LATEST_REGISTRY" ]; then
  echo "ERROR: No registry snapshot produced in $REGISTRY_OUT_DIR" >&2
  exit 3
fi

echo "Using registry snapshot: $LATEST_REGISTRY"

mapfile -t TRAINED_MODELS < <(LATEST_REGISTRY="$LATEST_REGISTRY" LEARNER_MODEL="$LEARNER_MODEL" python - <<'PY'
import json
import os
from pathlib import Path

registry = Path(os.environ["LATEST_REGISTRY"])
base = os.environ["LEARNER_MODEL"]
items = json.loads(registry.read_text(encoding="utf-8"))
for item in items:
    if not isinstance(item, dict):
        continue
    name = item.get("model_name")
    cfg = item.get("model_config") or {}
    if not isinstance(name, str):
        continue
    if not name.startswith(base + "-sft-moe-residual"):
        continue
    if bool(cfg.get("moe_enabled")) and str(cfg.get("moe_mode", "")).lower() in {"residual", "residual_skill", "residual-skill"}:
        print(name)
PY
)

if [ "${#TRAINED_MODELS[@]}" -eq 0 ]; then
  echo "ERROR: No residual MoE models found in $LATEST_REGISTRY for base $LEARNER_MODEL" >&2
  exit 4
fi

echo "=== Start evaluation for ${#TRAINED_MODELS[@]} residual MoE model(s) ==="

for model_name in "${TRAINED_MODELS[@]}"; do
  model_safe="${model_name//\//_}"
  out_dir="$EVAL_ROOT/$model_safe"
  mkdir -p "$out_dir"

  cmd=(python -m playpen.cli eval "$model_name" --suite "$EVAL_SUITE" --model-registry "$LATEST_REGISTRY" -r "$out_dir")
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
    PLAYPEN_MOE_EVAL_LOG_DIR="$out_dir/moe-routing" \
    PLAYPEN_MOE_EVAL_FLUSH_EVERY="$MOE_EVAL_ROUTING_FLUSH" \
    PLAYPEN_EVAL_INSTANCES_PATH="$PLAYPEN_EVAL_INSTANCES_PATH" \
    PLAYPEN_EVAL_INSTANCES_STATIC_PATH="$PLAYPEN_EVAL_INSTANCES_STATIC_PATH" \
    "${cmd[@]}"
  else
    PLAYPEN_EVAL_INSTANCES_PATH="$PLAYPEN_EVAL_INSTANCES_PATH" \
    PLAYPEN_EVAL_INSTANCES_STATIC_PATH="$PLAYPEN_EVAL_INSTANCES_STATIC_PATH" \
    "${cmd[@]}"
  fi
done

echo "All done."
echo "Experiment tag: $EXPERIMENT_TAG"
echo "Adapter base root: $ADAPTER_BASE_ROOT"
echo "Adapter model root: $ADAPTER_ROOT"
echo "Registry: $LATEST_REGISTRY"
echo "Eval results: $EVAL_ROOT"
if [ "$PLAYPEN_WANDB" = "1" ] && [ "$WANDB_MODE" = "offline" ]; then
  echo "W&B offline logs: $WANDB_DIR"
  echo "Sync later from a node with internet:"
  echo "  wandb sync --sync-all $WANDB_DIR/wandb"
fi
