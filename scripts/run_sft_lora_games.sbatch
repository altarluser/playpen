#!/bin/bash
#SBATCH -J playpen-sft-games
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
ADAPTER_ROOT="$SCRATCH_ROOT/playpen-adapters"
REGISTRY_OUT_DIR="$SCRATCH_ROOT/playpen-model-registries"
EVAL_ROOT="$SCRATCH_ROOT/playpen-eval/$SLURM_JOB_ID"
RUNTIME_REGISTRY="$REGISTRY_OUT_DIR/model_registry.runtime.${SLURM_JOB_ID}.json"

# Training/eval knobs (override via sbatch --export=...)
LEARNER_MODEL="${LEARNER_MODEL:-llama3-8b}"
TRAIN_TARGETS="${TRAIN_TARGETS:-all-games}"          # all-games | all | combined | comma-separated
PER_GAME_MAX_SAMPLES="${PER_GAME_MAX_SAMPLES:-620}"  # per-game cap in sft_trainer_lora.py
EVAL_SUITE="${EVAL_SUITE:-static}"                   # static | clem | all
EVAL_GAME="${EVAL_GAME:-}"                           # optional: e.g. ifeval
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-}"
EVAL_SKIP_GAMEPLAY="${EVAL_SKIP_GAMEPLAY:-0}"
PLAYPEN_WANDB="${PLAYPEN_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-llama3-sft-adapters}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-offline}"               # offline | online | disabled
WANDB_DIR="${WANDB_DIR:-$SCRATCH_ROOT/wandb}"

mkdir -p "$REPO_DIR/logs" "$HF_CACHE_ROOT" "$ADAPTER_ROOT" "$REGISTRY_OUT_DIR" "$EVAL_ROOT" "$WANDB_DIR"
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

# Pin this process to the first GPU assigned by Slurm so torch/transformers use a single logical cuda:0.
if [ -n "${SLURM_STEP_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS%%,*}"
elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS%%,*}"
fi

# Use scratch cache roots consistently for both datasets and models.
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_CACHE="$HF_CACHE_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_ROOT"
export TRANSFORMERS_CACHE="$HF_CACHE_ROOT"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT"
export NLTK_DATA="${NLTK_DATA:-$HF_CACHE_ROOT/nltk_data}"

# Keep online mode unless you explicitly export HF_*_OFFLINE=1.
: "${HF_HUB_OFFLINE:=0}"
: "${HF_DATASETS_OFFLINE:=0}"
export HF_HUB_OFFLINE HF_DATASETS_OFFLINE

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(python -V 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "HF cache: $HF_CACHE_ROOT"
echo "NLTK_DATA=$NLTK_DATA"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
nvidia-smi || true

# Make sure clemcore can discover games and write logs from cwd.
touch "$REPO_DIR/clembench.log"
cat > "$REPO_DIR/game_registry.json" <<'JSON'
[{"benchmark_path": "/home/bbmaltar/im/playpen/clembench"}]
JSON

# Route adapter outputs to scratch via symlink expected by trainer output_dir.
mkdir -p "$REPO_DIR/models"
if [ -e "$REPO_DIR/models/sft+lora" ] && [ ! -L "$REPO_DIR/models/sft+lora" ]; then
  echo "ERROR: $REPO_DIR/models/sft+lora exists and is not a symlink. Move it manually, then rerun." >&2
  exit 2
fi
ln -sfn "$ADAPTER_ROOT" "$REPO_DIR/models/sft+lora"

# Trainer env controls.
export PLAYPEN_TRAIN_TARGETS="$TRAIN_TARGETS"
export PLAYPEN_PER_GAME_MAX_SAMPLES="$PER_GAME_MAX_SAMPLES"
export PLAYPEN_MODEL_REGISTRY_DIR="$REGISTRY_OUT_DIR"
export PLAYPEN_WANDB
export WANDB_PROJECT
export WANDB_MODE
export WANDB_DIR
if [ -n "$WANDB_ENTITY" ]; then
  export WANDB_ENTITY
fi

# Build a runtime model registry that prefers local snapshot paths from HF cache.
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

payload = json.loads(base_registry.read_text(encoding="utf-8"))
entries = payload if isinstance(payload, list) else [payload]

def resolve_snapshot(repo_id: str) -> str:
    if explicit_snapshot:
        p = Path(explicit_snapshot).expanduser()
        if p.exists() and p.is_dir():
            return str(p)
    pattern = f"models--{repo_id.replace('/', '--')}/snapshots/*"
    snaps = sorted([p for p in hf_cache_root.glob(pattern) if p.is_dir()])
    return str(snaps[-1]) if snaps else ""

updated = False
for entry in entries:
    if not isinstance(entry, dict):
        continue
    if str(entry.get("model_name")) != learner_model:
        continue
    hf_id = entry.get("huggingface_id")
    if not isinstance(hf_id, str) or not hf_id:
        continue
    # Already local path.
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
    print(
        f"Warning: could not resolve local snapshot for {learner_model}; "
        f"will keep original huggingface_id from {base_registry}."
    )
PY
export PLAYPEN_SOURCE_MODEL_REGISTRY="$RUNTIME_REGISTRY"

echo "=== Start training ==="
echo "LEARNER_MODEL=$LEARNER_MODEL"
echo "TRAIN_TARGETS=$TRAIN_TARGETS"
echo "PER_GAME_MAX_SAMPLES=$PER_GAME_MAX_SAMPLES"
echo "PLAYPEN_WANDB=$PLAYPEN_WANDB WANDB_PROJECT=$WANDB_PROJECT WANDB_MODE=$WANDB_MODE"
python -m playpen.cli run examples/trl/sft_trainer_lora.py -l "$LEARNER_MODEL"

# Find newest registry snapshot emitted by trainer.
LATEST_REGISTRY="$(ls -1t "$REGISTRY_OUT_DIR"/model_registry.*.sft.*.json | head -n1 || true)"
if [ -z "$LATEST_REGISTRY" ]; then
  echo "ERROR: No registry snapshot produced in $REGISTRY_OUT_DIR" >&2
  exit 3
fi

echo "Using registry snapshot: $LATEST_REGISTRY"

# Build model list generated by this run (base-sft and base-sft-game_*).
mapfile -t TRAINED_MODELS < <(python - <<PY
import json
from pathlib import Path

registry = Path(r"$LATEST_REGISTRY")
base = r"$LEARNER_MODEL"
prefix = base + "-sft"
items = json.loads(registry.read_text(encoding="utf-8"))
for item in items:
    if not isinstance(item, dict):
        continue
    name = item.get("model_name")
    if isinstance(name, str) and (name == prefix or name.startswith(prefix + "-")):
        print(name)
PY
)

if [ "${#TRAINED_MODELS[@]}" -eq 0 ]; then
  echo "ERROR: No trained models found in $LATEST_REGISTRY for base $LEARNER_MODEL" >&2
  exit 4
fi

echo "=== Start evaluation for ${#TRAINED_MODELS[@]} models ==="

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
  "${cmd[@]}"
done

echo "All done."
echo "Adapters: $ADAPTER_ROOT"
echo "Registry: $LATEST_REGISTRY"
echo "Eval results: $EVAL_ROOT"
if [ "$PLAYPEN_WANDB" = "1" ] && [ "$WANDB_MODE" = "offline" ]; then
  echo "W&B offline logs: $WANDB_DIR"
  echo "Sync later from a node with internet:"
  echo "  wandb sync --sync-all $WANDB_DIR/wandb"
fi
