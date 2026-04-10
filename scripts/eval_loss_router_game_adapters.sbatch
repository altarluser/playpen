#!/bin/bash
#SBATCH -J playpen-eval-loss-router
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
RESULTS_DIR="${RESULTS_DIR:-$SCRATCH_ROOT/playpen-eval/loss-router-game-adapters/$SLURM_JOB_ID}"

# User knobs (override via sbatch --export=...)
LEARNER_MODEL="${LEARNER_MODEL:-llama3-8b}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$SCRATCH_ROOT/playpen-adapters/${LEARNER_MODEL}-adapters}"
BASE_REGISTRY_PATH="${BASE_REGISTRY_PATH:-$REPO_DIR/model_registry.json}"
ROUTED_MODEL_NAME="${ROUTED_MODEL_NAME:-${LEARNER_MODEL}-moe-loss-router}"
EVAL_SUITE="${EVAL_SUITE:-all}"                # static | clem | all
EVAL_GAME="${EVAL_GAME:-}"                     # optional single game
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-}"         # optional max tokens
EVAL_SKIP_GAMEPLAY="${EVAL_SKIP_GAMEPLAY:-0}"  # 1 to skip gameplay
LOSS_ROUTER_ID_REGIMES="${LOSS_ROUTER_ID_REGIMES:-id}"  # comma-separated
MOE_EVAL_ROUTING_LOGS="${MOE_EVAL_ROUTING_LOGS:-1}"
MOE_EVAL_ROUTING_FLUSH="${MOE_EVAL_ROUTING_FLUSH:-8}"

RUNTIME_REGISTRY="$REGISTRY_OUT_DIR/model_registry.loss_router.runtime.${SLURM_JOB_ID}.json"
MOE_CONFIG_PATH="$RESULTS_DIR/moe.loss_router.${SLURM_JOB_ID}.json"

mkdir -p "$REPO_DIR/logs" "$HF_CACHE_ROOT" "$REGISTRY_OUT_DIR" "$RESULTS_DIR"
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

if [ ! -f "$BASE_REGISTRY_PATH" ]; then
  echo "ERROR: BASE_REGISTRY_PATH not found: $BASE_REGISTRY_PATH" >&2
  exit 2
fi
if [ ! -d "$ADAPTER_ROOT" ]; then
  echo "ERROR: ADAPTER_ROOT not found: $ADAPTER_ROOT" >&2
  exit 3
fi

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(python -V 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "HF cache: $HF_CACHE_ROOT"
echo "Registry base: $BASE_REGISTRY_PATH"
echo "Adapter root: $ADAPTER_ROOT"
echo "Results: $RESULTS_DIR"
nvidia-smi || true

BASE_REGISTRY_PATH="$BASE_REGISTRY_PATH" \
ADAPTER_ROOT="$ADAPTER_ROOT" \
LEARNER_MODEL="$LEARNER_MODEL" \
RUNTIME_REGISTRY="$RUNTIME_REGISTRY" \
MOE_CONFIG_PATH="$MOE_CONFIG_PATH" \
ROUTED_MODEL_NAME="$ROUTED_MODEL_NAME" \
LOSS_ROUTER_ID_REGIMES="$LOSS_ROUTER_ID_REGIMES" \
python - <<'PY'
import json
import re
import os
from pathlib import Path

base_registry_path = Path(os.environ["BASE_REGISTRY_PATH"]).expanduser()
adapter_root = Path(os.environ["ADAPTER_ROOT"]).expanduser()
learner_model = str(os.environ["LEARNER_MODEL"])
runtime_registry = Path(os.environ["RUNTIME_REGISTRY"]).expanduser()
moe_config_path = Path(os.environ["MOE_CONFIG_PATH"]).expanduser()
routed_model_name = str(os.environ["ROUTED_MODEL_NAME"])
id_regimes = [x.strip() for x in str(os.environ.get("LOSS_ROUTER_ID_REGIMES", "id")).split(",") if x.strip()]

payload = json.loads(base_registry_path.read_text(encoding="utf-8"))
entries = payload if isinstance(payload, list) else [payload]
entries = [x for x in entries if isinstance(x, dict)]

base = None
for item in entries:
    if str(item.get("model_name")) == learner_model:
        base = item
        break
if base is None:
    raise SystemExit(f"ERROR: base model '{learner_model}' not found in {base_registry_path}")

def latest_checkpoint(path: Path):
    cks = [p for p in path.glob("checkpoint-*") if p.is_dir()]
    if not cks:
        return None
    def idx(p: Path):
        m = re.search(r"checkpoint-(\d+)$", p.name)
        return int(m.group(1)) if m else -1
    cks.sort(key=idx)
    return cks[-1]

experts = []
new_entries_by_name = {str(x.get("model_name")): dict(x) for x in entries if x.get("model_name")}

for game_dir in sorted(adapter_root.iterdir()):
    if not game_dir.is_dir():
        continue
    if not game_dir.name.startswith("game_"):
        continue
    ckpt = latest_checkpoint(game_dir)
    if ckpt is None:
        continue
    game_name = game_dir.name[len("game_"):]
    model_name = f"{learner_model}-sft-game_{game_name}"

    spec = dict(base)
    spec["model_name"] = model_name
    cfg = dict(spec.get("model_config") or {})
    cfg["peft_model"] = str(ckpt)
    spec["model_config"] = cfg
    new_entries_by_name[model_name] = spec
    experts.append(model_name)

if not experts:
    raise SystemExit(f"ERROR: no game_* adapters with checkpoint-* found under {adapter_root}")

runtime_entries = list(new_entries_by_name.values())
runtime_registry.parent.mkdir(parents=True, exist_ok=True)
runtime_registry.write_text(json.dumps(runtime_entries, indent=2), encoding="utf-8")

moe_cfg = {
    "name": routed_model_name,
    "default_model": learner_model,
    "loss_router": {
        "experts": experts,
        "id_regimes": id_regimes or ["id"],
    },
}
moe_config_path.parent.mkdir(parents=True, exist_ok=True)
moe_config_path.write_text(json.dumps(moe_cfg, indent=2), encoding="utf-8")

print(f"Runtime registry written: {runtime_registry}")
print(f"MoE loss-router config written: {moe_config_path}")
print(f"Discovered experts: {len(experts)}")
for x in experts:
    print(" -", x)
PY

cmd=(python -m playpen.cli eval "$ROUTED_MODEL_NAME" --moe "$MOE_CONFIG_PATH" --model-registry "$RUNTIME_REGISTRY" --suite "$EVAL_SUITE" -r "$RESULTS_DIR")
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
echo "Runtime registry: $RUNTIME_REGISTRY"
echo "MoE config: $MOE_CONFIG_PATH"
echo "Results: $RESULTS_DIR"
