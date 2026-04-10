#!/bin/bash
#SBATCH -J playpen-train-eval-cluster-lr
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
RESULTS_ROOT="${RESULTS_ROOT:-$SCRATCH_ROOT/playpen-eval/cluster-loss-router/$SLURM_JOB_ID}"
RUNTIME_REGISTRY="$REGISTRY_OUT_DIR/model_registry.cluster.runtime.${SLURM_JOB_ID}.json"

# Training knobs
LEARNER_MODEL="${LEARNER_MODEL:-llama3-8b}"
TRAINER_SCRIPT="${TRAINER_SCRIPT:-examples/trl/sft_trainer_lora.py}"
TRAIN_TARGETS="${TRAIN_TARGETS:-cluster_all,cluster_wordguessing,cluster_explorationnavigation,cluster_cooperation}"  # forwarded to PLAYPEN_TRAIN_TARGETS
PER_GAME_MAX_SAMPLES="${PER_GAME_MAX_SAMPLES:-620}"
PLAYPEN_WANDB="${PLAYPEN_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-llama3-sft-cluster-adapters}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DIR="${WANDB_DIR:-$SCRATCH_ROOT/wandb}"

# Eval knobs (applies to per-adapter eval and routed eval)
EVAL_SUITE="${EVAL_SUITE:-all}"                         # static | clem | all
EVAL_GAME="${EVAL_GAME:-}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-}"
EVAL_SKIP_GAMEPLAY="${EVAL_SKIP_GAMEPLAY:-0}"
BASELINE_MODEL="${BASELINE_MODEL:-$LEARNER_MODEL}"
EVAL_CLUSTER_SUFFIXES="${EVAL_CLUSTER_SUFFIXES:-cluster_all,cluster_wordguessing,cluster_explorationnavigation,cluster_cooperation}"
ROUTER_EXPERT_SUFFIXES="${ROUTER_EXPERT_SUFFIXES:-cluster_wordguessing,cluster_explorationnavigation,cluster_cooperation}"
ROUTED_MODEL_NAME="${ROUTED_MODEL_NAME:-${LEARNER_MODEL}-moe-loss-router-cluster}"
LOSS_ROUTER_ID_REGIMES="${LOSS_ROUTER_ID_REGIMES:-id}"
MOE_EVAL_ROUTING_LOGS="${MOE_EVAL_ROUTING_LOGS:-1}"
MOE_EVAL_ROUTING_FLUSH="${MOE_EVAL_ROUTING_FLUSH:-8}"

RESULTS_ADAPTERS="$RESULTS_ROOT/per-adapter"
RESULTS_ROUTED="$RESULTS_ROOT/loss-router"
MODEL_LIST_FILE="$RESULTS_ROOT/cluster_models.txt"
ROUTER_EXPERTS_FILE="$RESULTS_ROOT/router_experts.txt"
MOE_CONFIG_PATH="$RESULTS_ROOT/moe.loss_router.cluster.${SLURM_JOB_ID}.json"

mkdir -p "$REPO_DIR/logs" "$HF_CACHE_ROOT" "$ADAPTER_ROOT" "$REGISTRY_OUT_DIR" "$RESULTS_ROOT" "$RESULTS_ADAPTERS" "$RESULTS_ROUTED" "$WANDB_DIR"
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

# Route adapter outputs to scratch via symlink expected by trainer output_dir.
mkdir -p "$REPO_DIR/models"
if [ -e "$REPO_DIR/models/sft+lora" ] && [ ! -L "$REPO_DIR/models/sft+lora" ]; then
  echo "ERROR: $REPO_DIR/models/sft+lora exists and is not a symlink. Move it manually, then rerun." >&2
  exit 2
fi
ln -sfn "$ADAPTER_ROOT" "$REPO_DIR/models/sft+lora"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(python -V 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "HF cache: $HF_CACHE_ROOT"
echo "Trainer script: $TRAINER_SCRIPT"
echo "LEARNER_MODEL=$LEARNER_MODEL"
echo "TRAIN_TARGETS=$TRAIN_TARGETS"
echo "EVAL_CLUSTER_SUFFIXES=$EVAL_CLUSTER_SUFFIXES"
echo "ROUTER_EXPERT_SUFFIXES=$ROUTER_EXPERT_SUFFIXES"
echo "Results root: $RESULTS_ROOT"
nvidia-smi || true

# Build runtime model registry with local base snapshot when possible.
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
    if Path(hf_id).expanduser().exists():
        updated = True
        break
    if "/" not in hf_id:
        continue
    snap = resolve_snapshot(hf_id)
    if not snap:
        continue
    entry["huggingface_id"] = snap
    cfg = entry.get("model_config")
    if not isinstance(cfg, dict):
        cfg = {}
    cfg["requires_api_key"] = False
    entry["model_config"] = cfg
    updated = True
    break

runtime_registry.write_text(json.dumps(entries, indent=2), encoding="utf-8")
print(f"Runtime registry written: {runtime_registry}")
if updated:
    print(f"Updated {learner_model} to local snapshot path.")
else:
    print("Warning: base model path not converted to local snapshot.")
PY

export PLAYPEN_SOURCE_MODEL_REGISTRY="$RUNTIME_REGISTRY"
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

echo "=== Start cluster adapter training ==="
python -m playpen.cli run "$TRAINER_SCRIPT" -l "$LEARNER_MODEL"

LATEST_REGISTRY="$(ls -1t "$REGISTRY_OUT_DIR"/model_registry.*.sft.*.json | head -n1 || true)"
if [ -z "$LATEST_REGISTRY" ]; then
  echo "ERROR: No registry snapshot produced in $REGISTRY_OUT_DIR" >&2
  exit 3
fi
echo "Using trained registry: $LATEST_REGISTRY"

# Resolve exact cluster models for eval and router experts.
LATEST_REGISTRY="$LATEST_REGISTRY" \
LEARNER_MODEL="$LEARNER_MODEL" \
EVAL_CLUSTER_SUFFIXES="$EVAL_CLUSTER_SUFFIXES" \
ROUTER_EXPERT_SUFFIXES="$ROUTER_EXPERT_SUFFIXES" \
MODEL_LIST_FILE="$MODEL_LIST_FILE" \
ROUTER_EXPERTS_FILE="$ROUTER_EXPERTS_FILE" \
python - <<'PY'
import json
import os
from pathlib import Path

registry = Path(os.environ["LATEST_REGISTRY"])
base = os.environ["LEARNER_MODEL"]
out = Path(os.environ["MODEL_LIST_FILE"])
router_out = Path(os.environ["ROUTER_EXPERTS_FILE"])
eval_suffixes = [x.strip().lower() for x in os.environ.get("EVAL_CLUSTER_SUFFIXES", "").split(",") if x.strip()]
router_suffixes = [x.strip().lower() for x in os.environ.get("ROUTER_EXPERT_SUFFIXES", "").split(",") if x.strip()]

items = json.loads(registry.read_text(encoding="utf-8"))
if isinstance(items, dict):
    items = [items]
name_set = {str(item.get("model_name")) for item in items if isinstance(item, dict) and isinstance(item.get("model_name"), str)}

def resolve_model_for_suffix(suffix: str) -> str:
    candidates = [
        f"{base}-sft-{suffix}",
        f"{base}-sft-game_{suffix}",
    ]
    for c in candidates:
        if c in name_set:
            return c
    return ""

picked = []
missing_eval = []
for suffix in eval_suffixes:
    m = resolve_model_for_suffix(suffix)
    if m:
        picked.append(m)
    else:
        missing_eval.append(suffix)

router_experts = []
missing_router = []
for suffix in router_suffixes:
    m = resolve_model_for_suffix(suffix)
    if m:
        router_experts.append(m)
    else:
        missing_router.append(suffix)

picked = sorted(set(picked))
router_experts = sorted(set(router_experts))

if missing_eval:
    raise SystemExit("ERROR: missing eval cluster model(s) for suffixes: " + ", ".join(missing_eval))
if missing_router:
    raise SystemExit("ERROR: missing router expert model(s) for suffixes: " + ", ".join(missing_router))
if not picked:
    raise SystemExit(f"ERROR: no cluster adapters selected from {registry}")
if not router_experts:
    raise SystemExit(f"ERROR: no router experts selected from {registry}")

out.write_text("\n".join(picked) + ("\n" if picked else ""), encoding="utf-8")
router_out.write_text("\n".join(router_experts) + ("\n" if router_experts else ""), encoding="utf-8")

print(f"Cluster adapters selected for per-adapter eval: {len(picked)}")
for x in picked:
    print(" -", x)
print(f"Cluster experts selected for loss-router: {len(router_experts)}")
for x in router_experts:
    print(" *", x)
PY

mapfile -t CLUSTER_MODELS < "$MODEL_LIST_FILE"
if [ "${#CLUSTER_MODELS[@]}" -eq 0 ]; then
  echo "ERROR: empty cluster model list: $MODEL_LIST_FILE" >&2
  exit 4
fi
mapfile -t ROUTER_EXPERTS < "$ROUTER_EXPERTS_FILE"
if [ "${#ROUTER_EXPERTS[@]}" -eq 0 ]; then
  echo "ERROR: empty router experts list: $ROUTER_EXPERTS_FILE" >&2
  exit 5
fi

echo "=== Evaluate baseline model ($BASELINE_MODEL) ==="
model_name="$BASELINE_MODEL"
model_safe="${model_name//\//_}"
out_dir="$RESULTS_ADAPTERS/$model_safe"
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
echo "Running baseline eval: ${cmd[*]}"
if [ "$MOE_EVAL_ROUTING_LOGS" = "1" ]; then
  PLAYPEN_MOE_EVAL_LOG_DIR="$out_dir/moe-routing" \
  PLAYPEN_MOE_EVAL_FLUSH_EVERY="$MOE_EVAL_ROUTING_FLUSH" \
  "${cmd[@]}"
else
  "${cmd[@]}"
fi

echo "=== Evaluate each cluster adapter (${#CLUSTER_MODELS[@]}) ==="
for model_name in "${CLUSTER_MODELS[@]}"; do
  model_safe="${model_name//\//_}"
  out_dir="$RESULTS_ADAPTERS/$model_safe"
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

  echo "Running adapter eval: ${cmd[*]}"
  if [ "$MOE_EVAL_ROUTING_LOGS" = "1" ]; then
    PLAYPEN_MOE_EVAL_LOG_DIR="$out_dir/moe-routing" \
    PLAYPEN_MOE_EVAL_FLUSH_EVERY="$MOE_EVAL_ROUTING_FLUSH" \
    "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
done

# Build loss-router MoE config over selected cluster experts only.
ROUTER_EXPERTS_FILE="$ROUTER_EXPERTS_FILE" \
LEARNER_MODEL="$LEARNER_MODEL" \
ROUTED_MODEL_NAME="$ROUTED_MODEL_NAME" \
LOSS_ROUTER_ID_REGIMES="$LOSS_ROUTER_ID_REGIMES" \
MOE_CONFIG_PATH="$MOE_CONFIG_PATH" \
python - <<'PY'
import json
import os
from pathlib import Path

model_list = Path(os.environ["ROUTER_EXPERTS_FILE"])
learner_model = os.environ["LEARNER_MODEL"]
routed_model_name = os.environ["ROUTED_MODEL_NAME"]
id_regimes = [x.strip() for x in os.environ.get("LOSS_ROUTER_ID_REGIMES", "id").split(",") if x.strip()]
moe_cfg_path = Path(os.environ["MOE_CONFIG_PATH"])

experts = [x.strip() for x in model_list.read_text(encoding="utf-8").splitlines() if x.strip()]
if not experts:
    raise SystemExit("ERROR: no experts for loss-router config")

cfg = {
    "name": routed_model_name,
    "default_model": learner_model,
    "loss_router": {
        "experts": experts,
        "id_regimes": id_regimes or ["id"],
    },
}
moe_cfg_path.parent.mkdir(parents=True, exist_ok=True)
moe_cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(f"Wrote loss-router config: {moe_cfg_path}")
print(f"Experts: {len(experts)}")
for x in experts:
    print(" -", x)
PY

echo "=== Evaluate loss-router over selected cluster experts ==="
cmd=(python -m playpen.cli eval "$ROUTED_MODEL_NAME" --moe "$MOE_CONFIG_PATH" --model-registry "$LATEST_REGISTRY" --suite "$EVAL_SUITE" -r "$RESULTS_ROUTED")
if [ -n "$EVAL_MAX_TOKENS" ]; then
  cmd+=( -L "$EVAL_MAX_TOKENS" )
fi
if [ -n "$EVAL_GAME" ]; then
  cmd+=( -g "$EVAL_GAME" )
fi
if [ "$EVAL_SKIP_GAMEPLAY" = "1" ]; then
  cmd+=( --skip_gameplay )
fi

echo "Running routed eval: ${cmd[*]}"
if [ "$MOE_EVAL_ROUTING_LOGS" = "1" ]; then
  PLAYPEN_MOE_EVAL_LOG_DIR="$RESULTS_ROUTED/moe-routing" \
  PLAYPEN_MOE_EVAL_FLUSH_EVERY="$MOE_EVAL_ROUTING_FLUSH" \
  "${cmd[@]}"
else
  "${cmd[@]}"
fi

echo "Done."
echo "Trained registry: $LATEST_REGISTRY"
echo "Cluster model list: $MODEL_LIST_FILE"
echo "Router experts list: $ROUTER_EXPERTS_FILE"
echo "Per-adapter eval results: $RESULTS_ADAPTERS"
echo "Routed eval results: $RESULTS_ROUTED"
echo "Loss-router MoE config: $MOE_CONFIG_PATH"
