#!/bin/bash
#SBATCH -J playpen-eval-adapter
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
BASE_REGISTRY="$REPO_DIR/model_registry.json"
SCRATCH_ROOT="/scratch/usr/bbmaltar"
HF_CACHE_ROOT="$SCRATCH_ROOT/hf-cache"
REGISTRY_OUT_DIR="$SCRATCH_ROOT/playpen-model-registries"
RESULTS_DIR="${RESULTS_DIR:-$SCRATCH_ROOT/playpen-eval/adapter-only/$SLURM_JOB_ID}"
RUNTIME_REGISTRY="$REGISTRY_OUT_DIR/model_registry.eval_adapter.runtime.${SLURM_JOB_ID}.json"

# Eval knobs (override via sbatch --export=...)
REGISTRY_PATH="${REGISTRY_PATH:-}"
LEARNER_MODEL="${LEARNER_MODEL:-llama3-8b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_NAMES="${MODEL_NAMES:-}"
MODEL_GROUP="${MODEL_GROUP:-game-adapters}"   # game-adapters | all-sft | moe-only | adapters
EVAL_SUITE="${EVAL_SUITE:-all}"
EVAL_GAME="${EVAL_GAME:-}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-}"
EVAL_SKIP_GAMEPLAY="${EVAL_SKIP_GAMEPLAY:-0}"
MOE_EVAL_ROUTING_LOGS="${MOE_EVAL_ROUTING_LOGS:-1}"
MOE_EVAL_ROUTING_FLUSH="${MOE_EVAL_ROUTING_FLUSH:-8}"
MODEL_LIST_FILE="$RESULTS_DIR/model_list.txt"

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

if [ -z "$REGISTRY_PATH" ]; then
  REGISTRY_PATH="$(ls -1t "$REGISTRY_OUT_DIR"/model_registry.*.sft.*.json 2>/dev/null | head -n1 || true)"
fi
if [ -z "$REGISTRY_PATH" ]; then
  REGISTRY_PATH="$BASE_REGISTRY"
fi
if [ ! -f "$REGISTRY_PATH" ]; then
  echo "ERROR: REGISTRY_PATH not found: $REGISTRY_PATH" >&2
  exit 2
fi

# Resolve a model name if not passed, then rewrite huggingface_id to local snapshot where possible.
REGISTRY_PATH="$REGISTRY_PATH" \
RUNTIME_REGISTRY="$RUNTIME_REGISTRY" \
LEARNER_MODEL="$LEARNER_MODEL" \
MODEL_NAME="$MODEL_NAME" \
MODEL_NAMES="$MODEL_NAMES" \
MODEL_GROUP="$MODEL_GROUP" \
MODEL_LIST_FILE="$MODEL_LIST_FILE" \
HF_CACHE_ROOT="$HF_CACHE_ROOT" \
python - <<'PY'
import json
import os
from pathlib import Path

registry_path = Path(os.environ["REGISTRY_PATH"])
runtime_registry = Path(os.environ["RUNTIME_REGISTRY"])
learner_model = (os.environ.get("LEARNER_MODEL") or "").strip()
model_name = (os.environ.get("MODEL_NAME") or "").strip()
model_names_raw = (os.environ.get("MODEL_NAMES") or "").strip()
model_group = (os.environ.get("MODEL_GROUP") or "game-adapters").strip().lower()
model_list_file = Path(os.environ["MODEL_LIST_FILE"])
hf_cache_root = Path(os.environ["HF_CACHE_ROOT"])
explicit_snapshot = (os.getenv("HF_LOCAL_SNAPSHOT_PATH") or "").strip()

payload = json.loads(registry_path.read_text(encoding="utf-8"))
entries = payload if isinstance(payload, list) else [payload]

selected = []
if model_name:
    selected = [model_name]
elif model_names_raw:
    selected = [x.strip() for x in model_names_raw.split(",") if x.strip()]
else:
    names = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        cfg = item.get("model_config") or {}
        name = item.get("model_name")
        if not isinstance(name, str):
            continue
        names.append((name, cfg))

    if model_group == "game-adapters":
        prefix = f"{learner_model}-sft-game_"
        selected = [n for n, _ in names if n.startswith(prefix)]
    elif model_group == "all-sft":
        prefix = f"{learner_model}-sft"
        selected = [n for n, _ in names if n == prefix or n.startswith(prefix + "-")]
    elif model_group == "moe-only":
        selected = [n for n, cfg in names if cfg.get("moe_enabled") or n.endswith("-sft-moe")]
    else:
        selected = [n for n, cfg in names if cfg.get("peft_model") or cfg.get("peft_models") or cfg.get("moe_enabled")]

if not selected:
    raise SystemExit(
        "ERROR: no models selected. Set MODEL_NAME or MODEL_NAMES, or adjust MODEL_GROUP."
    )

entry_by_name = {}
for item in entries:
    if isinstance(item, dict):
        n = item.get("model_name")
        if isinstance(n, str):
            entry_by_name[n] = item

missing = [n for n in selected if n not in entry_by_name]
if missing:
    raise SystemExit("ERROR: selected model(s) not found in registry: " + ", ".join(missing))

def resolve_snapshot(repo_id: str) -> str:
    if explicit_snapshot:
        p = Path(explicit_snapshot).expanduser()
        if p.exists() and p.is_dir():
            return str(p)
    pattern = f"models--{repo_id.replace('/', '--')}/snapshots/*"
    snaps = sorted([p for p in hf_cache_root.glob(pattern) if p.is_dir()])
    return str(snaps[-1]) if snaps else ""

updated = 0
for name in selected:
    entry = entry_by_name[name]
    hf_id = entry.get("huggingface_id")
    if not isinstance(hf_id, str) or not hf_id:
        continue
    if Path(hf_id).expanduser().exists():
        updated += 1
        continue
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
    updated += 1

runtime_registry.write_text(json.dumps(entries, indent=2), encoding="utf-8")
model_list_file.write_text("\n".join(selected) + "\n", encoding="utf-8")
print(f"Runtime registry written: {runtime_registry}")
print(f"Selected models: {len(selected)}")
print(f"Local base path resolved for {updated}/{len(selected)} selected model(s).")
if updated < len(selected):
    print("Warning: some selected model(s) still reference non-local huggingface_id.")
PY

mapfile -t MODEL_LIST < "$MODEL_LIST_FILE"
if [ "${#MODEL_LIST[@]}" -eq 0 ]; then
  echo "ERROR: empty model list in $MODEL_LIST_FILE" >&2
  exit 3
fi

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(python -V 2>&1)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "HF cache: $HF_CACHE_ROOT"
echo "Registry: $RUNTIME_REGISTRY"
echo "Models selected: ${#MODEL_LIST[@]}"
echo "Results: $RESULTS_DIR"
nvidia-smi || true

for MODEL_NAME in "${MODEL_LIST[@]}"; do
  model_safe="${MODEL_NAME//\//_}"
  out_dir="$RESULTS_DIR/$model_safe"
  mkdir -p "$out_dir"

  cmd=(python -m playpen.cli eval "$MODEL_NAME" --suite "$EVAL_SUITE" --model-registry "$RUNTIME_REGISTRY" -r "$out_dir")
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
    "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
done

echo "Done."
