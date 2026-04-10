#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/hf_cache_offline.sh download
  scripts/hf_cache_offline.sh pack
  scripts/hf_cache_offline.sh install
  scripts/hf_cache_offline.sh verify
  scripts/hf_cache_offline.sh all-online

Environment variables:
  HF_HOME
    Cache root for Hugging Face files.
    Default: <repo>/.hf-cache/huggingface

  NLTK_DATA
    Cache root for NLTK resources.
    Default: $HF_HOME/nltk_data

  NLTK_RESOURCES
    Optional comma-separated NLTK resources to prefetch/verify.
    Default: punkt,punkt_tab,stopwords,wordnet,omw-1.4

  ARCHIVE_PATH
    Path to compressed cache archive (.tar.gz).
    Default: <repo>/hf-cache.tar.gz

  TARGET_HF_HOME
    Install target for offline nodes (used by install/verify).
    Default: $HOME/.cache/huggingface

  MODEL_IDS
    Optional comma-separated Hugging Face model ids to prefetch/verify.
    Example: MODEL_IDS='Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.2-1B-Instruct'

  HF_TOKEN
    Optional Hugging Face token. If unset, script will try to load token from key.json.

  ALLOW_MODEL_DOWNLOAD_FAILURE
    If set to 1, model download/verify errors are printed but do not fail the command.
    Default: 0

  SKIP_NLTK
    If set to 1, skip NLTK prefetch/verify.
    Default: 0

Notes:
  - Run 'download' and 'pack' on a machine with internet.
  - Move ARCHIVE_PATH to your cluster.
  - Run 'install' then 'verify' on the cluster.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${HF_HOME:=$REPO_DIR/.hf-cache/huggingface}"
: "${NLTK_DATA:=$HF_HOME/nltk_data}"
: "${NLTK_RESOURCES:=punkt,punkt_tab,stopwords,wordnet,omw-1.4}"
: "${ARCHIVE_PATH:=$REPO_DIR/hf-cache.tar.gz}"
: "${TARGET_HF_HOME:=$HOME/.cache/huggingface}"
: "${TARGET_NLTK_DATA:=$TARGET_HF_HOME/nltk_data}"
: "${MODEL_IDS:=}"
: "${ALLOW_MODEL_DOWNLOAD_FAILURE:=0}"
: "${SKIP_NLTK:=0}"

load_hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then
    return 0
  fi

  if [ -f "$REPO_DIR/key.json" ]; then
    HF_TOKEN="$(
      python - <<'PY'
import json
from pathlib import Path

path = Path("key.json")
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)

for section in ("huggingface", "hf"):
    token = ((data or {}).get(section) or {}).get("api_key")
    if token:
        print(str(token).strip())
        raise SystemExit(0)

print("")
PY
    )"
  fi

  if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HF_HUB_TOKEN="$HF_TOKEN"
    echo "Loaded Hugging Face token from env/key.json."
  fi
}

prefetch_python() {
  HF_HOME="$HF_HOME" \
  MODEL_IDS="$MODEL_IDS" \
  ALLOW_MODEL_DOWNLOAD_FAILURE="$ALLOW_MODEL_DOWNLOAD_FAILURE" \
  NLTK_DATA="$NLTK_DATA" \
  NLTK_RESOURCES="$NLTK_RESOURCES" \
  SKIP_NLTK="$SKIP_NLTK" \
  python - <<'PY'
import os
from datasets import load_dataset

model_ids = [m.strip() for m in os.environ.get("MODEL_IDS", "").split(",") if m.strip()]
cache_dir = os.environ["HF_HOME"]
allow_model_fail = os.environ.get("ALLOW_MODEL_DOWNLOAD_FAILURE", "0").strip() == "1"
nltk_data_dir = os.environ["NLTK_DATA"]
nltk_resources = [r.strip() for r in os.environ.get("NLTK_RESOURCES", "").split(",") if r.strip()]
skip_nltk = os.environ.get("SKIP_NLTK", "0").strip() == "1"

needed = [
    ("colab-potsdam/playpen-data", "interactions", "train"),
    ("colab-potsdam/playpen-data", "instances", "validation"),
    ("colab-potsdam/playpen-data", "instances-static", "validation"),
    ("clembench-playpen/SFT-Final-Dataset", None, "train"),
]

for repo_id, config_name, split in needed:
    kwargs = {"path": repo_id, "split": split, "cache_dir": cache_dir}
    if config_name is not None:
        kwargs["name"] = config_name
    ds = load_dataset(**kwargs)
    print(f"Prefetched dataset: {repo_id} [{config_name}] split={split} rows={len(ds)}")

if model_ids:
    from huggingface_hub import snapshot_download
    failures = []
    for model_id in model_ids:
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_files_only=False,
            )
            print(f"Prefetched model: {model_id}")
        except Exception as e:
            failures.append((model_id, str(e)))
            print(f"Model prefetch failed: {model_id} -> {e}")
    if failures and not allow_model_fail:
        raise SystemExit(
            "One or more model downloads failed. Set ALLOW_MODEL_DOWNLOAD_FAILURE=1 to continue anyway."
        )
else:
    print("No MODEL_IDS set; skipping model prefetch.")

if skip_nltk:
    print("SKIP_NLTK=1; skipping NLTK prefetch.")
else:
    import nltk
    os.makedirs(nltk_data_dir, exist_ok=True)
    downloaded = []
    failed = []
    for resource in nltk_resources:
        ok = False
        try:
            ok = bool(nltk.download(resource, download_dir=nltk_data_dir, quiet=True))
        except Exception as e:
            failed.append((resource, str(e)))
            continue
        if ok:
            downloaded.append(resource)
        else:
            failed.append((resource, "download() returned False"))
    if downloaded:
        print("Prefetched NLTK resources:", ", ".join(downloaded))
    if failed:
        for name, err in failed:
            print(f"NLTK prefetch failed: {name} -> {err}")
        raise SystemExit("One or more NLTK downloads failed.")
PY
}

verify_python() {
  HF_HOME="$TARGET_HF_HOME" \
  MODEL_IDS="$MODEL_IDS" \
  ALLOW_MODEL_DOWNLOAD_FAILURE="$ALLOW_MODEL_DOWNLOAD_FAILURE" \
  NLTK_DATA="$TARGET_NLTK_DATA" \
  NLTK_RESOURCES="$NLTK_RESOURCES" \
  SKIP_NLTK="$SKIP_NLTK" \
  HF_DATASETS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  python - <<'PY'
import os
from datasets import load_dataset

cache_dir = os.environ["HF_HOME"]
model_ids = [m.strip() for m in os.environ.get("MODEL_IDS", "").split(",") if m.strip()]
allow_model_fail = os.environ.get("ALLOW_MODEL_DOWNLOAD_FAILURE", "0").strip() == "1"
nltk_data_dir = os.environ["NLTK_DATA"]
nltk_resources = [r.strip() for r in os.environ.get("NLTK_RESOURCES", "").split(",") if r.strip()]
skip_nltk = os.environ.get("SKIP_NLTK", "0").strip() == "1"

needed = [
    ("colab-potsdam/playpen-data", "interactions", "train"),
    ("colab-potsdam/playpen-data", "instances", "validation"),
    ("colab-potsdam/playpen-data", "instances-static", "validation"),
    ("clembench-playpen/SFT-Final-Dataset", None, "train"),
]

for repo_id, config_name, split in needed:
    kwargs = {"path": repo_id, "split": split, "cache_dir": cache_dir}
    if config_name is not None:
        kwargs["name"] = config_name
    ds = load_dataset(**kwargs)
    print(f"Offline dataset OK: {repo_id} [{config_name}] split={split} rows={len(ds)}")

if model_ids:
    from huggingface_hub import snapshot_download
    failures = []
    for model_id in model_ids:
        try:
            local_dir = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            print(f"Offline model OK: {model_id} -> {local_dir}")
        except Exception as e:
            failures.append((model_id, str(e)))
            print(f"Offline model check failed: {model_id} -> {e}")
    if failures and not allow_model_fail:
        raise SystemExit(
            "One or more model checks failed. Set ALLOW_MODEL_DOWNLOAD_FAILURE=1 to continue anyway."
        )
else:
    print("No MODEL_IDS set; skipping model verification.")

if skip_nltk:
    print("SKIP_NLTK=1; skipping NLTK verification.")
else:
    import nltk
    nltk.data.path.insert(0, nltk_data_dir)
    resource_paths = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    missing = []
    for resource in nltk_resources:
        lookup = resource_paths.get(resource, resource)
        try:
            nltk.data.find(lookup)
            print(f"Offline NLTK OK: {resource} ({lookup})")
        except LookupError:
            missing.append((resource, lookup))
    if missing:
        for name, lookup in missing:
            print(f"Offline NLTK missing: {name} ({lookup})")
        raise SystemExit("One or more NLTK resources are missing.")
PY
}

cmd="${1:-}"
case "$cmd" in
  download)
    load_hf_token
    mkdir -p "$HF_HOME" "$NLTK_DATA"
    echo "Downloading datasets/models into HF_HOME=$HF_HOME"
    echo "Downloading NLTK resources into NLTK_DATA=$NLTK_DATA"
    prefetch_python
    ;;

  pack)
    mkdir -p "$(dirname "$ARCHIVE_PATH")"
    parent_dir="$(dirname "$HF_HOME")"
    base_dir="$(basename "$HF_HOME")"
    if [ ! -d "$HF_HOME" ]; then
      echo "HF cache directory not found: $HF_HOME" >&2
      exit 1
    fi
    echo "Packing $HF_HOME -> $ARCHIVE_PATH"
    tar -C "$parent_dir" -czf "$ARCHIVE_PATH" "$base_dir"
    ls -lh "$ARCHIVE_PATH"
    ;;

  install)
    if [ ! -f "$ARCHIVE_PATH" ]; then
      echo "Archive not found: $ARCHIVE_PATH" >&2
      exit 1
    fi
    tmp_dir="$(mktemp -d /tmp/hf-cache-install.XXXXXX)"
    trap 'rm -rf "$tmp_dir"' EXIT

    echo "Unpacking archive: $ARCHIVE_PATH"
    tar -xzf "$ARCHIVE_PATH" -C "$tmp_dir"

    src_dir="$tmp_dir/$(basename "$HF_HOME")"
    if [ ! -d "$src_dir" ]; then
      echo "Expected cache directory not found in archive: $src_dir" >&2
      exit 1
    fi

    echo "Installing cache into TARGET_HF_HOME=$TARGET_HF_HOME"
    mkdir -p "$TARGET_HF_HOME"
    cp -a "$src_dir"/. "$TARGET_HF_HOME"/
    echo "Install complete."
    ;;

  verify)
    load_hf_token
    if [ ! -d "$TARGET_HF_HOME" ]; then
      echo "Target cache not found: $TARGET_HF_HOME" >&2
      exit 1
    fi
    echo "Verifying offline cache at TARGET_HF_HOME=$TARGET_HF_HOME"
    echo "Verifying offline NLTK cache at TARGET_NLTK_DATA=$TARGET_NLTK_DATA"
    verify_python
    ;;

  all-online)
    load_hf_token
    mkdir -p "$HF_HOME" "$NLTK_DATA"
    prefetch_python
    parent_dir="$(dirname "$HF_HOME")"
    base_dir="$(basename "$HF_HOME")"
    mkdir -p "$(dirname "$ARCHIVE_PATH")"
    tar -C "$parent_dir" -czf "$ARCHIVE_PATH" "$base_dir"
    ls -lh "$ARCHIVE_PATH"
    ;;

  -h|--help|help|"")
    usage
    ;;

  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 2
    ;;
esac
