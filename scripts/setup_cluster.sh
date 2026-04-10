#!/usr/bin/env bash
set -euo pipefail

pick_python() {
  # Allow callers to override interpreter choice explicitly.
  local candidates=()
  if [ -n "${PLAYPEN_PYTHON:-}" ]; then
    candidates+=("${PLAYPEN_PYTHON}")
  fi
  candidates+=(python3.12 python3.11 python3.10 python3)

  for py in "${candidates[@]}"; do
    if ! command -v "${py}" >/dev/null 2>&1; then
      continue
    fi

    if "${py}" -c 'import sys; raise SystemExit(0 if ((3, 10) <= sys.version_info[:2] < (3, 13)) else 1)'; then
      echo "${py}"
      return 0
    fi
  done

  return 1
}

if ! PYTHON_BIN="$(pick_python)"; then
  echo "ERROR: No supported Python interpreter found." >&2
  echo "Please install Python 3.10, 3.11, or 3.12 and re-run this script." >&2
  if command -v python >/dev/null 2>&1; then
    echo "Detected default 'python': $(python -V 2>&1)" >&2
  fi
  exit 1
fi

echo "Using interpreter: ${PYTHON_BIN} ($(${PYTHON_BIN} -V 2>&1))"
"${PYTHON_BIN}" -m venv --clear .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-deps clemcore==3.3.4

python -m pip install \
  "numpy<2,>=1.26.4" \
  "pandas>=2.2,<2.3" \
  "matplotlib>=3.8,<3.9" \
  pyyaml retry tqdm nltk openai anthropic cohere google-genai mistralai seaborn sparklines pylatex ordered-set markdown

python -m pip install "seaborn==0.12.2"
python -m pip install -e . --no-deps

python -m pip install \
  "accelerate==1.4.0" \
  "transformers==4.55.2" \
  "trl==0.21.0" \
  "peft==0.17.0"

if [ ! -d "clembench" ]; then
  git clone https://github.com/clp-research/clembench
fi

tmp_reqs="$(mktemp /tmp/clembench-reqs.XXXXXX.txt)"
trap 'rm -f "${tmp_reqs}"' EXIT
grep -v '^clemcore' clembench/requirements.txt > "${tmp_reqs}"
python -m pip install -r "${tmp_reqs}"

echo '[{"benchmark_path": "clembench"}]' > game_registry.json

echo "Cluster setup complete."
