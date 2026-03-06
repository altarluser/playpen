#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
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

grep -v '^clemcore' clembench/requirements.txt > /tmp/clembench-py312.txt
python -m pip install -r /tmp/clembench-py312.txt

echo '[{"benchmark_path": "clembench"}]' > game_registry.json

echo "Cluster setup complete."
