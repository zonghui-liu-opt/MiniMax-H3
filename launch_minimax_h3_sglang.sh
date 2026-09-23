#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_dir/scripts/media_tools_env.sh"

# FL2VA (disabled; retained for reference).
# Auto uses folding for the single-request recipes below and can
# select data parallel encoding for a compatible TP1 request batch.
# sglang serve \
#   --model-path /srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3 \
#   --num-gpus 4 \
#   --tp-size 2 \
#   --ulysses-degree 2 \
#   --performance-mode speed \
#   --host 0.0.0.0 \
#   --port 30010 \
#   --model-variant fl2va

# Ref2VA on physical H100 GPUs 4,5,6,7 (logical cuda:0,1,2,3).
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

exec sglang serve \
  --model-path /srv/workspace/Kirin_AI_DataLake/models/MiniMax-H3 \
  --num-gpus 4 \
  --tp-size 2 \
  --ulysses-degree 2 \
  --performance-mode speed \
  --host 0.0.0.0 \
  --port 30010 \
  --model-variant ref2va
