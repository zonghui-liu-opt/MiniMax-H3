#!/usr/bin/env bash
set -euo pipefail
# 默认推理 metadata_smoke_v5_multiseed.csv，同一猫咪/动作使用多个显式 seed。
# 默认复用 serve_smoke_8gpu.sh 启动的服务，推理结束后服务继续运行。
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$repo_dir/scripts/infer_smoke_8gpu.py" "$@"
