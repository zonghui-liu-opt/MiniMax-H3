#!/usr/bin/env bash
set -euo pipefail
# 默认仅推理 metadata_smoke_v3.csv，以 input_image 作为唯一首帧条件。
# 默认复用 serve_smoke_8gpu.sh 启动的服务，推理结束后服务继续运行。
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$repo_dir/scripts/infer_smoke_8gpu.py" "$@"
