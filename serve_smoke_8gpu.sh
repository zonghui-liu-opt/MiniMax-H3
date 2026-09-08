#!/usr/bin/env bash
set -euo pipefail
# 在独立终端或 tmux 中保持运行；推理客户端退出不会停止本服务。
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$repo_dir/scripts/infer_smoke_8gpu.py" --serve-only "$@"
