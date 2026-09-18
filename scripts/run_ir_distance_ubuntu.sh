#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${EP_DEMO_PYTHON:-}" ]]; then
  python_bin="$EP_DEMO_PYTHON"
elif [[ -x /home/robot/venvs/yolo_ros2/bin/python ]]; then
  python_bin=/home/robot/venvs/yolo_ros2/bin/python
else
  python_bin=python3
fi
exec "$python_bin" scripts/record_ir_distance.py "$@"
