#!/usr/bin/env bash
# Ubuntu/Jetson红外只读测量入口，所有参数原样传给Python。
# 脚本出错、变量未定义或管道任一命令失败时退出。
set -euo pipefail
# 切到仓库根目录，使模型和配置的相对路径保持一致。
cd "$(dirname "$0")/.."
# Python优先使用EP_DEMO_PYTHON，其次使用Jetson现有虚拟环境，最后回退python3。
if [[ -n "${EP_DEMO_PYTHON:-}" ]]; then
  python_bin="$EP_DEMO_PYTHON"
elif [[ -x /home/robot/venvs/yolo_ros2/bin/python ]]; then
  python_bin=/home/robot/venvs/yolo_ros2/bin/python
else
  python_bin=python3
fi
# 用Python替换当前shell，保留退出码和终止信号的传递。
exec "$python_bin" scripts/record_ir_distance.py "$@"
