#!/usr/bin/env bash
# Ubuntu/Jetson关节只读记录入口，支持超时中断后保存部分结果。
# Ubuntu/Jetson entry point. No ROS, model download, calibration or motion required.
# 脚本出错、变量未定义或管道任一命令失败时退出。
set -euo pipefail
# 使用脚本自身所在目录定位Python文件，与调用时的当前目录无关。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 允许通过PYTHON_BIN指定已安装RoboMaster SDK的Python环境。
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null || { echo "Python 3 not found" >&2; exit 2; }
# 帮助模式直接交给Python，不要求GNU timeout命令可用。
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec "$PYTHON_BIN" "$SCRIPT_DIR/record_joint_state.py" "$@"
fi
# 实机记录使用GNU timeout限制SDK连接或关闭卡住时的等待时间。
command -v timeout >/dev/null || { echo "Run on Ubuntu/Jetson with GNU timeout installed." >&2; exit 2; }
echo "EP state recorder: keep the robot in the pose you want to record."
echo "Default connection: EP Wi-Fi AP. Ctrl-C saves partial results."
# Signal only this recording process group if SDK connection/shutdown hangs.
exec timeout --signal=INT --kill-after=5s "${RECORD_TIMEOUT:-150s}" \
    "$PYTHON_BIN" "$SCRIPT_DIR/record_joint_state.py" "$@"
