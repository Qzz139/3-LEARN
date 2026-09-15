#!/usr/bin/env bash
# Ubuntu/Jetson entry point. No ROS, model download, calibration or motion required.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null || { echo "Python 3 not found" >&2; exit 2; }
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec "$PYTHON_BIN" "$SCRIPT_DIR/record_joint_state.py" "$@"
fi
command -v timeout >/dev/null || { echo "Run on Ubuntu/Jetson with GNU timeout installed." >&2; exit 2; }
echo "EP state recorder: keep the robot in the pose you want to record."
echo "Default connection: EP Wi-Fi AP. Ctrl-C saves partial results."
# Signal only this recording process group if SDK connection/shutdown hangs.
exec timeout --signal=INT --kill-after=5s "${RECORD_TIMEOUT:-150s}" \
    "$PYTHON_BIN" "$SCRIPT_DIR/record_joint_state.py" "$@"
