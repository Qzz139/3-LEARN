#!/usr/bin/env bash
set -eo pipefail

# Keep ROS 1 Noetic paths out of the ROS 2 simulation shell.
if [[ "${ROS_DISTRO:-}" != "" && "${ROS_DISTRO}" != "foxy" ]]; then
  echo "Current ROS_DISTRO=${ROS_DISTRO}; open a fresh shell before sourcing Foxy." >&2
  exit 1
fi
if [[ ! -f /opt/ros/foxy/setup.bash ]]; then
  echo "ROS 2 Foxy is missing: /opt/ros/foxy/setup.bash" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/foxy/setup.bash
# Reuse the Jetson's tested YOLO/PyTorch environment from system ROS Python.
venv_site=/home/robot/venvs/yolo_ros2/lib/python3.8/site-packages
if [[ -d "$venv_site" ]]; then
  export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${venv_site}"
fi
if ! ros2 pkg prefix gazebo_ros >/dev/null 2>&1; then
  echo "gazebo_ros is missing. Install ros-foxy-gazebo-ros-pkgs first." >&2
  exit 1
fi
if [[ ! -f "${repo_root}/ros2_ws/install/setup.bash" ]]; then
  echo "Build once: cd ${repo_root}/ros2_ws && colcon build --symlink-install" >&2
  exit 1
fi
source "${repo_root}/ros2_ws/install/setup.bash"
# Resolve the upstream RoboMaster EP meshes embedded in sorting.world.
export GAZEBO_MODEL_PATH="${repo_root}/ros2_ws/src:${GAZEBO_MODEL_PATH:-}"
set -u
exec ros2 launch ep_sorting_sim simulation.launch.py "$@"
