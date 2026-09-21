# RoboMaster EP 六物品分拣 ROS 2 工程

此工作区将已验证的抓放动作保留在 `ep_sorting/legacy`，通过 ROS 2 接口运行：

| 节点 | 接口 | 作用 |
| --- | --- | --- |
| `ep_camera` | `/ep/camera/image_raw` (`sensor_msgs/Image`) | EP 夹爪旁相机发布 BGR 图像 |
| `ep_detector` | `/ep/detections` (`vision_msgs/Detection2DArray`) | `yolo26m.pt` 检测网球（COCO 32）和水瓶（COCO 39） |
| `ep_arm_controller` | `/ep/sort_objects` (`ep_sorting_interfaces/action/SortObjects`) | 串行控制底盘、机械臂和夹爪，发布阶段反馈 |
| `ep_sorting_task` | Action 客户端 | 可选地发送一次完整分拣任务 |

阶段顺序由 [`config/sorting_state_machine.yaml`](../../config/sorting_state_machine.yaml) 定义。初始/终止状态、转移事件和失败分支会在启动时校验，实际运行时每次转移都写入 `run.json`。舵机 ID、限位、红外阈值、搜索方向、左右各三个放置朝向仍取自 [`config/object_sorting.json`](../../config/object_sorting.json)；不使用桌面网格。物品放置后按当前航向选择最近的待取方向。控制器仍执行原来的低速靠近、停止后复核、抓取、退回起始中心及分类放置流程。

## 环境和构建

目标设备为 Ubuntu 20.04 / Python 3.8，可安装 ROS 2 Foxy；**不要在同一个 shell 中同时 source ROS 1 Noetic 和 ROS 2 Foxy**。Foxy 已停止维护；若设备升级 Ubuntu，需对相应 ROS 2 发行版重新构建和验证。本工程依赖 `rclpy`、`sensor_msgs`、`vision_msgs`、`launch_ros`、`rosidl_default_generators`、`PyYAML`、`numpy`、`robomaster`、`ultralytics`，以及 EP 视频解码所需的 OpenCV。Jetson 的 PyTorch/Ultralytics 请使用与其 JetPack 匹配的版本，不在本工程里替换。

```bash
source /opt/ros/foxy/setup.bash
python3 -c 'import rclpy, yaml, numpy, cv2, robomaster, ultralytics; from vision_msgs.msg import Detection2DArray'
cd ~/projects/3-LEARN/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 interface show ep_sorting_interfaces/action/SortObjects
```

如果导入检查失败，应先在设备上补齐对应依赖。模型 `models/yolo26m.pt` 随 `ep_sorting` 安装；构建时须保留仓库根目录的 `scripts/`、`config/` 和 `models/`。

## 启动和日志

```bash
# 启动相机、检测、动作服务和任务客户端；默认不会发出运动命令
ros2 launch ep_sorting sorting.launch.py

# 另一个终端显式执行一次分拣
ros2 action send_goal /ep/sort_objects ep_sorting_interfaces/action/SortObjects '{execute: true}' --feedback

# 或启动后自动执行一次（仅在已准备好实机和物品时使用）
ros2 launch ep_sorting sorting.launch.py execute:=true
```

`log_root` 默认为 `~/projects/3-LEARN/work`，可在 launch 时覆盖，例如 `log_root:=/tmp/ep-work`。每轮生成 `ros2-sort-*` 目录和 `run.json`；其中包含检测相关阶段、抓取/放置结果、状态转移、错误和最终统计。异常时按原实机逻辑停车并将底盘 LED 置红；不会自动复位机械臂。Action 同时只接受一项任务；取消请求会在下一个流程记录点停止。

`sorting.launch.py` 仍只包装**真机 EP**。另有独立的 [Gazebo 仿真包](src/ep_sorting_sim/README.md)，使用上游 EP 外形、同一个 YOLO 检测节点和仿真控制节点，不会连接实机。该包已在 Jetson 上构建并启动；仿真运行结果请以对应的 `summary.json` 判断，不能由模型显示或录屏推断六件分拣成功。
