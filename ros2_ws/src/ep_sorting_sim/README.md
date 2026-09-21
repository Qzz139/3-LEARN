# Gazebo 分拣仿真

此工程在 Ubuntu 20.04 / ROS 2 Foxy / Gazebo Classic 11 上运行，与实机 EP 控制分开。`sorting.world` 使用 [jeguzzi/robomaster_ros](https://github.com/jeguzzi/robomaster_ros) 的 EP 网格和结构（MIT 许可；原始 `robomaster_description/LICENSE` 保留在工程中）。仿真只活动靠底盘的臂关节；夹爪侧连杆固定。六个可碰撞物体和左右各三个放置标记位于世界文件中。球面有简化的网球接缝；瓶子是通用水瓶近似模型，不宣称是农夫山泉官方模型。

臂上相机 `/ep/camera/image_raw` 交给现有 `yolo26m.pt` 检测节点，结果发到 `/ep/detections`。控制程序先按六个已知方位搜索，用检测框对准、用 `/ep/ir/range` 停车，再以 Gazebo 吸附器模拟抓取，倒回中心后按类别转向右侧球区或左侧瓶区。每件物品有独立吸附通道，当前通道将另外五件列为不受力对象，避免邻近物品一起移动。吸附器与标记位属于仿真近似，不用于实机标定。俯视相机为 `/sim/overhead/image_raw`。

## 在 Jetson 上运行

使用干净的 Foxy 终端，避免与 ROS 1 Noetic 环境混用。在仿真项目副本中执行：

```bash
cd ~/projects/3-LEARN-sim
source /opt/ros/foxy/setup.bash
cd ros2_ws && colcon build --symlink-install && cd ..
DISPLAY=:2 bash scripts/run_gazebo_sim_ubuntu.sh log_root:=$PWD/work
```

Launch 默认只显示场景，不会移动车辆。等待终端出现 `YOLO ready`，并确认 `/ep/detections` 已发布后，在另一个 Foxy 终端运行：

```bash
ros2 service call /sim/start std_srvs/srv/Trigger '{}'
```

停止仿真运动：`ros2 service call /sim/stop std_srvs/srv/Trigger '{}'`。每轮输出 `work/gazebo-sort-*/events.jsonl` 与 `summary.json`，其中有阶段、检测类别、停止距离、放置数和失败原因。Jetson 的时钟可能未同步，日志目录名中的 1970 年时间不能当作实际拍摄日期。

## VNC 录制

VNC `:2` 桌面显示 Gazebo 后，从 SSH 或 VNC 终端运行：

```bash
DISPLAY=:2 bash scripts/record_gazebo_vnc_ubuntu.sh work/simulation-demo.mp4
```

录制脚本优先用 FFmpeg；Jetson 未安装 FFmpeg 时自动用 GStreamer。结束录制用 Ctrl+C，等待 MP4 封装完成。录制画面不等同于六件分拣成功；以同一轮的 `summary.json` 为准。
