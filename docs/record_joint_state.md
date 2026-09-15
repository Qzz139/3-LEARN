# 记录当前机械臂姿态

在 Jetson/Ubuntu 上运行，不依赖 ROS 2 或 YOLO。只连接 SDK 并订阅反馈；
不调用回中、标定、运动、夹爪开合、舵机暂停或模式切换。

先让 Jetson 连上 EP 的 `RMEP-21c2f5` 网络，保持需要记录的姿态，并关闭其他占用
EP SDK 的程序。若 Wi-Fi 已保存，可在 Jetson 终端执行：

```bash
sudo nmcli connection up id RMEP-21c2f5 ifname wlan0
```

在仓库根目录执行：

```bash
bash scripts/record_joint_state.sh --label current-pose
```

默认采集 5 秒、10 Hz。记录两个姿态时使用不同标签，例如 `pose-a`、`pose-b`；
程序不会把照片中的两种姿态自动转换成关节数据。多网卡需要时可指定
`--local-ip <Jetson在EP无线网络上的IP>`；该地址不是 Mac 到 Jetson 的 SSH 地址。
非 AP 接入可指定 `--conn-type sta` 或 `--conn-type rndis`。

如使用虚拟环境，设置 `PYTHON_BIN=/绝对路径/venv/bin/python3`。
脚本使用 Jetson 已安装的 RoboMaster SDK，不自动安装或升级软件。

输出位于 `records/joint_states/<UTC时间>-<标签>/`：

- `samples.jsonl`：每次反馈的接收时间，以及舵机在线标志、原始速度和原始角度，
  或机械臂末端原始 x/y 值。
- `snapshot.json`：最后一次反馈、采集数量、各在线舵机角度变化范围和错误信息。

只有至少两个舵机在线、舵机及末端均有至少两次反馈、最后反馈不超过 1 秒且无错误时，
结果才标为 `complete`；它表示反馈采集完整，不表示已经完成物理关节标定或姿态稳定验证。
数据不足会保存 `partial` 并返回非零退出码；无读数不会生成虚构角度。
连接或 SDK 关闭卡住时，启动脚本最多等待 150 秒后中止自己的录制进程。

四个舵机槽位按 SDK DDS 数组下标 0–3 保存，不擅自对应到控制命令中的舵机编号。
原始角度不能直接当成度数传给 `servo.moveto`。末端 x/y 的含义是前后/上下，
保持 SDK 原值（包括可能出现的无符号整数表示），不是底盘水平面的 x/y。

你的控制目标是近底盘关节运动、远端关节保持。当前先保存原始基准状态；
`joint_mapping` 保持空值。照片不能确认主动舵机与连杆关节的对应关系，
也不能证明固定一个舵机就能保持指定远端关节相对角度；后续结合机构模型和实测确认。
本程序不施加锁定力，也不重放姿态。
