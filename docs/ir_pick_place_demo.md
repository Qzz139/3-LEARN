# 红外单次抓取—放置 demo

动作顺序固定为：

1. 抓夹连杆侧舵机先定位到记录值 `raw=601` 附近，随后不再向它发送位置命令。
2. 底盘肩部侧舵机外伸到 `raw=1073` 附近，夹爪松开，进入初始态。
3. 第一号红外反馈连续 3 帧不大于 30 mm 后，夹爪收紧。
4. 底盘侧舵机内收到 `raw=552`，底盘右转 90°。
5. 底盘侧舵机外伸，夹爪松开放置；机械臂内收，底盘左转 90°。
6. 底盘侧舵机外伸、夹爪保持松开，回到初始态。

`raw=552` 是根据当前 `601/1073` 姿态和 EP 连杆关节限位得到的首轮内收值，
实机低速测试后只需修改 `config/ir_pick_place.json` 中的
`arm.base_retracted_raw`。红外安装在夹爪上方并朝夹取方向，因此 30 mm 使用传感器到
目标表面的原始距离；当前使用反馈槽 0，可通过 `infrared.feedback_slot` 改端口。

官方 Python SDK 的 `Servo.moveto` 接收整数角度，所以三个记录值实际分别编码为最接近的
`600`、`1070` 和 `550`；反馈验收仍围绕配置中的原始目标值，并允许 `8` 个 raw 单位
（0.8°）误差。

YOLO 使用独立推理进程，默认模型为 `yolo26m.pt`。检测结果写入每次运行目录的
`detections.jsonl`，抓放状态机不读取检测结果；模型缺失、加载失败或推理异常均不会触发
抓放失败。

Ubuntu 启动脚本优先使用 Jetson 上现有的
`/home/robot/venvs/yolo_ros2/bin/python`（该环境同时包含 RoboMaster SDK 与
Ultralytics）；也可通过 `EP_DEMO_PYTHON=/path/to/python` 指定环境。

先预览配置和动作：

```bash
./scripts/run_ir_demo_ubuntu.sh
```

实机运行：

```bash
./scripts/run_ir_demo_ubuntu.sh --execute
```

程序异常时停止底盘、停止底盘侧舵机与夹爪，底盘装甲 LED 常亮红色；异常路径不会
自动回初始态。正常结束后机械臂外伸、夹爪松开并熄灭底盘 LED。
