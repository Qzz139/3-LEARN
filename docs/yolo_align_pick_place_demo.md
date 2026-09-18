# YOLO转向对准网球—抓取放置demo

本demo处理一个网球，验证“物品偏离正前方，底盘根据相机图像自行转向，再根据红外小步接近”。
使用EP原装摄像头与普通官方`yolo26m.pt`。COCO的`sports ball`类别用于本次网球场景。
网球可摆在转正后夹爪前方稍远的位置，默认总前进上限20cm；高度仍需匹配已验证的外伸夹爪高度。
红外装在夹爪前端，触发针对窗口与球面的间隙，不再减去机械臂长度。

## 已确认的动作参数

- 抓夹侧ID2/反馈槽1：先定位raw1073，随后固定。
- 底盘侧ID1/反馈槽0：外伸raw600，内收raw1190。
- 初始及结束：外伸、夹爪松开。
- 红外原始读数正数且≤20mm，连续三次新采样。此值不是尺量2cm的标定。
- 放置：启动朝向右侧90°，放置后返回启动朝向。
- 异常：请求停止、身体红灯、不自动回位。SDK的servo.pause禁用控制，不保证保持姿态。

## 运行过程

启动时加载并预热YOLO，之后进入初始态。每次图像推理时底盘静止。
选择画面中最靠近抓取轴的网球，以检测框中心横坐标计算偏差。
默认抓取轴为画面宽度的50%，即1280像素画面的x=640；当前实机截图显示夹爪大致位于这里。
`axis_x_ratio`可调整摄像头与夹爪的水平偏移，不能将默认轴视为精确相机外参标定。

球在左侧时SDK正角度左转，球在右侧时负角度右转；按误差比例计算，每次1–5°。
每次转动结束等待0.8秒，再读取newest图像重新识别。超过2秒的推理结果不用于控制。
连续三帧中心位于抓取轴±2.5%画面宽度内，再检查三个不同时间戳的新红外采样。
连续三次正数且≤20mm才夹紧；连续三次高于20mm则以0.02m/s前进约5mm后停车。
运动中只要一次读数≤20mm就立即请求停车，随后重新进行三帧图像与三次红外确认，不能直接夹紧。
每次前进后居中计数清零，停车0.4秒重新识别，偏离抓取轴则先纠正角度。
红外混合近/远读数时等待，不自动倒车寻找所谓“实际2cm”。零值、过期反馈、里程无进展、
超过前进上限或总对准时间120秒均停止亮红灯，不自动回位。

平移使用`drive_speed`，指令失联看门狗0.2秒，每25ms检查里程和红外。
没有采用`chassis.move(x=..., xy_speed=0.02)`，因为官方SDK会把其平移速度钳制到至少0.5m/s。
使用20Hz的`sub_position`位置反馈控制每段停止并记录实际路径，拒绝超过0.25秒的里程和红外反馈。
5mm是目标步长，实际停车会有反馈采样、制动及打滑误差，不保证严格5mm。

首次未识别到球时在启动朝向±60°内扫描。选中后用同类别、中心位移和面积连续性保持目标；
连续三帧丢失就停止，不自动改抓另一个球。这是单目标连续性匹配，多球遮挡时不能保证身份。

抓取后ID1内收，先按前进时记录的朝向和里程倒序退回各路段起点，再利用实际底盘yaw反馈
转至“启动朝向−90°”，外伸松爪放置；
然后内收、返回启动朝向、外伸松爪。比如对准时已左转30°，去放置方向需右转约120°。
实测本EP的SDK正转角对应yaw读数减少，`yaw_feedback_sign=-1`将实际姿态差
转换为与指令一致的朝向差；不能直接用原始yaw差做补偿。
返回路段起点的默认容差2mm，起始中心容差1cm；偏离路径、超时或中心误差过大则停止，
不在偏移位置继续放置。此处依靠轮式里程，不能消除真实地面打滑造成的定位误差。
后续六物品分拣仍需明确两类各自对应的放置区域。

## 命令与文件

```bash
cd ~/projects/3-LEARN
# 预览，不连接机器人
bash scripts/run_yolo_align_demo_ubuntu.sh
# 只看相机和YOLO结果，不发送运动或LED命令
bash scripts/run_yolo_align_demo_ubuntu.sh --observe
# 单次自动对准—抓取—放置
bash scripts/run_yolo_align_demo_ubuntu.sh --execute
```

代码：`scripts/yolo_align_pick_place_demo.py`，复用原demo的舵机、夹爪和底盘动作封装。
参数：`config/yolo_align_pick_place.json`。Ubuntu启动脚本使用原有Jetson Python环境。
距离参数位于`approach`：`step_m`步长、`speed_mps`前进速度、`return_speed_mps`返回速度、
`max_travel_m`总前进上限。红外触发仍使用`infrared.threshold_mm=20`，它是实测原始读数阈值，
不是实际2cm；官方标称量程下限10cm，近距可靠性仍需要实机验证。
日志：`work/yolo-align-*/run.json`、`detections.jsonl`、`latest_camera.jpg`、`latest_detections.jpg`。
标注图含检测框与抓取轴，方便核对“识别到了哪里、为何转动”。时间控制使用monotonic；
Jetson未同步系统时钟时，目录与UTC日期仍可能显示1970年。

接口依据：[Ultralytics检测输出](https://docs.ultralytics.com/modes/predict/)、
[大疆SDK底盘角度及姿态接口](https://robomaster-dev.readthedocs.io/en/latest/python_sdk/robomaster.html#module-robomaster.chassis)。
[SDK平移速度限制源码](https://github.com/dji-sdk/RoboMaster-SDK/blob/master/src/robomaster/chassis.py)、
[官方红外量程](https://www.dji.com/support/product/robomaster-ep-core)。

本地测试：`python3 -m unittest discover -s tests -v`。

实测进度：只读识别已检出左侧网球；首次自动测试转向两次后因目标连续性丢失停止，
尚未完成自动抓取。实测发现yaw符号与指令相反，已修正；详见[实测记录](yolo_alignment_hardware_tests.md)。
新增距离调整与返回中心功能已通过模拟反馈测试，尚未完成此版本的实机抓放验证。
