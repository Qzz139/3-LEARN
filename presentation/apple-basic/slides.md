---
theme: apple-basic
layout: intro-image-right
image: /assets/ep-camera.jpg
class: cover
colorSchema: light
aspectRatio: 16/9
canvasWidth: 1280
title: RoboMaster EP 视觉分拣
info: 总览、两页实机细节与参考资料
fonts:
  sans: 'Helvetica Neue, PingFang SC, Arial'
  mono: 'Menlo, monospace'
  local: Helvetica Neue,PingFang SC,Arial,Menlo
  provider: none
routerMode: hash
transition: none
drawings:
  enabled: false
mdc: true
---

# RoboMaster EP<br>视觉分拣

Jetson × YOLO26

<div class="cover-caption">网球与水瓶的自主识别、接近、抓取与分类放置</div>

---
layout: default
class: team
---

# 小组分工

| 成员 | 负责内容 |
| :--- | :--- |
| 许轩烨 | 总监、实机程序 |
| 占钟灵 | 仿真、程序 |
| 于瀚 | 模型、程序 |
| 公昕 | 仿真、程序 |
| 袁梓涵 | 场地搭建、PPT 制作 |

---
layout: default
class: overview
---

# 01　总览

<div class="lead">从视觉识别到实际分拣，建立完整任务流程。</div>

<div class="agenda">
<div><span>01</span><h2>总览</h2><p>任务目标、设备连接与系统分工</p></div>
<div><span>02</span><h2>实机细节</h2><p>Pickup → Align → Combine</p></div>
<div><span>03</span><h2>模型训练</h2><p>保留章节背景，后续补充</p></div>
<div><span>04</span><h2>仿真细节</h2><p>保留章节背景，后续补充</p></div>
</div>

---
layout: image-right
image: /assets/sorting-layout.png
class: scene
---

# 两类物品，六个目标

围绕小车摆放 **3 个网球、3 个水瓶**。

相对小车启动朝向：

- **右侧区域放球**。
- **左侧区域放瓶子**。
- 每区固定 **三个角度放置位**。

<div class="note">照片为实物布局；左右由小车启动朝向定义。</div>

---
layout: default
class: connections
---

# 开发与控制连接

<div class="network">
<div><h2>MacBook</h2><p>代码开发<br>SSH：远程终端<br>VNC：屏幕共享</p></div>
<span class="network-arrow">→</span>
<div><h2>Jetson OrinX</h2><p>Python 主程序<br>YOLO 图像推理<br>任务与动作控制</p></div>
<span class="network-arrow">→</span>
<div><h2>RoboMaster EP</h2><p>相机、红外反馈<br>底盘移动与转向<br>机械臂、夹爪执行</p></div>
</div>

<div class="network-caption"><b>SSH / VNC</b>　MacBook 远程控制 Jetson<br><b>Wi-Fi / SDK</b>　Jetson 连接 EP，传递图像、反馈与动作</div>

---
layout: default
class: research
---

# 02　实机细节

<div class="lead">Pickup → Align → Combine：从动作验证到自主分拣</div>

<div class="pipeline">
<div class="stage">
<div class="stage-index">STEP 01 · DEMO 1</div>
<h2>Pickup</h2>
<div class="stage-sub">单次抓取—放置</div>
<div class="stage-body">红外触发 → 夹紧<br>内收搬运 → 转向放置<br>YOLO 独立记录日志</div>
<div class="stage-result">建立可复用的抓放动作</div>
</div>
<div class="flow-arrow">→</div>
<div class="stage">
<div class="stage-index">STEP 02 · DEMO 2</div>
<h2>Align</h2>
<div class="stage-sub">视觉与距离闭环</div>
<div class="stage-body">YOLO 偏差 → 转向对准<br>红外反馈 → 小步接近<br>夹取内收 → 路径回位</div>
<div class="stage-result">处理偏离正前方的单个目标</div>
</div>
<div class="flow-arrow">→</div>
<div class="stage">
<div class="stage-index">STEP 03 · 正式程序</div>
<h2>Combine</h2>
<div class="stage-sub">六方位任务调度</div>
<div class="stage-body">方位状态 × 分类槽位<br>选择最近待取方向<br>复用对准与抓放闭环</div>
<div class="stage-result">球右、瓶左，每区三个位置</div>
</div>
</div>

<div class="research-foot">共用实机约束：ID1 外伸600 / 内收1190；ID2 先设1073后固定。放置后内收，再转向搜索。</div>
<div class="note">单球、单瓶 demo 已实机通过；六件整轮与快速调度待实机验证。舵机数值为原始反馈单位。</div>

---
layout: default
class: core-code
---

# 核心代码：动作、闭环、调度

<div class="code-label">Pickup <span>ir_pick_place_demo.py · 抓取与内收节选</span></div>

```python
self.gripper(False, "grasp_close")
self.move_base(self.arm["base_retracted_raw"], "carry_retract")
```

<div class="code-label">Align <span>yolo_align_pick_place_demo.py · 联合判定节选</span></div>

```python
if stable >= a['stable_frames']:
    distance_state = self.infrared_state()
    if distance_state == 'ready':
        return target
```

<div class="code-label">Combine <span>object_sorting.py · 选方向与完成放置后的状态更新节选</span></div>

```python
direction = nearest_pending_direction(self.pick_directions, current_heading)
# ……完成该方位的抓取、回位、分类放置与位置检查……
counts[class_id] += 1
direction['status'] = 'taken'
```

<div class="note">当前配置：视觉死区±4%；demo2 原始红外阈值25 mm、正式程序24 mm；正式回位容差5 cm。</div>

---
layout: section
class: blank-section
---

<div class="chapter-number">03</div>

# 模型训练

MODEL TRAINING

<!-- 仅保留背景与标题，不填充具体内容。 -->

---
layout: section
class: blank-section
---

<div class="chapter-number">04</div>

# 仿真细节

SIMULATION

<!-- 仅保留背景与标题，不填充具体内容。 -->

---
layout: default
class: references
---

# References

## 01　RoboMaster EP 模型与 ROS 接入

[jeguzzi / robomaster_ros](https://github.com/jeguzzi/robomaster_ros)

<div class="note">EP 仿真所参考的机器人模型与 ROS 2 接入资源。</div>

## 02　Ultralytics YOLO 官方文档

[docs.ultralytics.com/models/yolo26/](https://docs.ultralytics.com/models/yolo26/)

<div class="note">当前实机默认使用普通官方 yolo26m.pt。</div>
