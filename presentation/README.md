# 分拣项目演示稿

15 页，16:9 横版。采用 Marp，自定义简洁黑白主题。

- `sorting-presentation.html`：图片已嵌入，可独立打开；方向键翻页，F 全屏。
- `sorting-presentation.pdf`：横版 PDF，保留可选文字和可点击的引用链接。
- `sorting-presentation.md`：演示稿内容；小组分工已按提供信息填写。
- `theme.css`：版式与字体。
- `assets/`：项目真实照片，包含本次提供的场地布局图。

修改 Markdown 或 CSS 后，安装 Node.js、Python 3 和 Chrome，再执行：

```bash
bash presentation/export.sh
```

首次执行通过 npx 下载 Marp CLI 4.5.1。也可用 `MARP_BIN=/path/to/marp` 指定已有 CLI。
训练与仿真章节仅保留背景、标题；未填写训练或仿真成果。

内容核对依据：当前 `scripts/`、`config/` 和 `docs/`，以及项目聊天中的实机反馈。
引用页只有指定的两个链接：

- https://github.com/jeguzzi/robomaster_ros
- https://docs.ultralytics.com/models/yolo26/

`robomaster_ros` 提供 ROS 2 驱动与机器人描述，本稿将其作为 EP 模型与 ROS 接入参考。
默认使用普通官方 `yolo26m.pt`；本稿不声称已完成自定义训练。
单物体 demo 已通过现场确认；六件整轮与快速调度仍待实机验证。
