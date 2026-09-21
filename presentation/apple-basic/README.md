# Apple Basic 版演示稿

采用官方 `@slidev/theme-apple-basic` 主题，10 页、16:9 横版。

- `apple-basic.html`：独立 HTML，图片、样式和脚本已内嵌，直接打开即可；方向键翻页。
- `apple-basic.pdf`：横版 PDF。
- `slides.md`：可编辑源稿。
- `styles/index.css`：中文字体、字号与学术导图样式。
- `preview.jpg`：全部页面预览。

实机细节只有两页：第6页展示 **Pickup → Align → Combine** 导图，第7页展示对应的核心代码节选。
小组分工、设备连接与两个引用保留；模型训练、仿真细节只保留背景与标题。

修改后重新导出（需 Node.js 22+、Python 3 和 Chrome）：

```bash
cd presentation/apple-basic
npm install --ignore-scripts
CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" bash export.sh
```

Ubuntu 可将 `CHROME_BIN` 改为本机 Chrome/Chromium 可执行文件路径。
若已安装 Playwright Chromium，可以省略 `CHROME_BIN`。

主题参考：https://github.com/slidevjs/themes/tree/main/packages/theme-apple-basic
使用官方主题的标题、图片与章节布局，另加适合本项目的导图和代码页排版。
