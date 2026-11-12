# 文档维护记录

## 内容与图示

实现说明依据感知、规划、任务调度和执行接口整理。流程图分别描述传统控制、三个完整任务及 VLA 驻车抓取；书本操作按“先开抽屉、再取书、放置后关屉”的实际顺序绘制。

历史检查记录保存在以下文件，供维护时追溯：

- [源码与说明对应检查](../evidence/documentation_checks/review.json)
- [流程图检查](../evidence/documentation_checks/flowchart_checks.json)
- [图形渲染检查](../evidence/documentation_checks/render_review_v3.json)
- [代码、数学和模型测试](VALIDATION.md)

流程图生成命令：

```bash
python scripts/generate_architecture_diagrams.py --output /tmp/robot_vla_flowcharts
```

首页预览与两份离线 HTML 均由对应 Markdown 生成，更新文字时应一并更新。
