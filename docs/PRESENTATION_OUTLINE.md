# 展示与PPT内容建议

这是素材组织提纲，不是新实验结果。

|页|主题|可用材料|表达边界|
|---|---|---|---|
|1|目标与机器人系统|README顶部、三项任务封面|原版UR5；虚拟RGB-D不是完整传感器物理模型|
|2|分层架构|TECHNICAL_REPORT第3节Mermaid|导航/FSM/几何专家与VLA分工|
|3|三任务连续演示|media/demos/door_full、fruit_full、books_full|成功开发实例不等于独立成功率|
|4|大书问题诊断|PROBLEMS_AND_SOLUTIONS前三行|IK、碰撞、坐标系、执行分开|
|5|坐标系误报修复|旧25.032mm/新2.834mm；真实第三书续作|旧中断不是已掉落，续作不是新5m全任务|
|6|香蕉接触与保持控制|docs/media/banana_contact.png，局部0.381m对照|保持控制有效范围；不声称形状是VLA唯一失败原因|
|7|门臂协同|docs/media/door_efficiency.png，36→5抓|同seed仿真时间对照，Oracle宽门|
|8|VLA数据和后训练|docs/media/training_loss.svg，batch4/10000/25517帧|loss不是物理成功率，train-only norm|
|9|实际结果与失败分层|evidence/formal、VLA固定30；三类短技能视频|完整任务、单项和展示构成分开|
|10|局限与后续研究|带载Dock、持物碰撞、VLA目标/接触纠错|只陈述已验证事实和可检验假设|

原始评分和源码快照均保留，可以说明问题如何被定位、如何验证，而无需把局部效果扩大成全部任务高可靠性。
