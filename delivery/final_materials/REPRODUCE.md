# 运行与代码使用

## 完整源码

`03_GitHub源码/robot_exam/`为完整导出，包含模型资产、控制、任务、VLA、脚本、测试与许可证；不含虚拟环境、权重、缓存和原始数据。传统方法与VLA共享安全执行器接口，因此完整项目只按一个工程运行。各自资料目录另有核心代码阅读副本/导航。

本机已有环境可在导出根目录运行：

```bash
MUJOCO_GL=egl /home/li/projects/robot_exam/.venv/bin/python -m pytest -q

# 使用新的输出目录，以下是开发复现，不覆盖冻结正式结果。
MUJOCO_GL=egl /home/li/projects/robot_exam/.venv/bin/python scripts/run_stage7_isolated.py \
  --task door --mode oracle --seed 2101 --door-profile aligned_handles_v3 \
  --config configs/door_continuous_contact_frozen_v4.json \
  --output outputs/new_door_demo --record

MUJOCO_GL=egl /home/li/projects/robot_exam/.venv/bin/python scripts/run_stage7_isolated.py \
  --task fruit --mode vision --seed 412 \
  --config configs/stage11_fruit_formal_baseline_v1.json \
  --output outputs/new_fruit_demo --record

MUJOCO_GL=egl /home/li/projects/robot_exam/.venv/bin/python scripts/run_stage7_isolated.py \
  --task drawer --mode oracle --seed 200009 --book-profile distinct_v2 \
  --config configs/stage20_books_scatter_frozen_v28.json \
  --output outputs/new_books_demo --record
```

新机器按`requirements.lock.txt`、`environment.yml`与`scripts/setup_env.sh`创建独立Python3.10仿真环境。源中保留历史脚本，不能按旧运行状态自动启动。部分数据测试需要外置桌面数据；服务器模型测试还需独立Python3.11环境、权重和GPU。

## 训练与推理

实际服务器源码位于VLA目录及GitHub源码中的`server/snapshots/actual_training_20261108/`，包含项目冻结代码、OpenPI和训练工具。当前项目源码与训练冻结源码不是同一Git版本，来源清单分别保存。

查看`scripts/stage22_launch_training.sh`、`stage22_serve_policy.sh`和`openpi_ur5_train.py --help`，按数据盘实际路径设置`ROBOT_EXAM_STAGE22_ROOT`、`ROBOT_EXAM_BASE_ROOT`。不能直接复制旧实例地址/历史PID启动。模型服务仅在用户需要时手动启动，本次没有启动训练或推理。

VLA评估入口`scripts/evaluate_multifruit_policy_phase.py`需要合法初始化fixture、真实模型服务、9999检查点和匹配norm；只恢复初态一次，后续通过合法关节执行器控制。完整数据仍位于同桌面的`VLA多水果双视角数据_2026-11-06`，本包只附数据索引/统计，视频不能代替关节标签。

## 上传GitHub

只上传`03_GitHub源码/robot_exam/`即可获得代码与精简技术说明。不要直接上传整个桌面资料包，里面有媒体和原始实验归档。训练权重、Tokenizer、原始图像/深度、账户认证不在源码导出中。原作者尚需确定自研根许可证，第三方许可证必须保留。

本次仅准备本地文件，没有创建远程仓库、上传数据或发布项目。数据/模型如需公开，另行核对分发许可和大小。
