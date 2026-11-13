# 复现、资料下载与跨系统使用

## 只阅读报告和制作PPT

其他系统不需要GPU或MuJoCo。浏览GitHub README，下载本仓库，阅读docs/TECHNICAL_REPORT.md与PROBLEMS_AND_SOLUTIONS.md；图表位于docs/media，原始统计在evidence，代表视频在media/demos。完整展示集合已发布为[v1.0.0附件](https://github.com/li2311675549/robot-vla/releases/tag/v1.0.0)，需另行下载。

```bash
git clone https://github.com/li2311675549/robot-vla.git
cd robot-vla
```

## 本地Linux仿真

已验证Ubuntu22.04/Python3.10/MuJoCo3.6.0。新机器使用独立环境，不修改已有Python、CUDA或驱动。已有环境可直接运行；新环境参考：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
MUJOCO_GL=egl python -m pytest -q
```

无界面渲染需要 EGL 支持。模型资源已包含在仓库中。Windows 可直接阅读文档和视频，仿真运行尚未在 Windows 上单独验证。

新开发回合使用不存在的输出目录：

```bash
MUJOCO_GL=egl python scripts/run_stage7_isolated.py  --task door --mode oracle --seed 2101 --door-profile aligned_handles_v3  --config configs/door_continuous_contact_frozen_v4.json  --output outputs/new_door_demo --record

MUJOCO_GL=egl python scripts/run_stage7_isolated.py  --task fruit --mode vision --seed 412  --config configs/stage11_fruit_formal_baseline_v1.json  --output outputs/new_fruit_demo --record

MUJOCO_GL=egl python scripts/run_stage7_isolated.py  --task drawer --mode oracle --seed 200009 --book-profile distinct_v2  --config configs/stage20_books_scatter_frozen_v28.json  --output outputs/new_books_demo --record
```

上述命令用于单次开发运行。评估改进效果时，另选未参与调参的种子，并固定配置。

## VLA训练与推理

服务器使用独立Python3.11 OpenPI环境，源码和锁文件见server/snapshots/actual_training_20261108。实际batch4/10000更新配置在configs/pi05_multifruit_stage22_batch4_v2.json，norm见服务器快照checkpoint9999_assets。权重、Tokenizer和完整训练数据需作者另行提供，未随GitHub发布；视频不能代替动作标签。

训练和推理入口分别为 `scripts/stage22_launch_training.sh`、`scripts/stage22_serve_policy.sh`。启动前按脚本要求配置资源限制与运行开关，并将 `ROBOT_EXAM_STAGE22_ROOT`、`ROBOT_EXAM_BASE_ROOT` 设置为本机数据路径。

当前模型评估为驻车状态下指定目标抓起保持。`evaluate_multifruit_policy_phase.py` 连接模型服务，并将预测动作送入机械臂执行器。底盘导航和完整任务调度仍由传统模块负责。

## 核对本说明中的计算与模型入口

使用现有仿真环境，在项目根目录执行：

```bash
MUJOCO_GL=egl python scripts/verify_documented_principles.py --output outputs/new_documentation_check --model-smoke
```

脚本检查反投影、书本几何和相对坐标等 5 项计算，并加载三个模型各推进 50 步，用于确认计算与场景初始化。输出目录需尚未存在。

## 运行版本化的10回合评测

评测入口为`scripts/evaluate_frozen.py --protocol <协议JSON> --output <新目录>`。门/水果/书本协议分别为`configs/door_continuous_contact_formal_v4.json`、`configs/stage11_fruit_formal_protocol_v1.json`、`configs/stage20_books_scatter_formal_v28_protocol.json`。每次保存代码版本、协议与全部回合结果。检验新策略时应使用未参与调参的种子。

当前版本的运行结果另存新目录，并记录源码版本、配置与全部回合结果。

Release v1.0.0 保留发布时附件，最新文字说明以 main 分支为准。
