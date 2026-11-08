# 复现、资料下载与跨系统使用

## 只阅读报告和制作PPT

其他系统不需要GPU或MuJoCo。浏览GitHub README，下载本仓库，阅读docs/TECHNICAL_REPORT.md与PROBLEMS_AND_SOLUTIONS.md；图表位于docs/media，原始统计在evidence，代表视频在media/demos。完整展示集合拟发布为v1.0.0附件，实际发布状态以Releases为准。

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

EGL需本机实际支持。模型资源已包含，不自动下载大数据。Windows仅阅读资料无需环境；本仓库没有宣称Windows图形/物理链路已独立验证。

新开发回合使用不存在的输出目录：

```bash
MUJOCO_GL=egl python scripts/run_stage7_isolated.py  --task door --mode oracle --seed 2101 --door-profile aligned_handles_v3  --config configs/door_continuous_contact_frozen_v4.json  --output outputs/new_door_demo --record

MUJOCO_GL=egl python scripts/run_stage7_isolated.py  --task fruit --mode vision --seed 412  --config configs/stage11_fruit_formal_baseline_v1.json  --output outputs/new_fruit_demo --record

MUJOCO_GL=egl python scripts/run_stage7_isolated.py  --task drawer --mode oracle --seed 200009 --book-profile distinct_v2  --config configs/stage20_books_scatter_frozen_v28.json  --output outputs/new_books_demo --record
```

开发示例不保证每次成功，也不等于重新开展正式评测。旧正式版本原始源码见benchmarks/frozen，不能假设旧工作区提交SHA会存在于新发布仓库历史中。已曝光正式种子不用于调参。

## VLA训练与推理

服务器使用独立Python3.11 OpenPI环境，源码和锁文件见server/snapshots/actual_training_20261108。实际batch4/10000更新配置在configs/pi05_multifruit_stage22_batch4_v2.json，norm见服务器快照checkpoint9999_assets。权重、Tokenizer和完整训练数据需作者另行提供，未随GitHub发布；视频不能代替动作标签。

训练/推理分别使用scripts/stage22_launch_training.sh、stage22_serve_policy.sh。脚本具有资源及显式授权守门，不因clone仓库自动运行。按实际数据盘路径设置ROBOT_EXAM_STAGE22_ROOT与ROBOT_EXAM_BASE_ROOT，不照搬旧实例地址或PID。

当前评估仅驻车指定目标抓起保持；底盘、完整分类FSM和传统专家独立保留。真实模型服务可通过evaluate_multifruit_policy_phase.py连接正常执行器，不改运行中qpos。
