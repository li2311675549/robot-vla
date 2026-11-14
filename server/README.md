# VLA 服务器代码与资产

- `openpi/`：实际训练环境保留的 OpenPI 源码、Python 3.11 依赖与 `uv.lock`；保留上游许可。
- `assets/original_ur5/norm_stats.json`：已训练检查点使用的归一化统计。
- 本项目训练和服务入口在根目录 `scripts/stage22_launch_training.sh`、`stage22_serve_policy.sh`；原版 UR5 映射在 `src/robot_exam/vla/`。

该目录由原实际训练快照整理，OpenPI 源文件和锁文件逐字节保留；与主目录重复的机器人项目副本和过时下载助手已移出当前版本，原快照仍可由 Git 历史恢复。权重、Tokenizer 与数据集不随仓库发布。运行环境、数据和执行语义见 [VLA说明](../docs/VLA.md) 与 [运行说明](../docs/RUNNING.md)。

仿真使用 Python 3.10，服务器使用独立 Python 3.11 环境。启动脚本通过 `ROBOT_EXAM_BASE_ROOT/openpi` 寻找 OpenPI：部署时将这里的 `openpi/` 放置或链接到该路径，再按锁文件准备环境；不要将两套 Python 依赖混装。
