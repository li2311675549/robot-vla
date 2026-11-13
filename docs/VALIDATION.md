# 代码与模型测试记录

## 自动化测试

最近保存的完整测试记录为 **259 项通过，耗时 96.85 秒**，运行环境为 Python 3.10 与 MuJoCo。记录见[完整测试日志](../evidence/documentation_checks/pytest_final_v2.log)。

首次运行有一项失败：临时目录位于另一个 Git 工程内，影响了“无 Git 源码导出目录”测试的前提。改用独立临时目录后，单项复测与完整测试均通过。

[首次运行](../evidence/documentation_checks/pytest_final_v1.log) · [单项复测](../evidence/documentation_checks/pytest_revision_environment_v2.log)

## 数学与模型检查

|检查|内容|
|---|---|
|几何计算|反投影、书边截面、侧夹方向和工具坐标等 5 项算例|
|场景加载|门、水果、三书模型各运行 50 个物理步，模型警告计数为 0|
|执行器|每个场景 11 个执行器；门和抽屉由接触驱动|
|逆运动学|在独立运动学副本中求解距当前 TCP 5 mm 的目标，达到既定容差|

这些检查覆盖函数计算、模型初始化和接口；任务完成情况见[实验结果](FINAL_RESULTS.md)。

在已安装的仿真环境中运行：

```bash
MUJOCO_GL=egl python scripts/verify_documented_principles.py \
  --output outputs/new_documentation_check --model-smoke
```

输出目录需尚未存在。[数学与模型记录](../evidence/documentation_checks/principles_and_model_checks.json) · [IK 记录](../evidence/documentation_checks/ik_scratch_check.json)

## 版本与原始记录

历史任务源码保存在 [benchmarks/frozen](../benchmarks/frozen/)，训练源码及归一化文件保存在[服务器快照](../server/snapshots/actual_training_20261108/)。文件校验、文档检查和渲染记录集中在[维护记录](DOCUMENTATION_REVIEW.md)。
