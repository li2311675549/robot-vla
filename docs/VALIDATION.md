# 发布目录验证

当前公开目录已在既有Python3.10/MuJoCo环境实际运行完整回归：**259项全部通过，96.85秒**，含此前单独验证的ZIP辅助项。控制、模型与评分核心未改。此前258PASS/97.29秒及独立ZIP1PASS/0.02秒为旧验证记录，不再用于代替本轮完整结果。

首次完整回归为258通过、1失败（97.20秒）：验收将basetemp放入另一父Git工程，使一项“无Git源码导出目录”测试识别到父仓库，改变其前提。未修改源码或测试；改用独立/tmp目录后单项1PASS，再完整259PASS。原始[首次日志](../evidence/documentation_checks/pytest_final_v1.log)、[单项复核](../evidence/documentation_checks/pytest_revision_environment_v2.log)、[完整复核](../evidence/documentation_checks/pytest_final_v2.log)保留。

三项冻结历史源码52/47/65项SHA完整匹配，位于benchmarks/frozen；实际服务器源码快照1457文件均已SHA验证。发布版没有权重/原始数据/环境/认证信息。代表录像是实际连续物理回合，结果与范围见media/index.json。示例视频不代替冻结评测。

## 本轮README实现说明核验

- 源码一致性：11个相关公开模块与本地工程逐文件SHA一致。
- 数学核对：实际调用反投影、抓边几何、侧夹方向和工具坐标函数，共5项检查通过。
- 模型入口：门/水果/三书模型实际加载，各通过50个2ms物理步；各场景11执行器，无mocap/焊接，门和抽屉关节无直接执行器；各自data.warning计数为0。
- 这些是数学/初始化检查，不是新增抓取、导航或完整任务成功率。原258项源码导出回归与本轮文档检查分开记录。
- 复现本轮轻量检查：`python scripts/verify_documented_principles.py --output outputs/new_documentation_check --model-smoke`。

- 补充IK检查：在独立运动学数据中求一个距当前TCP 5mm的可达目标，达到原精确IK容差，实时qpos保持完全不变；不据此声称真实执行精度。

[四轮文档校核摘要](../evidence/documentation_checks/review.json) · [数学与模型记录](../evidence/documentation_checks/principles_and_model_checks.json) · [IK记录](../evidence/documentation_checks/ik_scratch_check.json)。
