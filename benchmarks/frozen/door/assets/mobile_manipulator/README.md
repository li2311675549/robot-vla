# 第二阶段完整移动机械臂

可加载模型为 `scenes/mobile_manipulator.xml`，重建入口为 `scripts/build_stage2.py`。
包含 Husky A200 简化参数模型、原版 UR5、Robotiq 2F85、虚拟 D435i RGB-D 和桌面工作站。
小型网格复用现有 UR5/2F85 资源，无新下载。来源和原始许可证见 `third_party/`、`assets/robotiq_2f85/LICENSE` 及 `configs/model_sources.lock.json`。

项目自建部分包括 Husky MJCF 适配、安装座/相机支架/外壳、桌面与测试物、执行器配置和组合文件。
`docs/03_mobile_manipulator_integration.md` 记录坐标、质量、接触参数与限制。
运行 `bash scripts/verify_stage2.sh --record` 生成各 Gate 的实际验收证据。
