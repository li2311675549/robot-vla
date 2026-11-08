# D435i 模型候选

[Menagerie realsense_d435i](https://github.com/google-deepmind/mujoco_menagerie/tree/main/realsense_d435i)
描述与 Apache-2.0 许可已保存至 `third_party/d435i_reference/`。
固定版本见 `configs/model_sources.lock.json`。其模型是固定相机外壳，**没有 camera 或 IMU sensor 定义**。
完整 OBJ 网格约 44MiB，本阶段未下载，原 MJCF 因而未实际加载。

第二阶段已在机器人支架安装简化外壳和命名为 d435i_rgbd 的理想相机，
实现内外参导出、RGB/米制深度、偏轴反投影和红色测试物三维定位，并用于整机抓取。
这是虚拟 D435i 的主要 RGB-D 功能；未加载上游完整外观，未模拟双目匹配、IR、噪声、IMU 或真实设备量程。
参数与实际误差见 `docs/03_mobile_manipulator_integration.md`。
