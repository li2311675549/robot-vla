# Husky A200 模型候选

采用 [husky/husky noetic-devel](https://github.com/husky/husky/tree/noetic-devel/husky_description)，
固定提交见 `configs/model_sources.lock.json`。
已保存 `third_party/husky/` 中的 BSD-3-Clause 许可及主要 xacro 原文。
四个连续轮关节，轮半径 0.1651m、轴距 0.512m、轮距 0.555m（均来自该版本）。

第二阶段已根据这些原文参数生成 `scenes/husky.xml` 与完整整机场景。
保留官方底盘质量/完整惯性、重心、轮尺寸与单轮惯性，使用 primitive 碰撞体；
通过 freejoint、四轮速度执行器与地面接触实现行驶，未下载 DAE 外观资源。
前进、后退、滑移转向、弯线路径和停车已实际验证；摩擦与执行器增益未做真机辨识。
参数、误差与限制见 `docs/03_mobile_manipulator_integration.md`。
