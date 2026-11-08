# 原版 UR5，非 UR5e

`ur5.xml` 和 `scene.xml` 由 `scripts/build_ur5.py` 生成。上游为
[ROS-Industrial universal_robot](https://github.com/ros-industrial/universal_robot)，
提交与原文件 Git blob 校验值见 `configs/model_sources.lock.json`。
源描述和 BSD-3-Clause 许可位于 `third_party/universal_robot/ur_description/`。

使用原版 UR5 的 `config/ur5`、六个旋转关节、质量惯性与 collision STL。
不使用 `config/ur5e` 或 UR5e 网格。上游顶层 ur5.xacro 注释误写 UR5e，
实际 include 为 `ur5_macro.xacro`，本项目进一步通过连杆长度/质量测试核对。

本项目新增六个位置执行器、关节阻尼、home keyframe、地面和照明；
排除底座与肩部的机械装配重叠碰撞。物理碰撞网格同时用于显示，未下载 COLLADA 外观模型。
上游惯性包含圆柱近似，base mass 也有上游不确定性注释；不是厂商标定动力学。

已验证：六轴独立执行器运动/跟踪、限位/力矩约束、RGB-D、视频和 GUI。
第二阶段新增：独立 DH/FK 核对、带姿态约束的 DLS IK、2F85 法兰装配、物理负载抓放与整机演示，见 `scenes/mobile_manipulator.xml`。
仍未验证：自碰撞全工作空间、通用避碰规划与硬件等效精度。
