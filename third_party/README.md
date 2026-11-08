# 第三方来源与边界

所有下载文件的来源、提交、字节数和 Git blob SHA1 见 `configs/model_sources.lock.json`。
`scripts/fetch_models.py` 只下载清单中的约 3.6MB 文件，保留原文并拒绝覆盖内容不同的文件。
测试同时复核清单。不会下载整个仓库或其他机械臂资源。

| 路径 | 上游 | 许可 | 本阶段用途 |
|---|---|---|---|
| universal_robot/ur_description | ROS-Industrial universal_robot | BSD-3-Clause，见目录内 LICENSE | 原版 UR5 描述、参数与 STL；派生 MJCF 在 assets/ur5 |
| husky | Clearpath husky/husky | BSD-3-Clause，见 LICENSE | A200 描述原文检查，尚未转换 |
| ../assets/robotiq_2f85 | DeepMind Menagerie，来源 ROS-Industrial Robotiq | BSD-2-Clause，见 LICENSE | 原 MJCF/网格；自建 smoke.xml 避开上游悬挂方块示例 |
| d435i_reference | DeepMind Menagerie，来源 RealSense | Apache-2.0，见 LICENSE | 仅描述/许可，缺少外观网格，未加载 |

许可按具体模型目录核验，不以 Menagerie 仓库级说明替代。
