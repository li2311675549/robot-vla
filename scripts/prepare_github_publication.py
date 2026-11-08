#!/usr/bin/env python3
"""Build a source-only publication plus a small set of authentic continuous demos."""
import argparse,hashlib,json,os,re,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();assert not out.exists()
 assert shutil.disk_usage(ROOT).free>25.15*1024**3
 base=Path('/home/li/桌面/移动机械臂_传统方法与VLA_技术代码资料_2026-11-08/03_GitHub源码/robot_exam')
 shutil.copytree(base,out,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache','outputs'))
 for folder in ['src','scripts','tests','configs','assets','scenes','third_party','tools']:
  for f in (ROOT/folder).rglob('*'):
   if not f.is_file() or f.is_symlink() or '__pycache__' in f.parts or f.suffix in ('.pyc','.pyo'):continue
   assert f.stat().st_size<5*1024**2;copy(f,out/f.relative_to(ROOT))
 for name,target in [('83_system_technical_summary.md','TECHNICAL_REPORT.md'),('84_problem_solution_evidence.md','PROBLEMS_AND_SOLUTIONS.md')]:copy(ROOT/'docs'/name,out/'docs'/target)
 issues={
  'third_book_fix.json':'docs/results/stage10_third_book_fix_acceptance.json',
  'book_geometry.json':'docs/results/stage20_precision_book_milestone_v2.json',
  'book_swept_path.json':'docs/results/stage20_local_resume_quality_v4.json',
  'books_formal_summary.json':'outputs/stage20/books_scatter_formal_v28/summary.json',
  'door_contact_acceptance.json':'docs/results/door_continuous_contact_acceptance_v1.json',
  'door_efficiency_comparison.json':'outputs/stage14/continuous_door/analysis_final_v1/comparison.json',
  'banana_full_closure.json':'outputs/stage7/banana_video_seed420/full_failure/result.json',
  'banana_regulated_hold.json':'outputs/stage7/banana_video_seed420/regulated_success/result.json',
  'vla_training.json':'docs/results/stage22_training10000_completion_v1.json',
  'vla_fixed30.json':'docs/results/stage24_tenfruit_fixed30_acceptance_v1.json',
  'traditional_object_showcase.json':'docs/results/stage25_traditional_object70_display_v1.json',
  'vla_showcase.json':'docs/results/stage25_display30_composition_v1.json'}
 for name,source in issues.items():copy(ROOT/source,out/'evidence/issues'/name)
 for group,path in [('door','outputs/stage14/continuous_door/formal_contact_v4'),('fruit','outputs/stage11/fruit_formal_baseline_v1'),('books','outputs/stage20/books_scatter_formal_v28')]:
  for name in ['summary.json','episodes.json','freeze.json','source_integrity.json']:
   if (ROOT/path/name).is_file():copy(ROOT/path/name,out/'evidence/formal'/group/name)
 copy(ROOT/'outputs/stage22/training10000_metrics_v1/loss_windows100.csv',out/'evidence/training/loss_windows100.csv')
 source_demos=[
  ('door','Door: open, enter, close','outputs/stage14/continuous_door/full_seed2101_recorded_v3',210,'full_task_oracle_wide_door'),
  ('fruit','Ten-fruit sorting','outputs/stage7/fruit_seed412_complete_recorded',265,'full_task_development_legacy_layout'),
  ('books','Three books and return home','outputs/stage20/books_dynamic_edge_dev200009_v27_recorded',1010,'full_task_development_distinct_books')]
 rows=[]
 for name,title,rel,frame_time,scope in source_demos:
  parent=ROOT/rel;result=json.loads((parent/'result.json').read_text());assert result['success']
  src=parent/'video/demo.mp4';dst=out/'media/demos'/(name+'_full.mp4');dst.parent.mkdir(parents=True,exist_ok=True)
  os.link(src,dst);copy(parent/'result.json',out/'media/records'/(name+'_full.json'))
  image=out/'docs/media'/(name+'.png');image.parent.mkdir(parents=True,exist_ok=True)
  subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-ss',str(frame_time),'-i',str(src),'-frames:v','1',str(image)],check=True,timeout=40)
  rows.append(dict(id=name,title=title,scope=scope,source=str(src),video=str(dst.relative_to(out)),sha256=sha(dst),result=str(Path('media/records')/(name+'_full.json')),sim_seconds=result['sim_seconds']))
 display=Path('/home/li/桌面/VLA三类水果_30条真实测试展示记录_2026-11-08');m=json.loads((display/'展示记录清单.json').read_text())
 for category in ['apple','banana','orange']:
  r=next(x for x in m['videos'] if x['category']==category and x['success']);src=display/r['display_video'];dst=out/'media/demos'/('vla_'+category+'.mp4');os.link(src,dst);assert sha(dst)==r['video_sha256']
  copy(Path(r['result']),out/'media/records'/('vla_'+category+'.json'))
  image=out/'docs/media'/('vla_'+category+'.png')
  subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-sseof','-1','-i',str(src),'-frames:v','1',str(image)],check=True,timeout=30)
  rows.append(dict(id='vla_'+category,title='VLA instructed '+category+' grasp',scope='parked_single_target_grasp_hold_no_place',source=r['video'],video=str(dst.relative_to(out)),sha256=sha(dst),result=str(Path('media/records')/('vla_'+category+'.json')),scene_seed=r['scene_seed'],sim_seconds=r['sim_seconds']))
 put(out/'media/index.json',json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 for source,name in [('outputs/stage7/plots/banana_closure_seed420.png','banana_contact.png'),('outputs/stage14/continuous_door/analysis_final_v1/comparison.png','door_efficiency.png')]:copy(ROOT/source,out/'docs/media'/name)
 hero='''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="320" viewBox="0 0 1200 320"><defs><linearGradient id="bg"><stop stop-color="#0b1220"/><stop offset="1" stop-color="#18344a"/></linearGradient></defs><rect width="1200" height="320" rx="22" fill="url(#bg)"/><path d="M0 260H1200M0 210H1200M0 160H1200" stroke="#fff" opacity=".035"/><text x="60" y="66" font-family="Arial,sans-serif" font-size="17" letter-spacing="4" fill="#67e8f9">MOBILE MANIPULATION · MUJOCO · VLA</text><text x="56" y="145" font-family="Arial,sans-serif" font-weight="bold" font-size="64" fill="#f8fafc">Husky × UR5</text><text x="60" y="193" font-family="Arial,sans-serif" font-size="23" fill="#cbd5e1">Plan motion. Reason about contact. Learn actions.</text><g font-family="Arial,sans-serif" font-size="16" fill="#cbd5e1"><text x="60" y="265">A* NAVIGATION</text><text x="285" y="265">PHYSICAL CONTACT</text><text x="555" y="265">DUAL-VIEW π0.5</text></g><g fill="none" stroke="#67e8f9" stroke-width="11" stroke-linecap="round" stroke-linejoin="round"><path d="M910 226V176L841 120L884 69L977 109"/><path d="M977 109L1007 91M977 109L1005 132"/></g><g fill="#0b1220" stroke="#67e8f9" stroke-width="5"><circle cx="910" cy="176" r="11"/><circle cx="841" cy="120" r="11"/><circle cx="884" cy="69" r="11"/></g><rect x="829" y="226" width="186" height="42" rx="9" fill="#fbbf24"/><circle cx="863" cy="275" r="20" fill="#334155" stroke="#94a3b8" stroke-width="4"/><circle cx="982" cy="275" r="20" fill="#334155" stroke="#94a3b8" stroke-width="4"/></svg>'''
 put(out/'docs/media/hero.svg',hero)
 readme='''<div align="center">
<img src="docs/media/hero.svg" alt="Husky and original UR5 mobile manipulation" width="100%" />

# Robot VLA

**真实接触移动操作 · 几何规划控制 · 双视角 π₀.₅**

`Husky A200` · `Original UR5` · `Robotiq 2F-85` · `MuJoCo 3.6.0`

[技术总结](docs/TECHNICAL_REPORT.md) · [问题与解决措施](docs/PROBLEMS_AND_SOLUTIONS.md) · [视频演示](#视频演示) · [运行说明](docs/RUNNING.md) · [实验原始记录](evidence/)
</div>

## 项目概览

在同一个MuJoCo物理系统中完成门操作、十果分类、三书抽屉收纳，并将双视角π₀.₅接入正常机械臂执行器。传统导航/FSM与学习型短技能共存；没有通过物体瞬移、隐藏焊接或被动关节驱动伪造接触操作。

<table><tr><td width="33%"><b>移动与规划</b><br/>A*、外形膨胀、真实轮驱动、Dock与操作驻车</td><td width="33%"><b>接触与操作</b><br/>多初值IK、持物碰撞、推书侧抓、门臂协同</td><td width="33%"><b>视觉语言动作</b><br/>双RGB、实际状态、绝对关节动作、真实后训练与闭环推理</td></tr></table>

## 视频演示

点击封面查看或下载真实连续视频。前三项是完整传统任务开发演示，后三项是驻车指定目标抓起保持；VLA录像没有放置或完整十果分类。演示实例不替代下方固定评测。

<table>
<tr><td width="33%" align="center"><a href="media/demos/door_full.mp4"><img src="docs/media/door.png" width="100%" alt="开门入室关门"/></a><br/><b>开门 · 入室 · 关门</b><br/>锁边把手 / Oracle / 宽门洞</td><td width="33%" align="center"><a href="media/demos/fruit_full.mp4"><img src="docs/media/fruit.png" width="100%" alt="十果完整分类"/></a><br/><b>10果连续分类</b><br/>开发回合 / 全部正确入筐</td><td width="33%" align="center"><a href="media/demos/books_full.mp4"><img src="docs/media/books.png" width="100%" alt="三书收纳与返航"/></a><br/><b>三书 · 三层 · 返航</b><br/>120/180/240mm长书本 / Oracle</td></tr>
<tr><td align="center"><a href="media/demos/vla_apple.mp4"><img src="docs/media/vla_apple.png" width="100%" alt="VLA苹果"/></a><br/><b>π₀.₅ 苹果抓取</b></td><td align="center"><a href="media/demos/vla_banana.mp4"><img src="docs/media/vla_banana.png" width="100%" alt="VLA香蕉"/></a><br/><b>π₀.₅ 香蕉抓取</b></td><td align="center"><a href="media/demos/vla_orange.mp4"><img src="docs/media/vla_orange.png" width="100%" alt="VLA橘子"/></a><br/><b>π₀.₅ 橘子抓取</b></td></tr>
</table>

[逐段评分与来源](media/index.json)。浏览器不支持直接播放时，可下载MP4后播放。更多案例集拟作为`v1.0.0`发布附件，发布状态见[Releases](https://github.com/li2311675549/robot-vla/releases)。

## 架构

```mermaid
flowchart LR
    TASK[任务FSM] --> NAV[A*与轮式导航]
    TASK --> GEO[几何专家与RGB-D]
    TASK --> VLA[双视角π₀.₅]
    GEO --> IK[IK与持物碰撞路径]
    VLA --> ACT[动作块与安全执行]
    IK --> CTRL[关节控制器]
    ACT --> CTRL
    NAV --> SIM[MuJoCo真实物理]
    CTRL --> SIM
    SIM --> FEEDBACK[相机、状态与接触反馈]
    FEEDBACK --> TASK
```

VLA只学习驻车机械臂七维动作；门/抽屉约束控制与底盘导航保留传统方案。机身理想RGB-D与腕部虚拟RGB不是完整D435i物理传感器模型。门/抽屉和部分书本使用声明的Oracle几何，接触/容纳评分有仿真辅助。

## 关键问题与解决措施

|问题|采取的方法|实际证据|
|---|---|---|
|第三书仍夹住却被判滑移|世界向量改为工具坐标下的相对位移，原阈值保留|同一时刻旧误差25.032mm、新误差2.834mm；重新实抓续作完成|
|大书侧抓与柜体碰撞|先推后侧抓、有向几何抓点、真实净空上界、持物完整扫掠与已验路径执行|多个完整开发回合；冻结散落协议仍有失败|
|香蕉力大仍滚转|区分抓取闭合与保持闭合，接触/位姿反馈|seed420局部保持0.70完成0.381m倒车；漂移18.12→3.04mm|
|门操作重复重抓|底盘/机械臂圆弧协同、安全触发重抓、协调退出|同初态36→5抓，2144.86→462.336仿真秒|
|数据与执行不一致|高层绝对目标/reference/ctrl分开，train-only匹配chunk统计与双频物理重放|实际batch4/10000更新，9999完整保存重载与物理执行|

以上是相应局部/开发对照结论，不外推成全部场景成功率。[完整问题证据链](docs/PROBLEMS_AND_SOLUTIONS.md)。

## 评测结果

|协议|结果|成功回合平均仿真时间|边界|
|---|---:|---:|---|
|正常门完整任务|10/10|451.237s|冻结保留种子，Oracle，1.57m门洞|
|历史布局完整十果分类|2/10|1771.290s|3000s协议，RGB-D及仿真辅助验证|
|新散落三尺寸书收纳返航|4/10|1207.823s|冻结保留种子，Oracle|
|VLA十果桌面指定目标抓取保持|14/30|11.300s|6验证布局与预声明噪声，不含放置|

传统已执行水果抬升为53/57（苹果20/23、香蕉15/16、橘子18/18），含重试和提前终止，与完整任务分母不同。VLA固定验证按类为2/10、6/10、6/10。展示集按成功/失败数量整理，含重复初态和历史诊断，不用其构成估计独立成功率。

本地清理后源码导出曾实际通过258/258测试，日志随证据保存；当前发布版会另附验证记录。模块测试、数据重放和loss下降都不等于完整任务通过。

## VLA数据与训练

- 输入：front/wrist RGB、实测六关节与夹爪、语言；不输入隐藏物体位姿。
- 动作：六关节高层绝对目标+夹爪，5Hz，10步chunk；内部32维填充。
- 实际训练：405片段、25517帧；旧单果20%/新多果80%加权采样。新十果数据为单目标抓取/放置，不是完整十果清空示教。
- 后训练：π₀.₅ LoRA配置，batch4、10000有效更新；最终checkpoint9999，validation flow loss0.01256467。
- norm：只用train，匹配采样和chunk；SHA `2315f953b3ee8949915f315b9c938ef6f0ca7a5e9397562d82c238d90b4c67c8`。
- 学习效果：已有真实抓取实例，尚未证明优于同条件专家或完成学习型完整分类。

[动作接口](docs/VLA_TECHNICAL.md) · [数据与训练](docs/VLA_DATA_TRAINING.md) · [实际服务器源码](server/snapshots/actual_training_20261108/)

## 快速运行

本地仿真使用Python3.10；服务器OpenPI使用独立Python3.11环境。不要把两套依赖混装。

```bash
# 在项目根目录，使用已安装的仿真环境
MUJOCO_GL=egl python -m pytest -q

# 新开发输出；不覆盖已有冻结正式结果
MUJOCO_GL=egl python scripts/run_stage7_isolated.py \
  --task door --mode oracle --seed 2101 --door-profile aligned_handles_v3 \
  --config configs/door_continuous_contact_frozen_v4.json \
  --output outputs/new_door_demo --record
```

首次安装参考`requirements.lock.txt`、`environment.yml`和`scripts/setup_env.sh`。VLA推理/训练另需合法权重、Tokenizer、原始数据、对应GPU及独立环境，仓库不会自动下载或启动训练。

## 目录与复现

```text
src/robot_exam/   控制、导航、感知、任务FSM、数据和VLA
scripts/         构建、运行、评估、采数、训练、推理和资料导出
configs/         模型来源、场景、控制器与冻结协议
assets/ scenes/ third_party/   小型模型与原始许可
server/          实际训练源码、OpenPI、锁文件与norm快照
media/           代表性连续视频、评分和来源
evidence/        原始统计与问题修复证据
docs/            最终技术总结、问题措施、运行说明
```

模型权重、完整训练数据、虚拟环境、缓存、认证信息和私有开发日记不在仓库中。第三方代码/模型遵循各自许可证；自研根许可证由作者另行确定。这里是MuJoCo仿真工程，不代表真实机器人部署结果。
'''
 put(out/'README.md',readme)
 with (out/'.gitignore').open('a') as f:f.write('\n# Explicit small representative demo assets\n!media/\n!media/demos/\n!media/demos/*.mp4\n')
 put(out/'.gitattributes','*.mp4 -text\n*.png -text\n*.stl -text\n*.md text eol=lf\n*.py text eol=lf\n')
 report={'status':'LOCAL_PUBLICATION_CANDIDATE_NOT_YET_UPLOADED','repository':'li2311675549/robot-vla','visibility_requested':'public','source_anchor_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'demos':rows,'model_weights_included':False,'raw_dataset_included':False,'new_training_or_inference':False,'total_file_bytes':sum(f.stat().st_size for f in out.rglob('*') if f.is_file())}
 put(ROOT/'outputs/stage26/publication_preparation_v1.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='demos'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
