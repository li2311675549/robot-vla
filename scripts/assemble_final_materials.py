#!/usr/bin/env python3
"""Assemble final technical/source/evidence materials locally; never alter metrics."""
import argparse,csv,hashlib,io,json,re,shutil,subprocess,tarfile
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text())
def write(p,text):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
def dump(p,value):write(p,json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def frozen_source(base,destination):
 freeze=load(base/'freeze.json');commit=freeze['git_commit']
 names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=ROOT,text=True).splitlines()
 allowed={'src','scripts','configs','assets','scenes','third_party'}
 files=[n for n in names if Path(n).parts[0] in allowed or n in ('pyproject.toml','requirements.in','requirements.lock.txt','environment.yml')]
 payload=subprocess.check_output(['git','archive',commit,'--',*files],cwd=ROOT)
 destination.mkdir(parents=True)
 with tarfile.open(fileobj=io.BytesIO(payload)) as bundle:
  for member in bundle.getmembers():
   if member.isdir():continue
   path=destination/member.name
   if not member.isfile() or not path.resolve().is_relative_to(destination.resolve()):raise ValueError('Unsafe frozen source')
   path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(bundle.extractfile(member).read());path.chmod(member.mode&0o777)
 mismatches=[]
 for name,expected in freeze['source_hashes'].items():
  p=destination/name
  if not p.is_file() or sha(p)!=expected:mismatches.append(name)
 receipt=dict(git_commit=commit,checked_frozen_files=len(freeze['source_hashes']),mismatches=mismatches,status='PASS_EXACT_FROZEN_SOURCE' if not mismatches else 'COMMIT_SNAPSHOT_DIFFERS_FROM_FROZEN_WORKTREE')
 dump(destination/'FROZEN_SOURCE_VERIFICATION.json',receipt)
 return receipt
def scan(root):
 hits=[]
 for p in root.rglob('*'):
  if not p.is_file():continue
  if p.suffix in ('.mp4','.png','.jpg','.stl','.obj','.npz'):continue
  b=p.read_bytes()
  if re.search(rb'-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----\r?\n[A-Za-z0-9+/=\r\n]+-----END',b):hits.append(str(p.relative_to(root)))
  if re.search(rb'\b(?:ghp_|github_pat_|sk-proj-|hf_)[A-Za-z0-9_]{24,}',b):hits.append(str(p.relative_to(root)))
 return sorted(set(hits))
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve()
 if out.exists():raise FileExistsError('Do not overwrite previous deliverables')
 if shutil.disk_usage(ROOT).free<26.2*1024**3:raise RuntimeError('Need26.2GiB before bounded media/source copy')
 tr=out/'01_传统方法';vl=out/'02_VLA';pub=out/'03_GitHub源码/robot_exam';guide=out/'04_运行与材料索引';out.mkdir(parents=True)
 docs=ROOT/'delivery/final_materials'
 for source,target in [('TRADITIONAL_TECHNICAL.md',tr/'技术文件/01_建模与算法.md'),('VLA_TECHNICAL.md',vl/'技术文件/01_架构与动作接口.md'),('VLA_DATA_TRAINING.md',vl/'技术文件/02_数据与训练.md'),('REQUIREMENTS_AND_LIMITS.md',guide/'任务要求与能力边界.md'),('REPRODUCE.md',guide/'运行与GitHub说明.md')]:copy(docs/source,target)
 # Whitelisted current project source: preserve relative import/model paths.
 skip={'.git','.venv','__pycache__','.pytest_cache','.cache','.ruff_cache','node_modules'}
 for folder in ['src','scripts','tests','configs','assets','scenes','third_party','tools']:
  for p in sorted((ROOT/folder).rglob('*')):
   rel=p.relative_to(ROOT)
   if not p.is_file() or p.is_symlink() or any(x in skip or x.endswith('.egg-info') for x in rel.parts) or p.suffix in ('.pyc','.pyo'):continue
   if p.stat().st_size>5*1024**2:raise ValueError('Review large source file '+str(rel))
   if p.name.startswith('.env') or p.suffix in ('.pem','.key','.safetensors','.ckpt','.pt','.pth','.mp4'):raise ValueError('Excluded file in source '+str(rel))
   copy(p,pub/rel)
 for name in ['.gitignore','pyproject.toml','environment.yml','requirements.in','requirements.lock.txt','requirements-vla-client.in','requirements-openpi-server-extra.in']:copy(ROOT/name,pub/name)
 shutil.copytree(ROOT/'delivery/final_materials',pub/'delivery/final_materials')
 # Actual current server code, not the older historical snapshot.
 snap=ROOT/'outputs/stage25/server_source_snapshot_v1'
 shutil.copytree(snap/'source',vl/'服务器训练源码/source')
 copy(snap/'verification.json',vl/'服务器训练源码/verification.json')
 shutil.copytree(snap/'source',pub/'server/snapshots/actual_training_20261108')
 copy(snap/'verification.json',pub/'server/snapshots/actual_training_20261108/DOWNLOAD_VERIFICATION.json')
 for name in ['TRADITIONAL_TECHNICAL.md','VLA_TECHNICAL.md','VLA_DATA_TRAINING.md','REQUIREMENTS_AND_LIMITS.md']:copy(docs/name,pub/'docs'/name)
 # Reading excerpts are separate from runnable full source.
 for folder in ['control','navigation','perception','tasks','skills']:
  for p in (ROOT/'src/robot_exam'/folder).glob('*.py'):copy(p,tr/'核心代码阅读'/folder/p.name)
 for p in (ROOT/'src/robot_exam/vla').glob('*.py'):copy(p,vl/'本地执行接口代码'/p.name)
 write(tr/'代码入口.md','# 传统方法代码入口\n\n完整可运行工程见../03_GitHub源码/robot_exam。核心代码阅读/为分模块阅读副本，不单独安装。构建入口scripts/build_stage2.py和build_tasks.py，运行run_stage7_isolated.py，评分tasks/scoring.py。\n')
 write(vl/'代码入口.md','# VLA代码入口\n\n服务器训练源码/source/是本次从无卡实例只读下载的实际训练项目、OpenPI和工具；完整项目在../03_GitHub源码/robot_exam。\n\n本地执行接口代码/为阅读副本；实际运行使用完整工程src/robot_exam/vla。配置见完整工程configs/pi05_multifruit_stage22_batch4_v2.json，统计见服务器训练源码/source/checkpoint9999_assets/original_ur5/norm_stats.json。\n')
 # Recompute three independent task metrics from the actual ten result records.
 suites={'door':'outputs/stage14/continuous_door/formal_contact_v4','fruit':'outputs/stage11/fruit_formal_baseline_v1','books':'outputs/stage20/books_scatter_formal_v28'}
 names={'door':'门开关','fruit':'十果分类','books':'新散落三书收纳返航'};metrics={};rows=[];frozen={}
 for task,root in suites.items():
  base=ROOT/root;episodes=load(base/'episodes.json');assert len(episodes)==10
  successes=[e for e in episodes if e['success']]
  published=load(base/'summary.json')['summary'];assert published['complete_successes']==len(successes)
  metrics[task]=dict(task=names[task],attempts=10,successes=len(successes),success_rate=len(successes)/10,mean_success_sim_seconds=sum(e['sim_seconds'] for e in successes)/len(successes),mean_success_wall_seconds=sum(e['wall_seconds'] for e in successes)/len(successes),failure_categories=dict(Counter(e.get('root_failure') or e.get('failure_category') for e in episodes if not e['success'])),source=root)
  dest=tr/'评估结果'/task
  frozen[task]=frozen_source(base,tr/'冻结评测源码'/task)
  for name in ['summary.json','episodes.json','freeze.json','source_integrity.json']:
   if (base/name).is_file():copy(base/name,dest/name)
  for e in episodes:
   p=base/str(e['seed'])/'result.json';raw=load(p);assert raw['success']==e['success'];copy(p,dest/str(e['seed'])/'result.json')
   rows.append(dict(task=task,seed=e['seed'],success=e['success'],sim_seconds=e['sim_seconds'],wall_seconds=e['wall_seconds'],failure_category=e.get('failure_category'),original_result=str(p.relative_to(ROOT)),result_sha256=sha(p)))
 dump(tr/'评估结果/三任务完整结果.json',metrics)
 with (tr/'评估结果/三任务逐回合.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 table=['# 三类完整任务：实际冻结结果','', '|任务|成功/回合|成功率|成功均时(仿真秒)|成功均时(墙钟秒)|','|---|---|---|---|---|']
 for task,m in metrics.items():table.append(f"|{m['task']}|{m['successes']}/10|{m['success_rate']:.0%}|{m['mean_success_sim_seconds']:.3f}|{m['mean_success_wall_seconds']:.3f}|")
 table+=['','门为Oracle正常锁边把手/1.57m宽门协议；水果为历史布局RGB-D加仿真辅助验证；三书为新版散落、120×80×15/180×120×20/240×160×25mm、Oracle。不同协议不混算。','', '视频目录按展示数量整理、含重复与开发版本；不能用视频成功比例替代本表独立10回合结果。历史窄幅小册子10/10亦不替代新版三书4/10。','', '水果主要失败：接触/搬运滑落、Dock；新版书本：推移跟踪5次、持书路径碰撞预检1次。原始回合和冻结配置见评估结果/。']
 write(tr/'技术文件/02_测试结果与局限.md','\n'.join(table)+'\n')
 copy(tr/'技术文件/02_测试结果与局限.md',pub/'docs/TRADITIONAL_RESULTS.md')
 dump(pub/'evidence/current_benchmarks.json',metrics)
 # Copy complete existing video packs, retaining original selection disclosures.
 shutil.copytree(Path('/home/li/桌面/移动机械臂三任务_45段视频_2026-11-06'),tr/'完整任务视频_45段')
 shutil.copytree(Path('/home/li/桌面/VLA十果桌面指令抓取_40段视频_2026-11-08'),vl/'十果桌面指令抓取视频_40段')
 for name in ['stage24_tenfruit_fixed30_acceptance_v1.json','stage24_tenfruit_each10_video_delivery_v1.json','stage22_training10000_completion_v1.json','stage23_training_scene_count_audit_v1.json','stage21_data_release_qualification_v1.json']:
  copy(ROOT/'docs/results'/name,vl/'实验结果'/name)
 fixed=ROOT/'outputs/stage24/ten_fruit_validation30_v1';copy(fixed/'summary.json',vl/'实验结果/固定30回合原始summary.json')
 ledger=ROOT/'outputs/stage24/tenfruit_collection40_success10_v1';copy(ledger/'all_attempts.json',vl/'实验结果/全部68回合.json')
 for i,row in enumerate(load(ledger/'all_attempts.json')):copy(Path(row['result']),vl/'实验结果/全部68逐回合'/f"{i:03d}_{row['category']}.json")
 shutil.copytree(ROOT/'outputs/stage22/training10000_metrics_v1',vl/'实验结果/训练曲线原始数值')
 data=Path('/home/li/桌面/VLA多水果双视角数据_2026-11-06')
 for name in ['training_data_spec.json','manifest_new_portable.json','manifest_old_portable.json','数据说明.md','数据交付校验.json']:copy(data/name,vl/'数据索引'/name)
 write(vl/'数据索引/原始数据位置.md','# 原始数据位置\n\n完整RGB、腕RGB、深度、关节动作和LeRobot文件保留在同桌面VLA多水果双视角数据_2026-11-06目录。此处只复制索引，不重复搬运全部原始数据，也不把原始数据纳入GitHub。实际训练只用405训练片段/25517帧，验证/测试隔离。\n')
 result=load(ROOT/'docs/results/stage24_tenfruit_fixed30_acceptance_v1.json')
 vs=['# VLA实际验证与展示视频','', '|类别|固定验证成功/尝试|固定验证成功率|本次展示成功视频|','|---|---|---|---|']
 for c,label in [('apple','苹果'),('banana','香蕉'),('orange','橘子')]:
  r=result['by_class'][c];vs.append(f"|{label}|{r['successes']}/{r['attempts']}|{r['observed_success_rate']:.0%}|10|")
 vs+=['','固定30合计14/30(46.7%)，6个初始验证布局、预声明RNG，非30独立布局；没有动用正式test。','', '40视频包含30成功10失败，累计68次真实执行30成功38失败；所有失败保留。成功布局数苹果1、香蕉4、橘子5，含已知初态重复。不能把展示构成当连续独立评测成功率。','', '成功定义：指定目标抬升至少55mm、末1秒双侧接触至少90%、末帧仍夹持，并通过驻车/旁物保护。不是落筐或完整十果分类。']
 write(vl/'技术文件/03_验证结果与适用范围.md','\n'.join(vs)+'\n');copy(vl/'技术文件/03_验证结果与适用范围.md',pub/'docs/VLA_RESULTS.md')
 # Media-independent public landing page.
 public='''# Mobile Manipulation with MuJoCo and π₀.₅

Husky A200 + original UR5 + Robotiq 2F-85, body RGB-D and virtual wrist RGB. Traditional planning/control and learned stationary manipulation share the same legal actuator execution layer.

## Traditional tasks

Door opening/entry/closing, randomized ten-fruit sorting, and three distinct books in three drawers with return home have continuous successful demonstrations. Frozen independent task results are door10/10, historical ten-fruit2/10, scattered three-books4/10. These are different protocols, not an overall success percentage. Door uses Oracle geometry and1.57m passage; book sizes120×80×15/180×120×20/240×160×25mm.

## Learned manipulation

π₀.₅ post-training: batch4,10000 optimizer updates,405 training clips/25517frames, old20%/new80% weighted sampling, dualRGB + measured7D state + language, absolute6joint+gripper goals,5Hz/10-step chunks. Checkpoint9999 loaded and physically evaluated. Fixed ten-fruit-table stationary target-grasp validation14/30 (apple2/10,banana6/10,orange6/10). Separate quota videos are illustrative repetitions and do not estimate an independent success rate. No complete learned ten-fruit sorting claim.

## Code

- `src/robot_exam/navigation`, `control`, `perception`, `tasks`: A*, dock/parking, multi-start IK, contact manipulation, FSM, scoring.
- `src/robot_exam/vla`, `data`: synchronized demonstrations, LeRobot/OpenPI mapping, training, inference and action execution.
- `scripts`, `configs`, `assets`, `scenes`, `third_party`, `tests`: runnable tools, versioned protocols, models, licences and regression.
- `server/snapshots/actual_training_20261108`: actual server project/OpenPI/tool source, checked file hashes. No weights or dataset.
- `docs`: final technical descriptions and measured results, without daily development diaries.

Local simulation uses Python3.10/MuJoCo3.6.0; server OpenPI uses a separate Python3.11 environment. See requirements.lock.txt/environment.yml and server uv.lock. From an activated simulation environment, run `MUJOCO_GL=egl python -m pytest -q`. Dataset/model-dependent integration checks require separately supplied assets. See docs for task entry points and scope.

Runtime acts only through actuators and physical contact; initialization and scratch planning are separate. No object teleportation or hidden grasp welding. Preserve all failed trials and keep development/validation/test distinct. Simulator warning counters and textual solver logs are different evidence.

Videos and raw datasets are distributed separately. This source export contains no credentials, model weights, tokenizer, caches or virtual environments. Self-authored root licence remains for the author to choose; preserve all third-party licences, including OpenPI and model assets. No automatic upload or remote publication is performed.
'''
 write(pub/'README.md',public)
 write(pub/'docs/RUNNING.md',(docs/'REPRODUCE.md').read_text().replace('`03_GitHub源码/robot_exam/`','本目录').replace('`03_GitHub源码/robot_exam/`','本目录'))
 readme='''# 移动机械臂项目：技术、代码与实验材料

本目录是制作报告/PPT及后续GitHub整理的最终资料包，不含每天开发流水账。

- **01_传统方法**：建模参数、导航/IK/接触/FSM算法、核心代码阅读副本、三个任务独立10回合原始结果、45段完整任务展示视频。
- **02_VLA**：双视角接口、数据/归一化/训练配置、无卡服务器实际源码、训练数值记录、固定30回合结果、全68回合账本、40段十果桌面指令抓取视频。
- **03_GitHub源码/robot_exam**：完整可运行源工程、必要模型/许可证、测试与精简技术文档；后续上传GitHub使用此目录，不上传整个媒体资料包。
- **04_运行与材料索引**：任务定义、能力边界、复现与上传说明。

## 结果口径

传统完整任务：门10/10；历史十果2/10；新版散落三书4/10。VLA固定十果桌面抓取30回合：苹果2/10、香蕉6/10、橘子6/10，合计14/30。VLA40视频按每类10成功整理，包含重复已知初态；不能用展示比例替代上述固定评测。

三类传统完整任务都有真实连续成功演示；VLA展示只是驻车指定水果抓起保持，不是全部水果入筐。按完整原始实验统计结果，展示数量不作为成功率。

## 制作材料时的阅读顺序

先读传统01建模与算法、02测试结果；再读VLA01接口、02数据训练、03验证结果；最后按视频索引选择片段和原始评分。技术依据、测试指标和展示视频分别归档，避免混淆。未生成PPT或报告成稿。

源码与归档均有SHA256清单。大模型权重/Tokenizer留服务器，原始数据留原桌面数据目录；不复制环境/缓存或账号密钥。不自动上传GitHub。
'''
 write(out/'README_先读.md',readme)
 # Source provenance and a strict secret scan before allowing publication.
 hits=scan(out);assert not hits,('Review potential credentials in files',hits)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 source_files=[p for p in pub.rglob('*') if p.is_file()]
 revision=dict(git_commit=commit,worktree_dirty_at_export=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),files={str(p.relative_to(pub)):sha(p) for p in source_files},scope='Whitelisted current source plus separately identified actual frozen training snapshot; excludes personal diaries, weights, data and credentials')
 dump(pub/'source_revision.json',revision)
 receipt=dict(status='ASSEMBLED_PENDING_EXPORTED_SOURCE_TEST',desktop_path=str(out),base_commit=commit,server_source_download=load(snap/'verification.json'),traditional_frozen_source=frozen,traditional_metrics=metrics,vla_fixed30='14/30',vla_display_success_per_class=10,vla_all_attempts=68,vla_successes=30,vla_failures=38,traditional_display_videos=len(list((tr/'完整任务视频_45段').rglob('*.mp4'))),vla_display_videos=len(list((vl/'十果桌面指令抓取视频_40段').rglob('*.mp4'))),source_files=len(source_files),potential_credentials_found=hits,daily_diaries_included=False,weights_or_raw_dataset_downloaded=False,training_or_inference_started=False,uploaded_to_github=False,free_gib=shutil.disk_usage(ROOT).free/1024**3)
 assert receipt['traditional_display_videos']==45 and receipt['vla_display_videos']==40
 dump(out/'资料包核验.json',receipt);dump(ROOT/'outputs/stage25/assembly_receipt_v1.json',receipt)
 print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
