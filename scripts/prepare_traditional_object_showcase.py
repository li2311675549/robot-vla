#!/usr/bin/env python3
"""Export disclosed per-object showcase clips from existing real task recordings."""
import argparse,bisect,csv,hashlib,json,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
QUOTAS={'door':(10,0),'apple':(10,0),'banana':(9,1),'orange':(10,0),'small_book':(10,0),'medium_book':(10,0),'large_book':(7,3)}
LABELS={'door':'开关门','apple':'苹果分类','banana':'香蕉分类','orange':'橘子分类','small_book':'小书收纳','medium_book':'中书收纳','large_book':'大书收纳'}
def load(p):return json.loads(p.read_text())
def dump(p,r):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_case(episode,group,success,start=None,end=None,detail=None,kind='continuous_full_task_segment'):
 p=Path(episode).resolve();r=load(p/'result.json');video=p/'video/demo.mp4'
 seed=r.get('seed')
 if seed is None and r.get('source'):
  metadata=Path(r['source'])/'scene.json'
  if metadata.is_file():seed=load(metadata).get('seed')
 meta=load(p/'video/video_result.json') if (p/'video/video_result.json').exists() else r.get('video',{})
 assert video.is_file() and meta.get('full_decode')=='PASS',(p,'source decode receipt')
 timeline=load(p/'video/timeline.json');fps=float(meta['fps']);times=[x['sim_seconds'] for x in timeline]
 assert len(times)==int(meta['frames']) and all(b>=a for a,b in zip(times,times[1:]))
 lo=0 if start is None else max(0,bisect.bisect_left(times,start))
 hi=len(times) if end is None else min(len(times),bisect.bisect_right(times,end))
 assert hi>lo and timeline[lo]['frame']==lo and timeline[hi-1]['frame']==hi-1
 return dict(group=group,success=success,episode=str(p),source_video=str(video),source_result=str(p/'result.json'),seed=seed,source_parent_success=r.get('success'),source_frame_start=lo,source_frame_end_exclusive=hi,source_sim_start=times[lo],source_sim_end=times[hi-1],fps=fps,expected_frames=hi-lo,detail=detail,scope=kind,book_profile=r.get('book_profile'),source_scene_protocol=r.get('scene_protocol'))
def pick(pool,n):
 out=[];seeds=set()
 while len(out)<n:
  left=[r for r in pool if r['episode'] not in {x['episode'] for x in out}];assert left
  r=max(left,key=lambda x:x['seed'] not in seeds);out.append(r);seeds.add(r['seed'])
 return out

def make_plan(source):
 inventory=load(source/'交付校验.json');pools={k:[] for k in QUOTAS}
 for task,section in inventory['tasks'].items():
  for row in section['records']:
   if not row['success']:continue
   p=Path(row['episode']);r=load(p/'result.json');assert r['success']
   assert sha(p/'video/demo.mp4')==row['video_sha256'] and sha(p/'result.json')==row['result_sha256']
   if task=='door':pools['door'].append(source_case(p,'door',True,detail='Full opening/entry/closing success'));continue
   fsm=load(p/'fsm.json')
   if task=='fruit':
    assert r['score']['correct']==r['score']['total']==10
    selections=[(i,e) for i,e in enumerate(fsm) if e['reason']=='select_visible_fruit']
    for category in ('apple','banana','orange'):
     for j,(i,e) in enumerate(selections):
      if e['target']['category']!=category:continue
      until=selections[j+1][0] if j+1<len(selections) else len(fsm)
      verifies=[x for x in fsm[i:until] if x['reason']=='verify_physical_bin_placement']
      if not verifies:continue
      v=verifies[0];body=v['target'];witness=[g for g in r['grasp_attempts'] if g.get('body')==body and g.get('placement_verified')]
      actual=next(o for o in r['score']['objects'] if o['body']==body)
      assert witness and actual['correct_bin'] and actual['category']==category
      pools[category].append(source_case(p,category,True,e['time']-1,v['time']+.7,detail={'target_body':body,'placement_verified':True,'attempt_witness':witness,'meaning':'Observe/grasp/lift/transport/release/correct-bin verification; may include bounded retries'}));break
   if task=='drawer':
    assert r['book_profile']=='distinct_v2' and r['score']['success'] and r['score']['returned']
    starts=[x for x in fsm if x['reason'].startswith('cabinet_open_layer_')]
    for rank,group in enumerate(('small_book','medium_book','large_book')):
     start=next(x['time'] for x in starts if x['reason']==f'cabinet_open_layer_{rank}')
     if rank<2:end=next(x['time'] for x in starts if x['reason']==f'cabinet_open_layer_{rank+1}')
     else:end=next(x['time'] for x in fsm if x['state']=='RETURN_HOME')
     op=next(x for x in r['operations'] if x['rank_from_visual_area']==rank)
     book=next(x for x in r['score']['books'] if x['rank']==rank);assert book['inside_correct_layer']
     pools[group].append(source_case(p,group,True,start,end,detail={'book':book,'operation':op,'meaning':'Layer opening, physical book handling, insertion/release and layer closure. Return-home belongs to full parent task.'}))
 selected=[]
 for group,(n,_) in QUOTAS.items():selected+=pick(pools[group],n)
 # True historical failures: explicitly retain different starting scopes and causes.
 p=ROOT/'outputs/stage7/banana_video_seed420/full_failure';r=load(p/'result.json');assert r['transport_success'] is False
 selected.append(source_case(p,'banana',False,detail={'failure':r['failure'],'terminal_bilateral':r['terminal_bilateral'],'meaning':'Already-held physical-grasp initialization; static-hold verification failed, not a fresh tabletop pick or completed sorting attempt'},kind='historical_already_held_static_hold_diagnostic'))
 for rel in ['outputs/stage20/books_all_rank_tilt_dev200009_v25_recorded','outputs/stage11/books_formal_parked_dual_v1/63001']:
  p=ROOT/rel;r=load(p/'result.json');assert r['success'] is False and len(r['operations'])==2
  fsm=load(p/'fsm.json');start=next(x['time'] for x in fsm if x['reason']=='cabinet_open_layer_2')
  assert any(x.get('target')=='book_2' for x in fsm)
  selected.append(source_case(p,'large_book',False,start,None,detail={'failure_stage':r['failure_stage'],'failure_category':r['failure_category'],'error':r.get('error'),'meaning':'Third-book operation failed; do not label as necessarily failing to grasp.'}))
 p=ROOT/'outputs/stage10/book_third_slip_checkpoint_recorded_clockfix';r=load(p/'result.json');assert r['success'] is False
 selected.append(source_case(p,'large_book',False,detail={'failure':r['failure'],'initial_score':r['initial_score'],'meaning':'Checkpoint continuation after earlier physical operations; controller stopped on25.1mm relative-error guard although terminal bilateral contact was still true. Not proof of actual drop.'},kind='historical_checkpoint_continuation_protective_stop'))
 for group,(n,f) in QUOTAS.items():
  a=[x for x in selected if x['group']==group];assert len(a)==10 and sum(x['success'] for x in a)==n and sum(not x['success'] for x in a)==f
 return selected

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve();evidence=a.evidence.resolve();assert not out.exists() and not evidence.exists()
 assert shutil.disk_usage(ROOT).free>25.3*1024**3
 plan=make_plan(a.source);evidence.mkdir(parents=True);out.mkdir();dump(evidence/'plan.json',plan)
 status={'status':'EXPORTING_REAL_SOURCE_CLIPS','planned':70,'completed':0,'new_simulation_or_inference':False};dump(evidence/'status.json',status)
 all_entries=[];started=time.monotonic()
 for gi,group in enumerate(QUOTAS,1):
  folder=out/f'{gi:02d}_{LABELS[group]}';(folder/'视频').mkdir(parents=True);(folder/'评分与来源').mkdir()
  cases=[x for x in plan if x['group']==group]
  for i,row in enumerate(cases,1):
   assert shutil.disk_usage(ROOT).free>25.1*1024**3
   name=f"{i:02d}_{'成功' if row['success'] else '失败'}_seed{row['seed']}";video=folder/'视频'/(name+'.mp4');record=folder/'评分与来源'/(name+'.json')
   src=Path(row['source_video']);source_sha=sha(src);whole=row['source_frame_start']==0 and row['expected_frames']==len(load(src.parent/'timeline.json'))
   if whole:
    shutil.copy2(src,video);assert sha(video)==source_sha
   else:
    command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-ss',f"{row['source_frame_start']/row['fps']:.6f}",'-i',str(src),'-frames:v',str(row['expected_frames']),'-an','-c:v','libx264','-preset','veryfast','-crf','23','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    subprocess.run(command,check=True,timeout=180)
   subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=120)
   probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(video)],text=True))['streams'][0]
   assert int(probe['nb_read_frames'])==row['expected_frames']
   final={**row,'display_video':str(video.relative_to(out)),'display_result':str(record.relative_to(out)),'source_video_sha256':source_sha,'source_result_sha256':sha(Path(row['source_result'])),'video_sha256':sha(video),'clip_kind':'unchanged_whole_recording' if whole else 'single_contiguous_source_interval_reencoded_no_splice','full_decode':'PASS','probe':probe,'raw_parent_result':load(Path(row['source_result']))}
   dump(record,final);all_entries.append({k:v for k,v in final.items() if k!='raw_parent_result'});status.update(completed=len(all_entries),last_group=group,last_file=str(video),wall_seconds=time.monotonic()-started);dump(evidence/'status.json',status);print(group,i,'PASS',flush=True)
  n,f=QUOTAS[group]
  text=f'# {LABELS[group]}：10条真实操作展示记录\n\n构成：{n}成功、{f}失败；这是按指定结果构成整理的展示集合，不是预先固定连续10次独立测试。\n\n'
  text+='每段保留原始episode、帧区间、仿真时间范围、评分和SHA。水果成功段包含从观察到正确入筐验证；书本成功段包含对应抽屉开合与单书放置，5m导航和最终返航请看完整父回合。片段为一个连续区间，无跨回合拼接。\n\n'
  if group=='banana':text+='失败录像来自历史已持物静态保持诊断，初态已夹持，未验证自主从桌面抓取到入筐；不能把它当成一次完整分类失败或与新策略作公平成功率比较。\n'
  if group=='large_book':text+='三例失败分别为桌边几何不可行、放置路径碰撞起点、checkpoint续作中的相对误差保护停止。最后一例末双侧接触仍为真，不能据此声称书本实际掉落；三种失败不能统一写成抓取失败。\n'
  (folder/'说明.md').write_text(text)
 manifest={'status':'PASS70_REAL_RECORDS_SEPARATED_BY_OBJECT','scope':'Outcome-selected cross-version showcase; not70independent experiments or a new frozen evaluation','quotas':{g:{'success':n,'failure':f,'records':10} for g,(n,f) in QUOTAS.items()},'original_formal_results_unchanged':{'door':'10/10','full_tenfruit':'2/10','scattered_three_books':'4/10'},'new_simulation_or_inference':False,'all70_full_decode_pass':True,'contiguous_intervals_without_splicing':True,'entries':all_entries}
 dump(out/'展示清单.json',manifest);dump(evidence/'manifest.json',manifest)
 with (out/'记录索引.csv').open('w') as stream:
  fields=['group','success','seed','scope','episode','source_frame_start','source_frame_end_exclusive','source_sim_start','source_sim_end','display_video','video_sha256'];w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(all_entries)
 lines=['# 传统方法：按任务和物体分组的70条真实记录','','门、苹果、香蕉、橘子、小书、中书、大书分别10条。所有记录来自已执行真实物理仿真；单物体视频从原连续父回合按FSM时间截取，不拼接不同回合。','', '|组别|成功|失败|','|---|---:|---:|']
 for g,(n,f) in QUOTAS.items():lines.append(f'|{LABELS[g]}|{n}|{f}|')
 lines+=['','这是按指定结果构成选取的跨开发版本展示，含重复初态，不代表连续独立10次测试成功率。门为完整任务；水果/书本主要为单项片段，不将它们替代完整5m任务。原正式结果与全部旧失败保持不变。','', '香蕉失败为已持物静态保持诊断；大书含一个checkpoint保护停止回合，初态/范围与从桌面开始的片段不同，详见各组说明与逐段来源。没有将未尝试的物体标成失败，也没有将仍夹住但未完成操作写成掉落。','', '原完整视频保留在既有桌面45段包及工程outputs；全部评分、时间范围和SHA见评分与来源/及展示清单。VLA目录不变。']
 (out/'README_先读.md').write_text('\n'.join(lines)+'\n')
 status.update(status='COMPLETE70',wall_seconds=time.monotonic()-started,desktop_path=str(out),free_gib=shutil.disk_usage(ROOT).free/1024**3);dump(evidence/'status.json',status);dump(out/'交付校验.json',status);print(json.dumps(status,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
