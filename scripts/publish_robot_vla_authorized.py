#!/usr/bin/env python3
"""Publish the reviewed robot-vla export after explicit GitHub device authorization."""
import argparse,json,os,subprocess,time
from pathlib import Path
REPO='li2311675549/robot-vla'
def main():
 p=argparse.ArgumentParser();p.add_argument('--gh',type=Path,required=True);p.add_argument('--gh-config-dir',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bundle-plan',type=Path,required=True);p.add_argument('--publish-authorized',action='store_true');a=p.parse_args()
 if not a.publish_authorized:raise SystemExit('Explicit publication authorization required')
 source=a.source.resolve();a.output.mkdir(parents=True,exist_ok=True);status_path=a.output/'status.json'
 env={**os.environ,'GH_CONFIG_DIR':str(a.gh_config_dir),'GIT_SSH_COMMAND':'ssh -o BatchMode=yes -o ConnectTimeout=20'}
 state={'repository':REPO,'visibility':'public','started_epoch':time.time(),'state':'AUTH_ACCOUNT_CHECK','source':str(source),'weights_or_raw_dataset_uploaded':False}
 def save(**values):
  state.update(values);t=status_path.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');os.replace(t,status_path);print(json.dumps(values),flush=True)
 def gh(*args,check=True):return subprocess.run([str(a.gh),*args],env=env,text=True,capture_output=True,check=check,timeout=180)
 def git(*args):return subprocess.check_output(['git',*args],cwd=source,env=env,text=True).strip()
 try:
  assert gh('api','user','--jq','.login').stdout.strip()=='li2311675549'
  assert git('remote','get-url','origin')=='git@github.com:'+REPO+'.git'
  assert not git('status','--porcelain')
  probe=gh('api','repos/'+REPO,check=False)
  if probe.returncode:
   if '404' not in probe.stderr and 'Not Found' not in probe.stdout:raise RuntimeError('Repository lookup failed: '+probe.stderr[:500])
   save(state='CREATING_PUBLIC_REPOSITORY')
   gh('repo','create',REPO,'--public','--description','MuJoCo mobile manipulation with Husky, original UR5, physical contact control and dual-view pi0.5')
  metadata=json.loads(gh('api','repos/'+REPO).stdout);assert metadata['owner']['login']=='li2311675549' and metadata['private'] is False
  save(state='PUBLIC_REPOSITORY_CREATED',repository_url=metadata['html_url'])
  local=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/main').split()
  if remote and remote[0]!=local:
   known=subprocess.run(['git','merge-base','--is-ancestor',remote[0],local],cwd=source,env=env,capture_output=True)
   if known.returncode:raise RuntimeError('Existing remote main is not our publication ancestor; refusing overwrite')
  save(state='PUSHING_VERIFIED_SOURCE',code_commit=local)
  subprocess.run(['git','push','--porcelain','-u','origin','main'],cwd=source,env=env,check=True,timeout=1200)
  assert git('ls-remote','origin','refs/heads/main').split()[0]==local
  save(state='SOURCE_PUSH_VERIFIED')
  release=gh('api',f'repos/{REPO}/releases/tags/v1.0.0',check=False)
  if release.returncode:
   if '404' not in release.stderr and 'Not Found' not in release.stdout:raise RuntimeError('Release lookup failed')
   notes=a.output/'release_notes.md';notes.write_text('''# Robot VLA: technical report, code and authentic videos

This release accompanies the reviewed MuJoCo mobile manipulation source. Traditional task protocols and learned short-skill validation are reported separately.

- traditional-cases70.zip: seven groups of10 illustrative per-object/task records,66 successes and4 failures, derived from34 parent episodes. Contains selected/repeated initial conditions and explicitly marked partial diagnostics; not70independent complete tasks.
- vla-cases30.zip: instructed stationary fruit grasp showcase,25 successes and5 failures selected from68 executions, not a new consecutive test. Original fixed ten-fruit-table validation remains14/30.
- technical-report-and-evidence.zip: technical summary, problems/solutions, figures, original results and source evidence.

Frozen traditional full-task results remain door10/10, historical ten-fruit2/10 and scattered distinct-books4/10. VLA videos do not show bin placement or a complete mobile ten-fruit task. All failures and scope distinctions are preserved.

No credentials, pretrained weights or complete raw training dataset are distributed. Original third-party licences apply.
''')
   gh('release','create','v1.0.0','--repo',REPO,'--target','main','--title','Robot VLA — code, videos and technical report','--notes-file',str(notes))
  save(state='UPLOADING_RELEASE_MATERIALS')
  root=Path(__file__).resolve().parents[1]
  subprocess.run([os.sys.executable,str(root/'scripts/upload_github_release_stream.py'),'--repo',REPO,'--tag','v1.0.0','--gh',str(a.gh),'--gh-config-dir',str(a.gh_config_dir),'--plan',str(a.bundle_plan),'--receipt',str(a.output/'release_assets.json'),'--publish-authorized'],env=env,check=True,timeout=3600)
  assets=json.loads((a.output/'release_assets.json').read_text());assert assets['status']=='PASS_ALL_RELEASE_ASSETS_UPLOADED'
  release=json.loads(gh('api',f'repos/{REPO}/releases/tags/v1.0.0').stdout)
  by_name={x['name']:x for x in release['assets']}
  for asset in assets['uploaded']:assert by_name[asset['asset']]['size']==asset['bytes']
  readme=source/'README.md';s=readme.read_text();old='更多案例集拟作为`v1.0.0`发布附件，发布状态见[Releases](https://github.com/li2311675549/robot-vla/releases)。'
  download='完整材料下载：\n\n'+ '\n'.join(f"- [{x['asset']}]({x['url']})" for x in assets['uploaded'])
  assert old in s;s=s.replace(old,download);readme.write_text(s)
  evidence=source/'evidence/release_assets.json';evidence.write_text(json.dumps(assets,indent=2)+'\n')
  subprocess.run(['git','add','README.md','evidence/release_assets.json'],cwd=source,check=True)
  subprocess.run(['git','commit','--quiet','-m','Link verified release materials and preserve upload digests'],cwd=source,check=True)
  subprocess.run(['git','push','--porcelain','origin','main'],cwd=source,env=env,check=True,timeout=600)
  final=git('rev-parse','HEAD');assert git('ls-remote','origin','refs/heads/main').split()[0]==final
  save(state='COMPLETE_PUBLIC_REPOSITORY_AND_RELEASE',final_commit=final,release_url=release['html_url'],assets=assets['uploaded'],finished_epoch=time.time())
 except Exception as error:
  save(state='STOPPED_WITH_ERROR',error_type=type(error).__name__,error=str(error)[:1000]);raise
 finally:
  result=gh('auth','logout','--hostname','github.com','--user','li2311675549',check=False)
  save(temporary_cli_login_cleanup_attempted=True,temporary_cli_logout_returncode=result.returncode)
if __name__=='__main__':main()
