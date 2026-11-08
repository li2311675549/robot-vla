#!/usr/bin/env python3
"""Read-only, allowlisted archive of actual training source; excludes weights/data/secrets."""
import argparse,hashlib,json,shlex,shutil,subprocess,tarfile
from pathlib import Path
REMOTE=r'''
from pathlib import Path
import hashlib,io,json,os,sys,tarfile
stage=Path('/root/autodl-tmp/robot_exam_stage22');base=Path('/root/autodl-tmp/robot_exam_training')
roots={'robot_exam':stage/'code','openpi':base/'openpi','stage_tools':stage/'tools','training_tools':base/'tools'}
allow={'.py','.sh','.toml','.lock','.md','.txt','.yaml','.yml','.json','.cfg','.ini','.xml','.proto','.c','.cc','.cpp','.h','.jinja2'}
skip={'.venv','.git','__pycache__','node_modules','datasets','dataset','checkpoints','cache','outputs','runs','results','site-packages'}
files={}
for label,root in roots.items():
 if not root.is_dir():continue
 for parent,dirs,names in os.walk(root):
  dirs[:]=sorted(x for x in dirs if x not in skip and not x.startswith('.'))
  for name in sorted(names):
   p=Path(parent)/name
   if p.is_symlink() or name.startswith('.') or any(x in name.lower() for x in ('credential','password','known_hosts','auth.json','token.json')):continue
   if p.suffix not in allow and not name.startswith(('LICENSE','NOTICE','COPYING')):continue
   if p.stat().st_size>5*1024**2:continue
   files[str(Path(label)/p.relative_to(root))]=p
norm=stage/'checkpoints/pi05_original_ur5_lora_v1/multifruit_stage22_batch4_v2/9999/assets/original_ur5/norm_stats.json'
assert norm.is_file();files['checkpoint9999_assets/original_ur5/norm_stats.json']=norm
size=sum(p.stat().st_size for p in files.values());assert size<80*1024**2
manifest={'scope':'Actual source/config/licences and small norm statistics only; no model/data/environment/credentials or remote writes','source_bytes':size,'openpi_upstream_commit':'Not asserted: source export without Git metadata','files':[dict(path=n,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in sorted(files.items())]}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as a:
 for n,p in sorted(files.items()):a.add(p,arcname=n,recursive=False)
 data=(json.dumps(manifest,indent=2)+'\n').encode();t=tarfile.TarInfo('SOURCE_MANIFEST.json');t.size=len(data);a.addfile(t,io.BytesIO(data))
'''
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('host','port','socket','known-hosts','output'):p.add_argument('--'+key,required=True)
 a=p.parse_args();out=Path(a.output).resolve()
 if out.exists():raise FileExistsError('Preserve previous snapshots')
 if shutil.disk_usage(Path.cwd()).free<25.5*1024**3:raise RuntimeError('Disk reserve25.5GiB')
 out.mkdir(parents=True);archive=out/'actual_server_source.tar.gz'
 remote=shlex.join(['/root/miniconda3/bin/python','-c',REMOTE])
 cmd=['ssh','-p',a.port,'-S',a.socket,'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(Path(a.known_hosts).resolve()),'root@'+a.host,remote]
 with archive.open('xb') as f:subprocess.run(cmd,stdout=f,check=True,timeout=240)
 target=out/'source';target.mkdir()
 with tarfile.open(archive) as t:
  for m in t.getmembers():
   q=target/m.name
   if not m.isfile() or not q.resolve().is_relative_to(target):raise ValueError('Unsafe archive member')
   q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(t.extractfile(m).read());q.chmod(m.mode&0o777)
 manifest=json.loads((target/'SOURCE_MANIFEST.json').read_text())
 for row in manifest['files']:
  q=target/row['path'];assert q.stat().st_size==row['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256']
 receipt=dict(status='PASS_SOURCE_DOWNLOAD_ALL_SHA256',files=len(manifest['files']),source_bytes=manifest['source_bytes'],archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),remote_files_modified=False,training_started=False,weights_downloaded=False)
 (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
