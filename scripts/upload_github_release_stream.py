#!/usr/bin/env python3
"""Upload ZIP_STORED release bundles without creating large local archives.

Uses an explicitly supplied, already authorized gh login; never prints tokens.
"""
import argparse,hashlib,http.client,json,os,subprocess,time,urllib.parse,urllib.request,zipfile
from pathlib import Path
class CountSink:
 def __init__(self):self.count=0
 def write(self,data):self.count+=len(data);return len(data)
 def tell(self):return self.count
 def seek(self,*args):raise OSError('stream is not seekable')
 def flush(self):pass
class NetworkSink(CountSink):
 def __init__(self,connection,total,label):super().__init__();self.connection=connection;self.total=total;self.label=label;self.hash=hashlib.sha256();self.last=time.monotonic()
 def write(self,data):
  self.connection.send(data);self.hash.update(data);n=super().write(data)
  if time.monotonic()-self.last>8:
   print(json.dumps({'upload':self.label,'sent_mib':round(self.count/1024**2,2),'total_mib':round(self.total/1024**2,2)}),flush=True);self.last=time.monotonic()
  return n
def write_zip(sink,files):
 with zipfile.ZipFile(sink,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
  for source,name in files:z.write(source,name)
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--tag',required=True);p.add_argument('--gh',type=Path,required=True);p.add_argument('--gh-config-dir',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--publish-authorized',action='store_true');a=p.parse_args()
 if not a.publish_authorized or a.repo!='li2311675549/robot-vla':raise SystemExit('Explicit authorized publication target required')
 env={**os.environ,'GH_CONFIG_DIR':str(a.gh_config_dir)}
 def gh(*args):return subprocess.check_output([str(a.gh),*args],env=env,text=True).strip()
 assert gh('api','user','--jq','.login')=='li2311675549'
 release=json.loads(gh('api',f'repos/{a.repo}/releases/tags/{a.tag}'));token=gh('auth','token','--hostname','github.com')
 endpoint=urllib.parse.urlsplit(release['upload_url'].split('{')[0]);assert endpoint.scheme=='https' and endpoint.hostname=='uploads.github.com'
 existing={x['name'] for x in release['assets']};receipts=[];plan=json.loads(a.plan.read_text())
 for bundle in plan['bundles']:
  name=bundle['name'];assert name not in existing,'Do not overwrite existing release assets'
  root=Path(bundle['source_root']).resolve();files=[]
  for source in sorted(root.rglob('*')):
   if source.is_file():
    assert not source.is_symlink() and not any(x in ('.git','.venv','__pycache__') for x in source.relative_to(root).parts)
    files.append((source,str(Path(bundle['archive_root'])/source.relative_to(root))))
  assert files
  before=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p,_ in files]
  count=CountSink();write_zip(count,files);assert count.count<2*1024**3
  conn=http.client.HTTPSConnection('uploads.github.com',timeout=120);path=endpoint.path+'?'+urllib.parse.urlencode({'name':name})
  conn.putrequest('POST',path);conn.putheader('Authorization','Bearer '+token);conn.putheader('Accept','application/vnd.github+json');conn.putheader('Content-Type','application/zip');conn.putheader('Content-Length',str(count.count));conn.putheader('User-Agent','robot-vla-materials');conn.endheaders()
  sink=NetworkSink(conn,count.count,name);write_zip(sink,files);assert sink.count==count.count
  response=conn.getresponse();data=response.read();conn.close()
  if response.status!=201:raise RuntimeError(f'Upload HTTP{response.status}: '+data.decode(errors='replace')[:700])
  asset=json.loads(data);assert asset['size']==count.count
  digest=asset.get('digest');assert digest in (None,'sha256:'+sink.hash.hexdigest())
  assert before==[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p,_ in files]
  receipt={'asset':name,'files':len(files),'bytes':count.count,'local_sha256':sink.hash.hexdigest(),'github_digest':digest,'url':asset['browser_download_url'],'asset_id':asset['id']};receipts.append(receipt)
  a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps({'repository':a.repo,'tag':a.tag,'uploaded':receipts,'status':'IN_PROGRESS'},indent=2)+'\n');print(json.dumps(receipt),flush=True)
 a.receipt.write_text(json.dumps({'repository':a.repo,'tag':a.tag,'uploaded':receipts,'status':'PASS_ALL_RELEASE_ASSETS_UPLOADED','credentials_in_receipt':False},indent=2)+'\n')
if __name__=='__main__':main()
