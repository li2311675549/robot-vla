#!/usr/bin/env python3
"""Download an allowlisted, checksummed source snapshot over an existing SSH socket.

No credentials, weights, data, environments, GPU work or remote file writes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

REMOTE_ROOT = '/root/autodl-tmp/robot_exam_training'
REMOTE_PROGRAM = r'''
from pathlib import Path
import hashlib,io,json,os,sys,tarfile
r=Path('/root/autodl-tmp/robot_exam_training')
extensions={'.py','.sh','.toml','.lock','.md','.txt','.yaml','.yml','.json','.cfg','.ini','.ipynb','.proto','.c','.cc','.cpp','.h','.jinja2','.xml'}
skip={'.venv','.git','__pycache__','node_modules','datasets','checkpoints','cache','outputs','runs','site-packages','python','checksum_libs','uv_bootstrap'}
roots=[r/'openpi',r/'tools',r/'robot_exam/src',r/'robot_exam/scripts',r/'robot_exam/configs']
roots += sorted((r/'runtime').glob('fruit_eval_v*'))
files={};excluded=[]
for root in roots:
 if not root.is_dir():continue
 for parent,dirs,names in os.walk(root):
  dirs[:]=sorted(d for d in dirs if d not in skip and not d.startswith('.'))
  for name in sorted(names):
   p=Path(parent)/name
   if p.is_symlink() or name.startswith('.') or not(p.suffix in extensions or name.startswith(('LICENSE','NOTICE','COPYING'))):continue
   if p.stat().st_size>4*1024**2:excluded.append(str(p.relative_to(r)));continue
   files[str(p.relative_to(r))]=p
total=sum(p.stat().st_size for p in files.values())
assert total<80*1024**2,('Unexpected source size',total)
manifest={'remote_root':str(r),'files':[{'path':n,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in sorted(files.items())],
 'source_bytes':total,'excluded_oversize_files':excluded,'git_metadata':'Remote OpenPI is a source export with no .git; no upstream commit invented.',
 'scope':'Source/configuration/licences only; no weights, datasets, logs, credentials, Python installs or remote writes.'}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for n,p in sorted(files.items()):archive.add(p,arcname=n,recursive=False)
 payload=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('SOURCE_MANIFEST.json');info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-socket', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--known-hosts', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Do not overwrite an earlier recovery snapshot')
    if shutil.disk_usage(Path.cwd()).free < 25.5 * 1024**3:
        raise RuntimeError('Need25.5GiB before a bounded source-only recovery')
    args.output.mkdir(parents=True)
    archive = args.output / 'server_sources.tar.gz'
    remote = shlex.join([REMOTE_ROOT+'/openpi/.venv/bin/python', '-c', REMOTE_PROGRAM])
    command = ['ssh', '-S', args.ssh_socket, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile='+str(args.known_hosts.resolve()),
               '-p', '43698', 'root@connect.weste.seetacloud.com', remote]
    with archive.open('xb') as output:
        subprocess.run(command, stdout=output, check=True, timeout=180)
    target = args.output / 'source'
    target.mkdir()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or not (target/member.name).resolve().is_relative_to(target.resolve()):
                raise ValueError('Unsafe source archive entry: '+member.name)
            destination = target/member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(bundle.extractfile(member).read())
    manifest = json.loads((target/'SOURCE_MANIFEST.json').read_text())
    for entry in manifest['files']:
        path=target/entry['path']
        assert path.stat().st_size==entry['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'],entry['path']
    summary = dict(status='PASS_SOURCE_DOWNLOAD_AND_ALL_HASHES',files=len(manifest['files']),
                   source_bytes=manifest['source_bytes'],archive_bytes=archive.stat().st_size,
                   archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   remote_source_modified=False,training_started=False,weights_downloaded=False)
    (args.output/'download_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
