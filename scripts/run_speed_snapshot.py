#!/usr/bin/env python3
"""Freeze per-run Python/config sources before launching a physical speed trial."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a,args=p.parse_known_args();output=a.output.resolve()
    if '--task' in args and args[args.index('--task')+1]=='drawer' and '--book-profile' not in args:
        p.error('Drawer speed comparisons require an explicit --book-profile; never silently use booklet')
    for option in ('--config','--data-output'):
        if option in args:
            index=args.index(option)+1;args[index]=str(Path(args[index]).resolve())
    if output.exists():raise FileExistsError('Keep all prior attempts')
    if shutil.disk_usage(output.parent).free<25*1024**3:raise RuntimeError('Keep25GiB safety reserve')
    code=output/'frozen_code';code.mkdir(parents=True)
    for name in ('src','scripts','configs'):
        shutil.copytree(ROOT/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Immutable robot/model resources stay byte-identical and are individually
    # hashed. Nothing is copied from data/video/history or external services.
    for name in ('assets','third_party','scenes'):(code/name).symlink_to(ROOT/name,target_is_directory=True)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    sources={str(path.relative_to(code)):digest(path) for name in ('src','scripts','configs')
             for path in (code/name).rglob('*') if path.is_file()}
    models={str(path.relative_to(ROOT)):digest(path) for name in ('assets','third_party','scenes')
            for path in (ROOT/name).rglob('*') if path.is_file() and '.git' not in path.parts}
    manifest={'source_git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'sources_sha256':sources,'immutable_resources_sha256':models,
              'invocation_args':args,
              'scope':'Actual code/config frozen at launch; shared models immutable; no physics/time/scoring changes'}
    (output/'execution_code.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env=os.environ.copy();env['PYTHONPATH']=str(code/'src')
    completed=subprocess.run([sys.executable,str(code/'scripts/run_task.py'),'--output',str(output),*args],env=env,cwd=output)
    changed=[name for name,sha in models.items()if digest(ROOT/name)!=sha]
    (output/'immutable_resources_check.json').write_text(json.dumps({'status':'FAIL'if changed else'PASS','changed':changed},indent=2)+'\n')
    raise SystemExit(completed.returncode if not changed else 2)


if __name__=='__main__':main()
