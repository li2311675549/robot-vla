#!/usr/bin/env python3
"""Finish a frozen recorded batch, aggregate every outcome, stop only its server.

No passwords, model updates, automatic retries, or task-score changes occur here.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

from _bootstrap import ROOT


def wilson(successes, attempts):
    if not attempts:
        return None
    z = 1.959963984540054
    p = successes / attempts
    denominator = 1 + z*z/attempts
    center = (p + z*z/(2*attempts)) / denominator
    radius = z * math.sqrt(p*(1-p)/attempts + z*z/(4*attempts**2)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def wait_for_ready(protocol,port,timeout=75,connect=None):
    """Admission is not a trial: never spawn physics before the model is ready."""
    if connect is None:
        sys.path.insert(0,'/home/li/vla/openpi/packages/openpi-client/src')
        from robot_exam.vla.client import connect_openpi_policy
        connect=connect_openpi_policy
    start=time.monotonic();attempts=0;last=None
    while time.monotonic()-start<timeout:
        attempts+=1
        try:policy=connect('127.0.0.1',port,compression='deflate')
        except Exception as exc:
            last=f'{type(exc).__name__}: {exc}';time.sleep(.5);continue
        try:metadata=policy.get_server_metadata()
        finally:policy._ws.close()
        if metadata.get('actual_cameras')!=['front','wrist'] or metadata.get('action_semantics')!='ur5_absolute_joint_goal_v1':
            raise ValueError('Reject wrong camera/action metadata before physical admission')
        expected=protocol.get('server_checkpoint_label')
        if expected is not None and str(metadata.get('checkpoint_step_label'))!=str(expected):
            raise ValueError('Reject wrong checkpoint before physical admission')
        return dict(status='READY_METADATA_CONFIRMED',attempts=attempts,metadata=metadata,
                    model_queries=0,physical_trials=0,wait_seconds=time.monotonic()-start)
    raise TimeoutError(f'Inference admission timeout; no physical trials started. Last error: {last}')


def stop_own_server(args):
    if not args.server_receipt:
        return {'status': 'NOT_CONFIGURED'}
    receipt = json.loads(args.server_receipt.read_text())
    # The unique guard log name and checkpoint identify our inference child;
    # never stop a trainer or a different inference service by process name.
    code = '''import json,os,pathlib,signal,time
receipt=json.loads(INPUT)
guard=int(receipt['guard_pid']);marker=receipt['guard_output']
proc=pathlib.Path('/proc')/str(guard)
if not proc.exists():
 print(json.dumps({'status':'GUARD_ALREADY_EXITED'}))
else:
 cmd=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 assert marker in cmd,'Guard identity mismatch; refuse stopping'
 children=(proc/'task'/str(guard)/'children').read_text().split()
 stopped=[]
 for child in children:
  p=pathlib.Path('/proc')/child
  if not p.exists():continue
  command=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
  if 'serve_openpi_original_ur5' in command and '/9999' in command:
   os.killpg(int(child),signal.SIGTERM);stopped.append(int(child))
 print(json.dumps({'status':'OWN_INFERENCE_CHILD_STOP_REQUESTED','child_pids':stopped,'training_updates':0}))
'''.replace('INPUT', repr(json.dumps(receipt)))
    ssh = ['ssh', '-S', args.ssh_socket, '-o', 'BatchMode=yes',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
           '-o', 'UserKnownHostsFile=' + str(ROOT/'outputs/server_checks/new_instance_20261101_v1/ssh_known_hosts'),
           '-p', '43698', 'root@connect.weste.seetacloud.com',
           shlex.join(['/root/autodl-tmp/robot_exam_training/openpi/.venv/bin/python', '-c', code])]
    result = subprocess.run(ssh, capture_output=True, text=True, timeout=35)
    return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                fallback=f"Server guard has an independent {receipt['timeout_seconds']}s timeout")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--server-receipt', type=Path)
    p.add_argument('--ssh-socket')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--vision-probe',action='store_true',help='After all physical trials, bounded read-only RGB sensitivity check before stopping inference')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output/'run_status.json'
    if status.exists() and not args.resume:
        raise FileExistsError('Preserve previous batch status')
    protocol = json.loads(args.protocol.read_text())
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    status.write_text(json.dumps(dict(status='RUNNING', started_utc=start, training_updates=0), indent=2)+'\n')
    if 'openpi' in protocol['policies']:
        try:
            admission=wait_for_ready(protocol,args.port)
        except Exception as exc:
            admission=dict(status='NOT_STARTED_INFERENCE_NOT_READY',error=dict(type=type(exc).__name__,message=str(exc)),
                           physical_trials=0,training_updates=0)
            (args.output/f'admission_failure_{time.time_ns()}.json').write_text(json.dumps(admission,indent=2)+'\n')
            status.write_text(json.dumps(admission,indent=2)+'\n')
            try:cleanup=stop_own_server(args)
            except Exception as failure:cleanup=dict(status='UNCONFIRMED',error=str(failure))
            (args.output/'inference_cleanup.json').write_text(json.dumps(cleanup,indent=2)+'\n')
            return 1
        (args.output/f'admission_ready_{time.time_ns()}.json').write_text(json.dumps(admission,indent=2)+'\n')
    matrix, analysis = args.output/'matrix', args.output/'analysis'
    cmd = [sys.executable, str(ROOT/'scripts/evaluate_remote_fruit_development.py'),
           '--protocol', str(args.protocol.resolve()), '--fixtures', str(args.fixtures.resolve()),
           '--output', str(matrix.resolve()), '--port', str(args.port)]
    if args.resume and matrix.exists():
        cmd.append('--resume')
    result = subprocess.run(cmd, cwd=ROOT)
    probe_returncode=None
    deadline=protocol.get('extra_probe_deadline_utc')
    probe_has_time=not deadline or datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=150)<datetime.datetime.fromisoformat(deadline)
    if args.vision_probe and not probe_has_time:probe_returncode='SKIPPED_SHUTDOWN_RESERVE'
    if args.vision_probe and result.returncode==0 and probe_has_time:
        try:
            probe_returncode=subprocess.run([sys.executable,str(ROOT/'scripts/probe_fixed_policy_vision.py'),
                '--matrix',str(matrix.resolve()),'--output',str((args.output/'vision_probe').resolve()),
                '--port',str(args.port)],cwd=ROOT,timeout=150).returncode
        except subprocess.TimeoutExpired:
            probe_returncode='WATCHDOG_TIMEOUT'
    cleanup = None
    try:
        cleanup = stop_own_server(args)
    except Exception as exc:
        cleanup = dict(status='STOP_UNCONFIRMED', error=str(exc), fallback='Four-hour server guard')
    (args.output/'inference_cleanup.json').write_text(json.dumps(cleanup, indent=2)+'\n')
    rows = json.loads((matrix/'episodes.json').read_text()) if (matrix/'episodes.json').exists() else []
    integrity = json.loads((matrix/'integrity.json').read_text()) if (matrix/'integrity.json').exists() else None
    analysis_returncode = None
    if rows and integrity is not None and not analysis.exists():
        analysis_returncode = subprocess.run([sys.executable, str(ROOT/'scripts/analyze_remote_fruit_evaluation.py'),
            '--matrix', str(matrix.resolve()), '--output', str(analysis.resolve())], cwd=ROOT).returncode
    grouped = {}
    for skill in protocol['skills']:
        cases = [row for row in rows if row['skill'] == skill]
        successes = sum(bool(row.get('success')) for row in cases)
        grouped[skill] = dict(attempts=len(cases), successes=successes,
            success_rate=successes/len(cases) if cases else None,
            wilson_95=wilson(successes,len(cases)),
            fully_decoded_videos=sum((row.get('video') or {}).get('full_decode')=='PASS' for row in cases))
    expected = len(protocol['scene_seeds'])*len(protocol['skills'])*len(protocol['policies'])*len(protocol.get('evaluation_noise_seeds',[None]))
    complete = result.returncode == 0 and len(rows) == expected and integrity is not None and not integrity['source_hash_mismatches']
    report = dict(status='COMPLETE' if complete else 'INCOMPLETE', started_utc=start,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        expected_cases=expected, completed_cases=len(rows), evaluation_returncode=result.returncode,
        analysis_returncode=analysis_returncode, checkpoint='9999', training_updates=0,
        readonly_vision_probe_returncode=probe_returncode,
        scope=protocol.get('evaluation_population','Stationary single-fruit skills, not original full examination tasks'),
        per_class=grouped, source_integrity=integrity, server_cleanup=cleanup,
        protocol=str(args.protocol), video_index=str(analysis/'VIDEO_INDEX.md'))
    status.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
