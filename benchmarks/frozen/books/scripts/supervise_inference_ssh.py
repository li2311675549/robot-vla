#!/usr/bin/python3
"""Bounded SSH reconnects; password only in RAM, never argv/env/disk/logs.

Run under a temporary user service. pexpect is the existing Ubuntu package;
the scientific Python and server environments are left unchanged.
"""
import argparse
import datetime
import getpass
import json
from pathlib import Path
import signal
import time

import pexpect


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--socket',required=True)
    p.add_argument('--local-port',type=int,required=True)
    p.add_argument('--known-hosts',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=5400)
    a=p.parse_args()
    if not 60<=a.seconds<=6000:raise ValueError('Keep within the user shutdown budget')
    if a.output.exists():raise FileExistsError('Preserve earlier tunnel attempts')
    a.output.mkdir(parents=True)
    password=getpass.getpass('AutoDL SSH password (RAM only): ')
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    started=time.monotonic();events=[];child=None
    def save(status,**extra):
        events.append(dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=status,**extra))
        (a.output/'status.json').write_text(json.dumps(dict(events=events,latest_status=status,
            supervisor_pid=__import__('os').getpid(),password_persisted=False,training_updates=0),indent=2)+'\n')
    try:
        for attempt in range(1,11):
            if time.monotonic()-started>=a.seconds:break
            cmd=['-N','-M','-S',a.socket,'-E',str(a.output/f'ssh_{attempt:02d}.log'),
                 '-o','ConnectTimeout=15','-o','ConnectionAttempts=1',
                 '-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3',
                 '-o','UserKnownHostsFile='+str(a.known_hosts),'-o','StrictHostKeyChecking=yes',
                 '-o','PubkeyAuthentication=no','-L',f'127.0.0.1:{a.local_port}:127.0.0.1:8000',
                 '-p','43698','root@connect.weste.seetacloud.com']
            save('CONNECTING',attempt=attempt)
            child=pexpect.spawn('/usr/bin/ssh',cmd,encoding='utf-8',echo=False,timeout=20)
            # No pexpect logfile: even authentication output cannot persist a secret.
            outcome=child.expect(['[Pp]assword:',pexpect.EOF,pexpect.TIMEOUT])
            if outcome==0:
                child.sendline(password)
                for _ in range(100):
                    if not child.isalive() or Path(a.socket).exists():break
                    time.sleep(.1)
            if child.isalive() and Path(a.socket).exists():
                save('CONNECTED',attempt=attempt,ssh_pid=child.pid)
                while child.isalive() and time.monotonic()-started<a.seconds:
                    if child.expect([pexpect.EOF,pexpect.TIMEOUT],timeout=1)==0:break
            child.close(force=True)
            save('DISCONNECTED',attempt=attempt,exitstatus=child.exitstatus,signalstatus=child.signalstatus)
            time.sleep(3)
        save('BOUNDED_SUPERVISOR_FINISHED')
    finally:
        if child is not None and child.isalive():child.close(force=True)
        password=None


if __name__=='__main__':main()
