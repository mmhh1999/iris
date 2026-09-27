"""Run a bounded experiment and persist status, including stalled/failed runs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main(a):
    folder=Path(a.record);folder.mkdir(parents=True,exist_ok=False)
    command=a.command
    if command and command[0]=='--':command=command[1:]
    if not command:raise ValueError('Command required')
    def save(state):
        tmp=folder/'status.json.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(folder/'status.json')
    log=folder/'run.log';start=time.monotonic()
    with log.open('wb') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state={'status':'running','pid':child.pid,'command':command,
               'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'max_seconds':a.max_seconds,'idle_seconds':a.idle_seconds}
        save(state)
        while child.poll() is None:
            elapsed=time.monotonic()-start;idle=time.time()-log.stat().st_mtime
            if elapsed>a.max_seconds or idle>a.idle_seconds:
                state['status']='timed_out' if elapsed>a.max_seconds else 'stalled'
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                break
            state.update(elapsed_seconds=elapsed,log_idle_seconds=idle);save(state);time.sleep(2)
        if state['status']=='running':state['status']='succeeded' if child.returncode==0 else 'failed'
        state.update(returncode=child.returncode,elapsed_seconds=time.monotonic()-start,
                     finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(state);print(json.dumps(state,indent=2),flush=True)
        if state['status']!='succeeded':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',required=True)
    p.add_argument('--max-seconds',type=int,default=1800);p.add_argument('--idle-seconds',type=int,default=300)
    p.add_argument('command',nargs=argparse.REMAINDER);main(p.parse_args())
