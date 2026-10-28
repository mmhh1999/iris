"""Start the local service and optionally open a standalone Windows app window."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
from gs_relight_baseline import selected_baseline

ROOT=Path(__file__).resolve().parents[1]
URL='http://localhost:8771/gs_relight_stages/app/'


def available(url=URL,manifest='scene.json'):
    try:
        with urllib.request.urlopen(url+manifest,timeout=2) as response:
            data=json.load(response)
            return data.get('sourceCount',data.get('gaussians'))==6100978
    except (urllib.error.URLError,ValueError,TimeoutError):return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-window',action='store_true');parser.add_argument('--world',action='store_true',help='Open the opt-in world-space direct-light preview');args=parser.parse_args()
    baseline=selected_baseline()
    target_url=baseline['viewer'] if baseline else ('http://localhost:8771/gs_relight_stages/app_world/' if args.world else URL)
    if baseline and not available(target_url,'viewer_manifest.json'):
        log_path=ROOT/'experiments/out/gs_relight_stages/legacy_viewer.log'
        log_path.parent.mkdir(parents=True,exist_ok=True)
        with log_path.open('ab')as log:
            subprocess.Popen([sys.executable,'-m','http.server','8769','--bind','127.0.0.1','--directory',str(ROOT/'experiments/out')],cwd=ROOT,stdout=log,stderr=log,start_new_session=True)
        for _ in range(50):
            if available(target_url,'viewer_manifest.json'):break
            time.sleep(.2)
        else:raise RuntimeError('Legacy viewer did not start; see '+str(log_path))
    service_url='http://localhost:8771/nucleus4d_transport_v3/' if baseline else URL
    service_manifest='viewer_manifest.json' if baseline else 'scene.json'
    if not available(service_url,service_manifest):
        log_path=ROOT/'experiments/out/gs_relight_stages/server.log'
        log_path.parent.mkdir(parents=True,exist_ok=True)
        with log_path.open('ab') as log:
            subprocess.Popen([sys.executable,str(ROOT/'experiments/gs_relight_server.py')],cwd=ROOT,stdout=log,stderr=log,start_new_session=True)
        for _ in range(50):
            if available(service_url,service_manifest):break
            time.sleep(.2)
        else:raise RuntimeError('Local viewer did not start; see '+str(log_path))
    print(target_url,flush=True)
    if not args.no_window:
        chrome=Path('/mnt/c/Program Files/Google/Chrome/Application/chrome.exe')
        if not chrome.exists():raise RuntimeError('Windows Chrome not found; open the printed URL manually')
        profile=ROOT/'experiments/out/gs_relight_stages/browser_profile_app_windows';profile.mkdir(exist_ok=True)
        windows_path=subprocess.check_output(['wslpath','-w',str(profile)],text=True).strip()
        subprocess.Popen([str(chrome),'--app='+target_url,'--user-data-dir='+windows_path,'--no-first-run','--no-default-browser-check','--disable-background-networking'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)


if __name__=='__main__':main()
