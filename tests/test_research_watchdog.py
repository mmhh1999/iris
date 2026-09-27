import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

WATCH=Path(__file__).resolve().parents[1]/'experiments/watch_research_run.py'


class ResearchWatchdogTests(unittest.TestCase):
    def test_records_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            p=subprocess.run([sys.executable,str(WATCH),'--record',str(out),'--',sys.executable,'-c','print("finished")'],capture_output=True,timeout=15)
            self.assertEqual(p.returncode,0)
            self.assertEqual(json.loads((out/'status.json').read_text())['status'],'succeeded')
            self.assertIn('finished',(out/'run.log').read_text())

    def test_stops_timed_out_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            p=subprocess.run([sys.executable,str(WATCH),'--record',str(out),'--max-seconds','1','--',sys.executable,'-c','import time; time.sleep(30)'],capture_output=True,timeout=15)
            state=json.loads((out/'status.json').read_text())
            self.assertNotEqual(p.returncode,0)
            self.assertEqual(state['status'],'timed_out')
            self.assertNotEqual(state['returncode'],0)

    def test_stops_child_when_disk_low(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            p=subprocess.run([sys.executable,str(WATCH),'--record',str(out),'--min-free-gb','1e9','--',sys.executable,'-c','import time; time.sleep(30)'],capture_output=True,timeout=15)
            state=json.loads((out/'status.json').read_text())
            self.assertNotEqual(p.returncode,0)
            self.assertEqual(state['status'],'disk_low')


if __name__=='__main__':unittest.main()
