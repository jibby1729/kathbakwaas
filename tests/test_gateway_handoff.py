import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from scripts.deploy_gateway import identity, verified


def test_process_identity_rejects_reused_pid():
    info = identity(psutil.Process())
    assert verified(info).pid == os.getpid()
    info['created'] -= 1
    with pytest.raises(RuntimeError, match='identity changed'):
        verified(info)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows-only handoff')
def test_stopping_supervisor_preserves_its_child(tmp_path):
    # Isolated dummy processes: never touch the live gateway or cloudflared.
    marker = tmp_path / 'child.json'
    code = '''import subprocess,sys,time,json,os
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],creationflags=subprocess.CREATE_NO_WINDOW)
Path(sys.argv[1]).write_text(json.dumps({'supervisor':os.getpid(),'child':child.pid}))
time.sleep(60)
'''
    parent = subprocess.Popen([sys.executable, '-c', code, str(marker)], creationflags=subprocess.CREATE_NO_WINDOW)
    owned = []
    try:
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(0.05)
        data = json.loads(marker.read_text())
        supervisor = psutil.Process(data['supervisor'])
        child = psutil.Process(data['child'])
        owned = [supervisor, child, *child.children(recursive=True)]
        snapshot = identity(child)
        supervisor.kill()
        supervisor.wait(timeout=5)
        assert verified(snapshot).is_running()
    finally:
        for process in reversed(owned):
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(owned, timeout=5)
        parent.wait(timeout=5)
