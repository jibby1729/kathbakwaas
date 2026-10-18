"""Run one GPU model at a time and restore the local web service afterwards."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / 'artifacts'
status = {'started': time.time(), 'runs': [], 'completed': False}
def save():
    (ART / 'expanded-status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')

try:
    for name, options in [
        ('stock', ['--backend', 'stock']),
        ('native', ['--backend', 'native']),
        ('typing', ['--backend', 'native', '--typing']),
    ]:
        status['active'] = name
        save()
        with (ART / f'expanded-{name}.log').open('w', encoding='utf-8') as log:
            code = subprocess.call([
                sys.executable, str(ROOT / 'scripts/compare_accuracy.py'),
                *options, '--corpus', 'artifacts/expanded-corpus.json',
                '--reference', 'artifacts/expanded-stock.json',
                '--report', f'expanded-{name}.json',
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status['runs'].append({'name': name, 'exit_code': code, 'finished': time.time()})
        save()
        if code:
            raise RuntimeError(f'{name} comparison failed: {code}')
    status['completed'] = True
except Exception as exc:
    status['error'] = repr(exc)
finally:
    # The previously running server is stopped by the operator before this script.
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
               KOSHUR_OPTIMIZE_EXECUTION='1', KOSHUR_NATIVE_DECODE='1')
    with (ART / 'accuracy-server.stdout.log').open('w') as out, (ART / 'accuracy-server.stderr.log').open('w') as err:
        server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app',
                                   '--host', '127.0.0.1', '--port', '8000', '--no-access-log'],
                                  cwd=ROOT, env=env, stdout=out, stderr=err,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
    (ART / 'accuracy-server.json').write_text(json.dumps({'launcher_pid': server.pid,
        'started_epoch': time.time(), 'port': 8000}, indent=2), encoding='utf-8')
    status.update(active=None, server_pid=server.pid, finished=time.time())
    save()
