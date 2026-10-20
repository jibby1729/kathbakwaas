"""Replace the gateway supervisor while retaining an existing Quick Tunnel.

Prepare and test on a spare port first. --prepare records exact process identities;
--deploy checks them again, takes ownership, and replaces only the gateway.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.hosting'
MANIFEST = STATE / 'gateway-handoff.json'
STATUS = STATE / 'gateway-supervisor.json'
SOURCES = ['app/sharing.py', 'app/analytics.py', 'scripts/deploy_gateway.py']


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def identity(process):
    return {'pid': process.pid, 'created': process.create_time(),
            'exe': process.exe(), 'cmdline': process.cmdline()}


def verified(info):
    process = psutil.Process(info['pid'])
    if identity(process) != info:
        raise RuntimeError('Process identity changed; refusing to stop it.')
    return process


def prepare():
    owners = {c.pid for c in psutil.net_connections(kind='tcp')
              if c.status == psutil.CONN_LISTEN and c.laddr.ip == '127.0.0.1' and c.laddr.port == 8010}
    if len(owners) != 1:
        raise RuntimeError('Expected one gateway listening on loopback port 8010.')
    gateway = psutil.Process(owners.pop())
    if 'app.sharing:app' not in gateway.cmdline():
        raise RuntimeError('Port 8010 is not the expected gateway.')
    # Windows virtual-environment launchers add a parent wrapper process.
    while gateway.parent() and 'app.sharing:app' in gateway.parent().cmdline():
        gateway = gateway.parent()
    supervisor = gateway.parent()
    if not supervisor or not any(arg.replace('\\', '/').endswith('scripts/share.py') for arg in supervisor.cmdline()):
        raise RuntimeError('Expected the original sharing supervisor. No processes changed.')
    if Path(supervisor.cwd()).resolve() != ROOT:
        raise RuntimeError('Sharing supervisor belongs to a different workspace.')
    tunnels = [p for p in supervisor.children() if Path(p.exe()).resolve() == ROOT / '.tools' / 'cloudflared.exe']
    if len(tunnels) != 1 or 'http://127.0.0.1:8010' not in tunnels[0].cmdline():
        raise RuntimeError('Could not uniquely identify the existing tunnel.')
    url = (STATE / 'url.txt').read_text().strip()
    if not re.fullmatch(r'https://[a-z0-9-]+\.trycloudflare\.com', url):
        raise RuntimeError('Unexpected public URL.')
    descendants = [gateway, *gateway.children(recursive=True)]
    console_host = str(Path(os.environ['SystemRoot']) / 'System32' / 'conhost.exe').lower()
    if any('app.sharing:app' not in p.cmdline() and p.exe().lower() != console_host for p in descendants):
        raise RuntimeError('Gateway has unexpected children; no processes changed.')
    # Windows creates a dedicated hidden console host. It exits with its clients;
    # only target the identified Python gateway processes explicitly.
    tree = [p for p in descendants if 'app.sharing:app' in p.cmdline()]
    data = {'supervisor': identity(supervisor), 'gateway': [identity(p) for p in tree],
            'tunnel': identity(tunnels[0]), 'url': url, 'source_hashes': hashes()}
    MANIFEST.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print('Prepared exact process identities. No processes stopped.')


def stop_tree(process):
    try:
        owned = [process, *process.children(recursive=True)]
        for child in reversed(owned):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(owned, timeout=10)
    except psutil.NoSuchProcess:
        pass


def start_gateway():
    env = dict(os.environ, KOSHUR_ANALYTICS_DB=str(STATE / 'traffic.sqlite3'))
    with (STATE / 'gateway-analytics.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.sharing:app',
            '--host', '127.0.0.1', '--port', '8010', '--no-access-log', '--no-proxy-headers'],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    return child


def deploy():
    if sys.platform != 'win32':
        raise RuntimeError('This handoff is for the tested Windows process layout only.')
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if hashes() != data['source_hashes']:
        raise RuntimeError('Sources changed after preparation; prepare and test again.')
    if (STATE / 'stop').exists() or (STATE / 'url.txt').read_text().strip() != data['url']:
        raise RuntimeError('Sharing was stopped or the URL changed; no processes stopped.')
    supervisor = verified(data['supervisor'])
    old_gateway = [verified(info) for info in data['gateway']]
    tunnel = verified(data['tunnel'])
    status = {'supervisor_pid': os.getpid(), 'tunnel_pid': tunnel.pid, 'url': data['url'], 'state': 'prepared'}

    def save():
        # Status writing must not end ownership of the live tunnel if a Windows
        # file reader briefly prevents replacement/writing.
        try:
            STATUS.write_text(json.dumps(status, indent=2), encoding='utf-8')
        except OSError as exc:
            print(f'Could not update deployment status: {exc}', file=sys.stderr, flush=True)

    save()
    # Terminate just the old Python supervisor, not its tree: its finally block
    # would stop cloudflared. On Windows children survive TerminateProcess.
    # The detached tunnel is now supervised below and keeps the same URL.
    supervisor.kill()
    supervisor.wait(timeout=10)
    verified(data['tunnel'])
    started = time.monotonic()
    for process in reversed(old_gateway):
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(old_gateway, timeout=10)
    child = start_gateway()
    status.update(state='starting', gateway_pid=child.pid)
    save()
    with httpx.Client(timeout=3, trust_env=False) as client:
        for _ in range(60):
            if child.poll() is not None:
                raise RuntimeError('New gateway failed. Tunnel was deliberately left running; see gateway-analytics.log.')
            try:
                response = client.get('http://127.0.0.1:8010/api/health')
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError('Gateway readiness timed out. Tunnel left running for recovery.')
    verified(data['tunnel'])
    assert (STATE / 'url.txt').read_text().strip() == data['url']
    status.update(state='ready', restart_seconds=round(time.monotonic() - started, 3))
    save()
    print(f"Gateway ready in {status['restart_seconds']} seconds; same tunnel PID and URL.", flush=True)
    try:
        while not (STATE / 'stop').exists():
            if not tunnel.is_running():
                raise RuntimeError('The existing tunnel exited; no replacement URL was created.')
            if child.poll() is not None:
                print('Gateway exited; restarting it without touching the tunnel.', flush=True)
                child = start_gateway()
                status.update(gateway_pid=child.pid, state='restarting')
                save()
                time.sleep(5)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    else:
        print('Stop-sharing requested.', flush=True)
    stop_tree(psutil.Process(child.pid))
    if tunnel.is_running():
        tunnel.kill()
        tunnel.wait(timeout=10)
    (STATE / 'url.txt').unlink(missing_ok=True)
    status['state'] = 'stopped'
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else deploy()
