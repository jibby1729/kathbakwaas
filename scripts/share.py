"""Own the gateway and tunnel processes; Ctrl+C or stop-sharing.cmd closes both."""
from contextlib import closing
import hashlib
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request

import httpx
from keep_awake import KeepAwake

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".hosting"
EXE = ROOT / ".tools" / "cloudflared.exe"
SHA256 = "4d2490e74082c5d48dbfedaf92576fa5bd1702379ab88eb7faa0cfbe8dc891b6"
DOWNLOAD = "https://github.com/cloudflare/cloudflared/releases/download/2026.10.1/cloudflared-windows-amd64.exe"


def install_tunnel():
    if not EXE.exists():
        EXE.parent.mkdir(exist_ok=True)
        temporary = EXE.with_suffix('.download')
        print('Downloading Cloudflare Tunnel...', flush=True)
        try:
            with urllib.request.urlopen(DOWNLOAD, timeout=30) as response:
                data = response.read(100 * 1024 * 1024)
            if hashlib.sha256(data).hexdigest() != SHA256:
                raise RuntimeError('Cloudflare download checksum failed.')
            temporary.write_bytes(data)
            temporary.replace(EXE)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(EXE.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError('Cloudflare executable checksum failed. See scripts/share.py for the pinned version.')


def main():
    install_tunnel()
    with httpx.Client(trust_env=False, timeout=5) as client:
        try:
            health = client.get("http://127.0.0.1:8000/api/health").json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("Start start.cmd first and wait for Ready.") from exc
        if health.get("state") != "ready":
            raise RuntimeError("Wait until the local translator says Ready, then run share.cmd again.")
        # Refuse a second sharing instance or an unrelated service on our port.
        with closing(socket.socket()) as probe:
            probe.bind(("127.0.0.1", 8010))
        STATE.mkdir(exist_ok=True)
        stop = STATE / "stop"
        stop.unlink(missing_ok=True)
        link = STATE / "url.txt"
        link.unlink(missing_ok=True)
        # Explicit empty config avoids accidentally using a personal named-tunnel config.
        config = STATE / "tunnel.yml"
        config.write_text("{}\n", encoding="utf-8")
        processes = []
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            with KeepAwake() as awake, \
                 (STATE / "gateway.log").open("w", encoding="utf-8") as gateway_log, \
                 (STATE / "tunnel.log").open("w", encoding="utf-8") as tunnel_log:
                gateway = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "app.sharing:app", "--host", "127.0.0.1",
                     "--port", "8010", "--no-access-log", "--no-proxy-headers"],
                    cwd=ROOT, stdout=gateway_log, stderr=subprocess.STDOUT, creationflags=flags)
                processes.append(gateway)
                for _ in range(50):
                    if gateway.poll() is not None:
                        raise RuntimeError("Gateway failed. See .hosting/gateway.log.")
                    try:
                        response = client.get("http://127.0.0.1:8010/api/health")
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError("Gateway did not start.")
                tunnel = subprocess.Popen(
                    [str(EXE), "tunnel", "--config", str(config), "--no-autoupdate",
                     "--url", "http://127.0.0.1:8010", "--metrics", "127.0.0.1:0"],
                    cwd=ROOT, stdout=tunnel_log, stderr=subprocess.STDOUT, creationflags=flags)
                processes.append(tunnel)
                print("Starting sharing. Anyone with the link can use the translator.", flush=True)
                print("Keep this window open. Ctrl+C or stop-sharing.cmd stops sharing.", flush=True)
                announced = False
                started = time.monotonic()
                while not stop.exists():
                    awake.refresh()
                    if any(process.poll() is not None for process in processes):
                        raise RuntimeError("Sharing stopped unexpectedly. See .hosting/*.log.")
                    if not announced:
                        logs = (STATE / "tunnel.log").read_text(encoding="utf-8", errors="replace")
                        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", logs)
                        if match:
                            link.write_text(match[0] + "\n", encoding="utf-8")
                            print(f"\nShare this link: {match[0]}\n", flush=True)
                            announced = True
                        elif time.monotonic() - started > 90:
                            raise RuntimeError("Cloudflare did not provide a link. See .hosting/tunnel.log.")
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            link.unlink(missing_ok=True)
            print("Sharing stopped. The local translator is still running.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Could not share: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
