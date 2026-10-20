"""Prevent idle sleep on AC power while sharing, without changing power settings."""
import ctypes
from ctypes import wintypes
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class PowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", wintypes.BYTE), ("BatteryFlag", wintypes.BYTE),
        ("BatteryLifePercent", wintypes.BYTE), ("SystemStatusFlag", wintypes.BYTE),
        ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD),
    ]


class KeepAwake:
    """Use and release on the same thread; leave display and lock timers alone."""
    def __init__(self):
        self.active = False
        self.verified = False
        self.kernel = None
        if sys.platform == "win32":
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
            self.kernel.SetThreadExecutionState.restype = wintypes.DWORD
            self.kernel.GetSystemPowerStatus.argtypes = [ctypes.POINTER(PowerStatus)]
            self.kernel.GetSystemPowerStatus.restype = wintypes.BOOL

    def refresh(self):
        if self.kernel is None:
            return
        status = PowerStatus()
        if not self.kernel.GetSystemPowerStatus(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        plugged_in = status.ACLineStatus == 1
        changed = plugged_in != self.active
        if plugged_in or self.active:
            flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if plugged_in else 0)
            if not self.kernel.SetThreadExecutionState(flags):
                raise RuntimeError("Windows could not update the keep-awake request.")
            self.active = plugged_in
            # Reasserting returns the previous thread execution flags, confirming
            # the SYSTEM_REQUIRED request was set, not just our Python boolean.
            previous = self.kernel.SetThreadExecutionState(flags)
            if not previous:
                raise RuntimeError("Windows could not verify the keep-awake request.")
            self.verified = bool(previous & ES_SYSTEM_REQUIRED) if plugged_in else False
        if changed:
            print("Idle sleep prevented while plugged in; the screen may turn off."
                  if self.active else "On battery: normal sleep settings apply. Plug in to keep hosting.",
                  flush=True)

    def __enter__(self):
        self.refresh()
        return self

    def __exit__(self, *_):
        if self.active:
            self.kernel.SetThreadExecutionState(ES_CONTINUOUS)
            self.active = False


def main():
    # Attach to an already-running sharing session without changing its public URL.
    import httpx
    parser = argparse.ArgumentParser(description=__doc__)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--hours', type=float, help='Duration, in hours (default: 12).')
    duration.add_argument('--until', help='Exact ISO 8601 deadline including timezone offset.')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    if args.until:
        try:
            expires = datetime.fromisoformat(args.until)
            if expires.utcoffset() is None:
                raise ValueError('A timezone offset is required.')
            expires = expires.astimezone(timezone.utc)
        except ValueError as exc:
            parser.error(str(exc))
        hours = (expires - now).total_seconds() / 3600
    else:
        hours = args.hours if args.hours is not None else 12
        expires = now + timedelta(hours=hours)
    if not 0 < hours <= 168:
        parser.error('--hours must be between 0 and 168.')
    state = Path(__file__).resolve().parents[1] / ".hosting"
    state.mkdir(exist_ok=True)
    deadline = time.monotonic() + hours * 3600

    def save_status(awake, running):
        status = {'pid': os.getpid(), 'running': running,
                  'idle_sleep_prevented': awake.active,
                  'windows_request_verified': awake.verified if awake.active else False,
                  'checked_at': datetime.now(timezone.utc).isoformat(),
                  'expires_at': expires.isoformat()}
        temporary = state / f'keep-awake-status-{os.getpid()}.tmp'
        try:
            temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
            temporary.replace(state / 'keep-awake-status.json')
        except OSError as exc:
            # A Windows reader may briefly hold the destination without delete
            # sharing. Status reporting must never terminate sleep prevention.
            print(f'Could not update status file; will retry: {exc}', file=sys.stderr, flush=True)

    with KeepAwake() as awake, httpx.Client(timeout=3, trust_env=False) as client:
        print(f"Keeping sharing awake for up to {hours:g} hours, until {expires.isoformat()}. Ctrl+C stops this helper.", flush=True)
        while time.monotonic() < deadline and (state / "url.txt").exists() and not (state / "stop").exists():
            try:
                response = client.get("http://127.0.0.1:8010/api/health")
                response.raise_for_status()
            except httpx.HTTPError:
                # Keep the laptop awake through a backend restart or temporary
                # overload so an outage does not also disable sleep prevention.
                pass
            awake.refresh()
            save_status(awake, True)
            time.sleep(min(5, max(0, deadline - time.monotonic())))
    save_status(awake, False)
    print("Keep-awake helper stopped; normal power settings apply.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
