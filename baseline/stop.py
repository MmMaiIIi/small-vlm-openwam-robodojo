"""Stop only recorded, start-time-verified process groups from this project."""
import argparse
import json
import os
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None

def group_owned(record, run):
    """Also recognize owned children if their group leader has already exited."""
    marker = ('ROBOT_AGENT_EXP_RUN=' + str(run)).encode()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            pid = int(proc.name)
            if os.getpgid(pid) != record['pid']:
                continue
            state = (proc / 'stat').read_text().split(') ', 1)[1].split()[0]
            if state != 'Z' and marker in (proc / 'environ').read_bytes().split(b'\0'):
                return True
        except (OSError, ProcessLookupError):
            continue
    return False

def cleanup(run):
    run = Path(run).resolve()
    if run.parent != ROOT / 'runs':
        raise ValueError('Refusing cleanup outside this project runs directory')
    manifest = run / 'pids.json'
    if not manifest.exists():
        return
    records = json.loads(manifest.read_text())
    targets = [r for r in records if r['pid'] != os.getpid() and
               (identity(r['pid']) == r['start_ticks'] or group_owned(r, run))]
    for r in reversed(targets):
        try:
            if r['role'] == 'qwen' and identity(r['pid']) == r['start_ticks']:
                os.kill(r['pid'], signal.SIGTERM)  # API server gracefully closes its engine.
            else:
                os.killpg(r['pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and any(group_owned(r, run) for r in targets):
        time.sleep(0.2)
    for r in targets:
        if group_owned(r, run):
            try:
                os.killpg(r['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run')
    args = parser.parse_args()
    active = ROOT / '.active_run'
    if args.run or active.exists():
        cleanup(args.run or active.read_text().strip())
