"""Launch RPent/local Qwen and isolated official robot runtimes, one episode."""
import argparse
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import threading
import time
import traceback
from multiprocessing import Pipe
from pathlib import Path

import yaml
from common import append, event, read_jsonl, timestamp
from stop import cleanup, identity

ROOT = Path(__file__).resolve().parents[1]


class RobotConnection:
    def __init__(self, conn, timeout):
        self.conn, self.timeout = conn, timeout

    def call(self, name, arguments):
        self.conn.send({'tool': name, 'arguments': arguments})
        if not self.conn.poll(self.timeout):
            raise TimeoutError('Robot tool IPC timeout')
        return self.conn.recv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['fill_pen_holder'], default='fill_pen_holder')
    parser.add_argument('--seed', type=int, choices=[0], default=0)
    parser.add_argument('--config', default=str(ROOT / 'configs/rpent_qwen2b_openwam.yaml'))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg['environment'].update(task=args.task, seed=args.seed)
    run = ROOT / 'runs' / (time.strftime('%Y%m%dT%H%M%S') + '_rpent_' + args.task + '_0')
    run.mkdir(parents=True); (ROOT / '.active_run').write_text(str(run))
    (run / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    for name in ('planner.jsonl', 'actor.jsonl', 'tools.jsonl', 'gpu.jsonl', 'events.jsonl', 'stdout.log'):
        (run / name).touch()
    print('RUN_DIRECTORY=' + str(run), flush=True)
    # Redirect this process too; the shell retains the run directory for monitoring.
    log = (run / 'stdout.log').open('ab', buffering=0)
    os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
    started = time.monotonic(); episode_start = None
    assert os.getpgrp() == os.getpid(), 'Launch through run_baseline.sh (dedicated process group)'
    records, children = [{'role': 'planner', 'pid': os.getpid(),
                          'start_ticks': identity(os.getpid()), 'argv': os.sys.argv}], {}
    (run / 'pids.json').write_text(json.dumps(records, indent=2))
    halted = threading.Event()
    summary = {'task': args.task, 'seed': args.seed, 'completed': False, 'success': False}
    conn = worker_conn = None
    def monitor():
        while not halted.is_set():
            try:
                value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True).strip().split(',')
                append(run / 'gpu.jsonl', dict(timestamp=timestamp(), unix_time=time.time(),
                    memory_used_mib=int(value[0]), utilization_percent=int(value[1])))
            except Exception as exc:
                event(run, 'gpu_sampling_error', error=repr(exc))
            halted.wait(cfg['logging']['gpu_sample_seconds'])
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    def spawn(role, argv, runtime=False, pass_fds=()):
        child_env = os.environ.copy()
        for name in ('PYTHONPATH', 'LD_LIBRARY_PATH'):
            child_env.pop(name, None)
        child_env.update(PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            ROBOT_AGENT_EXP_RUN=str(run),
            PYTHONPATH=cfg['paths']['robodojo'] + ':' + cfg['paths']['robodojo'] + '/XPolicyLab')
        if runtime:
            argv = ['bash', '-c', 'source /root/gpufree-data/deployment-logs/robodojo-runtime.env; exec "$@"', 'rpent'] + argv
        process = subprocess.Popen(argv, cwd=cfg['paths']['robodojo'] if runtime else ROOT,
            env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=pass_fds)
        children[role] = process
        records.append({'role': role, 'pid': process.pid, 'start_ticks': identity(process.pid), 'argv': argv})
        (run / 'pids.json').write_text(json.dumps(records, indent=2))
        event(run, 'process_started', role=role, pid=process.pid)
        return process
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGINT, interrupt); signal.signal(signal.SIGTERM, interrupt)
    try:
        from pydantic_ai import BinaryContent
        from rpent.dashboard.events import NullDashboardEventSink
        from local_qwen_planner import LocalQwenPlanner
        from rpent_tools import RobotAgentLoop, RobotToolkit, RULES
        versions = {'repos': {}, 'packages': json.loads(subprocess.check_output(
            [os.sys.executable, '-m', 'pip', 'list', '--format=json'], text=True)),
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (ROOT / 'baseline').glob('*.py')}}
        for name, path in {'research': str(ROOT), 'RPent': '/root/gpufree-data/RPent',
            'RoboDojo': cfg['paths']['robodojo'], 'XPolicyLab': cfg['paths']['robodojo']+'/XPolicyLab'}.items():
            versions['repos'][name] = subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
        (run / 'versions.json').write_text(json.dumps(versions, indent=2))
        model = LocalQwenPlanner(cfg['planner'], run)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); actor_port = sock.getsockname()[1]
        spawn('actor', [cfg['paths']['actor_python'], str(ROOT/'baseline/actor_server.py'),
                       '--run', str(run), '--port', str(actor_port)])
        deadline = time.monotonic() + cfg['runtime']['startup_timeout']
        while not any(e['event'] == 'actor_socket_ready' for e in read_jsonl(run/'events.jsonl')):
            if children['actor'].poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('OpenWAM startup failed')
            time.sleep(1)
        conn, worker_conn = Pipe(duplex=True)  # anonymous AF_UNIX robot IPC, not a model server
        spawn('isaac', [cfg['paths']['runtime_python'], str(ROOT/'baseline/rpent_simulator.py'),
            '--run', str(run), '--actor-port', str(actor_port), '--fd', str(worker_conn.fileno())],
            runtime=True, pass_fds=(worker_conn.fileno(),))
        worker_conn.close()
        if not conn.poll(cfg['runtime']['startup_timeout']):
            raise TimeoutError('Isaac initialization timeout')
        initial = conn.recv()
        driver = RobotConnection(conn, cfg['runtime']['robot_tool_timeout'])
        sink = NullDashboardEventSink()
        toolkit = RobotToolkit(driver, run, sink)
        loop = RobotAgentLoop(model, max_tokens=cfg['planner']['max_generation_tokens'],
                             reasoning_effort='high', dashboard_events=sink)
        episode_start = time.monotonic()
        event(run, 'coexistence_before_episode', memory_mib=int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)))
        result = loop.solve(system_prompt=RULES,
            user_message=[initial['instruction'], BinaryContent(initial['_image_bytes'], media_type='image/png')],
            toolkit=toolkit, max_turns=cfg['rpent']['max_turns'])
        summary.update(rpent_stats=result.stats, planner_error=result.error,
            finish=result.finish_result, completed=result.error is None, model_load_count=model.load_count)
        (run/'rpent_result.json').write_text(json.dumps({'messages': result.messages,
            'stats': result.stats, 'error': result.error, 'finish': result.finish_result}, default=str, indent=2))
        conn.send({'tool': 'close', 'arguments': {}})
        children['isaac'].wait(timeout=cfg['runtime']['shutdown_timeout'])
    except BaseException as exc:
        summary['error'] = repr(exc); traceback.print_exc()
        if conn is not None:
            try:
                conn.send({'tool': 'close', 'arguments': {}})
                children['isaac'].wait(timeout=cfg['runtime']['shutdown_timeout'])
            except Exception:
                pass
    finally:
        summary['wall_time'] = time.monotonic()-(episode_start or started)
        cleanup(run)
        halted.set(); thread.join(timeout=3)
        if conn is not None: conn.close()
        sim = json.loads((run/'simulator.json').read_text()) if (run/'simulator.json').exists() else {}
        summary.update(simulator=sim, success=sim.get('success', False), sim_time=sim.get('sim_time'))
        p, a, gpu = [read_jsonl(run/name) for name in ('planner.jsonl','actor.jsonl','gpu.jsonl')]
        latencies = sorted(x['request_latency'] for x in p)
        summary.update(planner_calls=len(p), input_tokens=sum(x.get('input_tokens') or 0 for x in p),
            output_tokens=sum(x.get('output_tokens') or 0 for x in p), reasoning_tokens=None,
            planner_mean_latency=statistics.mean(latencies) if p else None,
            planner_p50_latency=statistics.median(latencies) if p else None,
            planner_p95_latency=latencies[min(len(p)-1, int(.95*len(p)))] if p else None,
            OpenWAM_mean_latency=statistics.mean(x['latency'] for x in a) if a else None,
            planner_total_wall_time=sum(latencies), OpenWAM_total_inference_time=sum(x['latency'] for x in a),
            peak_VRAM_MiB=max((g['memory_used_mib'] for g in gpu if 'memory_used_mib' in g), default=None),
            OOM='out of memory' in (run/'stdout.log').read_text(errors='replace').lower())
        failure = str(summary.get('error') or summary.get('planner_error') or sim.get('error') or '')
        summary['failure_taxonomy'] = ('TIMING / PLANNER_DELIBERATION' if any(x.get('timeout') or x.get('generation_limit_hit') for x in p)
            else 'INTERFACE' if failure else 'UNKNOWN' if not summary['success'] else None)
        summary['pipeline_ready'] = bool(summary['completed'] and sim.get('clean_exit') and not sim.get('error')
            and len(p) >= 2 and len(a) >= 2 and sim.get('env_steps', 0) > 64 and not summary['OOM'])
        summary['end_to_end_wall_time'] = time.monotonic() - started
        (run/'summary.json').write_text(json.dumps(summary, indent=2))
        event(run, 'episode_end', **summary)
        timeline = ['# RPent episode timeline', '']
        for row in p:
            timeline.append(f"- {row['timestamp']}: RPent turn {row['rpent_turn_id']}, {row.get('parsed_tool')}, "
                            f"{row.get('subgoal')}, {row['request_latency']:.2f}s, output {row.get('output_tokens')} tokens")
        (run/'timeline.md').write_text('\n'.join(timeline)+'\n')
        print('RPENT_RESULT', json.dumps(summary), flush=True)
    raise SystemExit(0 if summary['pipeline_ready'] else 1)


if __name__ == '__main__':
    main()
