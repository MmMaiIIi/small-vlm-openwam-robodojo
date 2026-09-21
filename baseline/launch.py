"""Isolated three-process launcher, sampled GPU telemetry, and run summary."""
import argparse
import hashlib
import json
import os
import signal
import shutil
import socket
import statistics
import subprocess
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import yaml
from common import append, event, read_jsonl, timestamp
from stop import cleanup, identity

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['fill_pen_holder', 'classify_objects', 'put_bottles_into_dustbin',
         'play_tic_tac_toe', 'fill_egg_holder', 'organize_table', 'make_kong', 'play_stacking_toy']

def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()

def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def versions(cfg):
    repos = {'RoboDojo': cfg['paths']['robodojo'],
             'XPolicyLab': cfg['paths']['robodojo'] + '/XPolicyLab',
             'cuRobo': cfg['paths']['robodojo'] + '/third_party/curobo',
             'RPent': '/root/gpufree-data/RPent'}
    result = {'timestamp': timestamp(), 'repos': {}, 'environments': {}, 'research_files_sha256': {}}
    for name, path in repos.items():
        result['repos'][name] = {'path': path, 'commit': command(['git', 'rev-parse', 'HEAD'], path),
            'status': command(['git', 'status', '--porcelain'], path),
            'diff_sha256': hashlib.sha256(command(['git', 'diff'], path).encode()).hexdigest()}
    for name in ('runtime', 'actor', 'qwen'):
        python = cfg['paths'][name + '_python']
        result['environments'][name] = {'python': python, 'version': command([python, '-V']),
            'packages': command([python, '-m', 'pip', 'list', '--format=json'])}
    for path in sorted((ROOT / 'baseline').glob('*.py')):
        result['research_files_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result['nvidia_smi'] = command(['nvidia-smi'])
    result['openwam_revision'] = '2c1302294e3ba8319bbdb2c803b7a27de9292d03'
    result['qwen_revision'] = '15852e8c16360a2fea060d615a32b45270f8a8fc'
    return result

def aggregate(run, launch_started):
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    summary_path = run / 'summary.json'
    s = json.loads(summary_path.read_text()) if summary_path.exists() else {
        'completed': False, 'success': False, 'failure_reason': 'No simulator summary', 'failure_category': 'INTERFACE'}
    p, a, gpu, events = [read_jsonl(run / name) for name in
                        ('planner.jsonl', 'actor.jsonl', 'gpu.jsonl', 'events.jsonl')]
    s.update(task=cfg['environment']['task'], seed=cfg['environment']['seed'],
        summary_generated_at=timestamp(), postprocessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        end_to_end_wall_clock=time.monotonic() - launch_started,
        planner_calls=len(p), total_input_tokens=sum(x.get('input_tokens') or 0 for x in p),
        total_output_tokens=sum(x.get('output_tokens') or 0 for x in p),
        total_reasoning_tokens=sum(x['reasoning_tokens'] for x in p) if p and all(x.get('reasoning_tokens') is not None for x in p) else None,
        total_cached_tokens=sum(x['cached_tokens'] for x in p) if p and all(x.get('cached_tokens') is not None for x in p) else None,
        token_usage_complete=all(x.get('input_tokens') is not None and x.get('output_tokens') is not None for x in p),
        reasoning_responses=sum(bool((x.get('raw_response') or {}).get('choices', [{}])[0].get('message', {}).get('reasoning'))
                                for x in p if isinstance(x.get('raw_response'), dict)),
        total_planner_latency=sum(x['request_latency'] for x in p),
        total_WAM_inference_time=sum(x['latency'] for x in a),
        mean_planner_latency=statistics.mean(x['request_latency'] for x in p) if p else None,
        mean_WAM_latency=statistics.mean(x['latency'] for x in a) if a else None,
        peak_total_VRAM_MiB=max((x['memory_used_mib'] for x in gpu if 'memory_used_mib' in x), default=None),
        peak_process_VRAM_MiB={role: max((x['roles'][role] for x in gpu if role in x.get('roles', {})), default=None)
                               for role in ('qwen', 'actor', 'isaac')},
        VRAM_sampling_seconds=cfg['logging']['gpu_sample_seconds'],
        actual_actor_inferences=len(a), stale_actions_dropped=sum(x.get('stale_actions_dropped') or 0 for x in a))
    s['OOM'] = 'out of memory' in (run / 'stdout.log').read_text(errors='replace').lower()
    if (run / 'video.mp4').exists():
        try:
            s['video'] = json.loads(command(['ffprobe', '-v', 'error', '-show_entries',
                'stream=width,height,nb_frames,duration', '-of', 'json', str(run / 'video.mp4')]))
        except Exception as exc:
            s['video_error'] = repr(exc)
    s['pipeline_valid'] = bool(s.get('completed') and s.get('clean_exit') and
        len([x for x in p if x.get('parsed_action')]) >= 2 and a and s.get('env_steps', 0) > 0
        and s.get('video') and not s['OOM'])
    summary_path.write_text(json.dumps(s, indent=2))
    ready = next((e['unix_time'] for e in events if e['event'] == 'environment_ready'), None)
    if ready:
        concurrent = next((g for g in gpu if g.get('unix_time', 0) >= ready), None)
        (run / 'coexistence.json').write_text(json.dumps(concurrent, indent=2))
    # Some containers expose device memory but hide NVML per-process records.
    # Never report unavailable process measurements as zero.
    resident = {}
    previous = next((g['memory_used_mib'] for g in gpu if 'memory_used_mib' in g), 0)
    for role, marker in [('qwen', 'service_ready'), ('actor', 'service_ready'), ('isaac', 'environment_ready')]:
        milestone = next((e for e in events if e['event'] == marker and (role == 'isaac' or e.get('role') == role)), None)
        if milestone:
            nearby = [g for g in gpu if 'memory_used_mib' in g and g.get('unix_time', 0) >= milestone['unix_time']]
            if nearby:
                total = nearby[0]['memory_used_mib']
                resident[role] = total - previous
                previous = total
    s['sequential_startup_resident_estimates_MiB'] = resident
    s['component_VRAM_note'] = 'Startup total-memory deltas are estimates; per-process null means NVML unavailable in container.'
    summary_path.write_text(json.dumps(s, indent=2))
    lines = ['# Episode timeline', '', f"Task: {s['task']}, seed {s['seed']}", '',
        'Wall time is relative to environment readiness; sim time is control steps / 25 Hz.', '']
    for e in events:
        if e['event'] in ('decision', 'actor_inference', 'observation_returned', 'episode_error', 'episode_end'):
            t = e['unix_time'] - (ready or events[0]['unix_time'])
            fields = {k: v for k, v in e.items() if k not in ('timestamp', 'unix_time', 'event', 'traceback')}
            lines.append(f"- t={t:.2f}s {e['event']}: `{json.dumps(fields, ensure_ascii=False)}`")
    (run / 'timeline.md').write_text('\n'.join(lines) + '\n')
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, default='fill_pen_holder')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', default=str(ROOT / 'configs/baseline.yaml'))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg['environment'].update(task=args.task, seed=args.seed)
    run = ROOT / 'runs' / (time.strftime('%Y%m%dT%H%M%S') + '_' + args.task + '_' + str(args.seed))
    run.mkdir(parents=True)
    (ROOT / '.active_run').write_text(str(run))
    (run / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    source = run / 'source'
    source.mkdir()
    for path in (ROOT / 'baseline').glob('*.py'):
        shutil.copy2(path, source / path.name)
    for name in ('planner.jsonl', 'actor.jsonl', 'events.jsonl', 'gpu.jsonl', 'stdout.log'):
        (run / name).touch()
    print('RUN_DIRECTORY=' + str(run), flush=True)
    launch_started = time.monotonic()
    processes, records = {}, []
    halted = threading.Event()
    log = (run / 'stdout.log').open('ab', buffering=0)

    def spawn(role, argv, runtime=False):
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('LD_LIBRARY_PATH', None)
        env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', ROBOT_AGENT_EXP_RUN=str(run))
        env['PYTHONPATH'] = cfg['paths']['robodojo'] + ':' + cfg['paths']['robodojo'] + '/XPolicyLab'
        if runtime:
            argv = ['bash', '-c', 'source /root/gpufree-data/deployment-logs/robodojo-runtime.env; exec "$@"', 'baseline'] + argv
        p = subprocess.Popen(argv, cwd=cfg['paths']['robodojo'] if runtime else ROOT,
                             env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[role] = p
        records.append({'role': role, 'pid': p.pid, 'start_ticks': identity(p.pid), 'argv': argv})
        temp = run / 'pids.tmp'
        temp.write_text(json.dumps(records, indent=2))
        temp.replace(run / 'pids.json')
        event(run, 'process_started', role=role, pid=p.pid)
        return p

    def monitor():
        while not halted.is_set():
            try:
                mem, util = command(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits']).split(',')
                sample = dict(timestamp=timestamp(), unix_time=time.time(), memory_used_mib=int(mem),
                              utilization_percent=int(util), roles={})
                for line in command(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits']).splitlines():
                    pid, used = (int(v.strip()) for v in line.split(','))
                    try:
                        group = os.getpgid(pid)
                    except ProcessLookupError:
                        continue
                    for rec in list(records):
                        if rec['pid'] == group:
                            sample['roles'][rec['role']] = sample['roles'].get(rec['role'], 0) + used
                append(run / 'gpu.jsonl', sample)
            except Exception as exc:
                append(run / 'gpu.jsonl', dict(timestamp=timestamp(), error=repr(exc)))
            halted.wait(cfg['logging']['gpu_sample_seconds'])

    def wait_ready(role, check):
        deadline = time.monotonic() + cfg['harness']['startup_timeout']
        while time.monotonic() < deadline:
            if processes[role].poll() is not None:
                raise RuntimeError(role + ' exited during startup')
            try:
                if check():
                    event(run, 'service_ready', role=role)
                    print(role + ' ready', flush=True)
                    return
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise TimeoutError(role + ' startup timeout')

    def interrupted(signum, frame):
        raise KeyboardInterrupt('signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for key in ('runtime_python', 'actor_python', 'qwen_python', 'robodojo'):
            assert Path(cfg['paths'][key]).exists(), key
        assert Path(cfg['actor']['checkpoint'], 'config.yaml').exists()
        assert Path(cfg['planner']['model_path'], 'config.json').exists()
        (run / 'versions.json').write_text(json.dumps(versions(cfg), indent=2))
        qp, ap = port(), port()
        (run / 'ports.json').write_text(json.dumps({'qwen': qp, 'actor': ap}))
        p = cfg['planner']
        spawn('qwen', [cfg['paths']['qwen_python'], '-m', 'vllm.entrypoints.cli.main', 'serve', p['model_path'],
            '--served-model-name', p['model'], '--host', '127.0.0.1', '--port', str(qp),
            '--max-model-len', str(p['context_length']), '--max-num-seqs', '1',
            '--gpu-memory-utilization', str(p['gpu_memory_utilization']), '--kv-cache-memory-bytes', str(p['kv_cache_bytes']),
            '--enforce-eager', '--reasoning-parser', 'qwen3', '--limit-mm-per-prompt', '{"image":1,"video":0}'])
        def health():
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f'http://127.0.0.1:{qp}/health', timeout=2) as response:
                return response.status == 200
        wait_ready('qwen', health)
        spawn('actor', [cfg['paths']['actor_python'], str(source / 'actor_server.py'), '--run', str(run), '--port', str(ap)])
        def actor_ready():
            return any(e['event'] == 'actor_socket_ready' for e in read_jsonl(run / 'events.jsonl'))
        wait_ready('actor', actor_ready)
        sim = spawn('isaac', [cfg['paths']['runtime_python'], str(source / 'simulator.py'),
            '--run', str(run), '--actor-port', str(ap), '--qwen-port', str(qp)], runtime=True)
        deadline = time.monotonic() + cfg['harness']['episode_timeout']
        while sim.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError('episode watchdog timeout')
            for role in ('qwen', 'actor'):
                if processes[role].poll() is not None:
                    raise RuntimeError(role + ' exited during episode')
            time.sleep(2)
        event(run, 'simulator_process_exit', returncode=sim.returncode)
    except BaseException as exc:
        event(run, 'launcher_error', error=repr(exc), traceback=traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        cleanup(run)
        for p in processes.values():
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        halted.set()
        thread.join(timeout=5)
        log.close()
        summary = aggregate(run, launch_started)
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['pipeline_valid'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
