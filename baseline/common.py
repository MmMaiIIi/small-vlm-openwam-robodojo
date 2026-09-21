"""Small file-based run records shared by the three isolated processes."""
import datetime
import json
import time
from pathlib import Path


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def append(path, record):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


def event(run, name, **fields):
    append(Path(run) / 'events.jsonl', dict(timestamp=timestamp(), unix_time=time.time(), event=name, **fields))


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
