#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec 9>"$PROJECT_ROOT/.baseline.lock"
flock -n 9 || { echo 'Another baseline launcher owns this GPU project lock.' >&2; exit 1; }
source /root/gpufree-data/deployment-logs/data-disk.env
cleanup() {
  /root/gpufree-data/conda-envs/robodojo-runtime/bin/python "$PROJECT_ROOT/baseline/stop.py" || true
}
trap cleanup EXIT INT TERM
/root/gpufree-data/conda-envs/robodojo-runtime/bin/python "$PROJECT_ROOT/baseline/launch.py" "$@"
