#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /root/gpufree-data/conda-envs/robodojo-runtime/bin/python "$PROJECT_ROOT/baseline/stop.py" "$@"
