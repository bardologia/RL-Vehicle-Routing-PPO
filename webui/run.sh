#!/usr/bin/env bash
set -euo pipefail

PYTHON="$HOME/miniconda3/envs/routing-ppo/bin/python"
exec "$PYTHON" "$(dirname "$0")/serve.py" "$@"
