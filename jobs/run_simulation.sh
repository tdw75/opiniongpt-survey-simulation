#!/usr/bin/env bash
set -e

EXPERIMENT_NAME="$1"
shift

python3 scripts/run_all_models.py -experiment_name "$EXPERIMENT_NAME" "$@"
