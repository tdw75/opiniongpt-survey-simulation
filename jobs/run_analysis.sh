#!/usr/bin/env bash
set -e

EXPERIMENT_NAME="$1"
shift

python3 scripts/results_to_csv.py -experiment_name "$EXPERIMENT_NAME" "$@"
python3 scripts/clean_results.py -experiment_name "$EXPERIMENT_NAME" "$@"
python3 scripts/generate_marginal_analysis.py -experiment_name "$EXPERIMENT_NAME" "$@"
python3 scripts/generate_correlation_analysis.py -experiment_name "$EXPERIMENT_NAME" "$@"
python3 scripts/generate_visualisations.py -experiment_name "$EXPERIMENT_NAME" "$@"
