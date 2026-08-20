#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/generate_job_list.sh"
exec bash "${SCRIPT_DIR}/run_matrix_experiment.sh" "$@"
