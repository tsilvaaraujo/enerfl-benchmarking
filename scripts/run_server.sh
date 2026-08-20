#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPTS_DIR}/conf/cluster.conf"
source "${SCRIPTS_DIR}/lib/common.sh"

CONTAINER_CMD="${CONTAINER_CMD:-$(detect_container_cmd)}"
SUPERLINK_IP="${SUPERLINK_IP:?}"
PROJECT_DIR="${NODE_PROJECT_DIR}"

stage_project

RESULTS_DIR="${PROJECT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

DATASET_TAG="${DATASET:-unknown}"
ALGO_TAG="${ALGO:-FedAvg}"
MODEL_TAG="${MODEL_NAME:-unknown}"
CLIENTS_TAG="${NUM_CLIENTS:-0}"
EXEC_TAG="${EXEC_ID:-0}"
PROFILER_MODE="${PROFILER_MODE:-on}"
case "${PROFILER_MODE}" in
    off|false|0) PROF_TAG="p0" ;;
    *)           PROF_TAG="p1" ;;
esac
CSV_NAME="metrics_${DATASET_TAG}_${ALGO_TAG}_${MODEL_TAG}_c${CLIENTS_TAG}_r${EXEC_TAG}_${PROF_TAG}_${JOB_ID:-local}.csv"
CSV_NAME="${CSV_NAME// /_}"
PROFILE_CSV_HOST="${RESULTS_DIR}/${CSV_NAME}"
PROFILE_CSV_CONTAINER="/results/${CSV_NAME}"
echo "[server] $(hostname) ip=${SUPERLINK_IP} csv=${PROFILE_CSV_HOST} mode=${PROFILER_MODE}"

set +e
"${CONTAINER_CMD}" run \
    --bind "${PROJECT_DIR}:/project" \
    --bind "${RESULTS_DIR}:/results" \
    --env "SUPERLINK_IP=${SUPERLINK_IP}" \
    --env "PROFILE_CSV=${PROFILE_CSV_CONTAINER}" \
    --env "ALGO=${ALGO_TAG}" \
    --env "MODEL_NAME=${MODEL_TAG}" \
    --env "DATASET=${DATASET_TAG}" \
    --env "NUM_CLIENTS=${CLIENTS_TAG}" \
    --env "EXEC_ID=${EXEC_TAG}" \
    --env "PROFILER_MODE=${PROFILER_MODE}" \
    "${PROJECT_DIR}/${SUPERLINK_SIF}"
rc=$?
set -e

if [[ "${NODE_PROJECT_DIR}" != "${HOME_PROJECT_DIR}" && -f "${PROFILE_CSV_HOST}" ]]; then
    mkdir -p "${HOME_PROJECT_DIR}/results"
    cp -f "${PROFILE_CSV_HOST}" "${HOME_PROJECT_DIR}/results/${CSV_NAME}"
fi

echo "[server] finished rc=${rc}"
exit "${rc}"
