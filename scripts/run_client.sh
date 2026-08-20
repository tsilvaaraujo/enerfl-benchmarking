#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPTS_DIR}/conf/cluster.conf"
source "${SCRIPTS_DIR}/lib/common.sh"

CONTAINER_CMD="${CONTAINER_CMD:-$(detect_container_cmd)}"
SUPERLINK_IP="${SUPERLINK_IP:?}"
PROJECT_DIR="${NODE_PROJECT_DIR}"

stage_project

CLIENTS_PER_NODE="${CLIENTS_PER_NODE:-1}"
PARTITION_ID_BASE="${PARTITION_ID_BASE:-0}"
NUM_PARTITIONS="${NUM_PARTITIONS:-2}"
SIF="${PROJECT_DIR}/${SUPERNODE_SIF}"

LOG_DIR="${PROJECT_DIR}/state/${JOB_ID:-local}/client_logs"
mkdir -p "${LOG_DIR}"

echo "[client] $(hostname) clients/node=${CLIENTS_PER_NODE} base=${PARTITION_ID_BASE} link=${SUPERLINK_IP}"

pids=()
for i in $(seq 0 $((CLIENTS_PER_NODE - 1))); do
    pid_id=$(( PARTITION_ID_BASE + i ))
    port=$(( 9094 + i ))
    log_file="${LOG_DIR}/supernode_p${pid_id}_$(hostname -s).log"
    echo "[client] spawn partition-id=${pid_id} port=${port} -> ${log_file}"

    "${CONTAINER_CMD}" exec \
        --bind "${PROJECT_DIR}:/project:ro" \
        --env  "SUPERLINK_IP=${SUPERLINK_IP}" \
        --env  "PARTITION_ID=${pid_id}" \
        --env  "NUM_PARTITIONS=${NUM_PARTITIONS}" \
        --env  "ALGO=${ALGO:-FedAvg}" \
        --env  "MODEL_NAME=${MODEL_NAME:-unknown}" \
        --env  "DATASET=${DATASET:-unknown}" \
        --env  "PROFILER_MODE=${PROFILER_MODE:-on}" \
        "${SIF}" \
        flower-supernode \
            --insecure \
            --superlink               "${SUPERLINK_IP}:9092" \
            --clientappio-api-address "0.0.0.0:${port}" \
            --isolation               subprocess \
            --node-config             "partition-id=${pid_id} num-partitions=${NUM_PARTITIONS}" \
        > "${log_file}" 2>&1 &
    pids+=("$!")
done

cleanup() {
    for p in "${pids[@]}"; do kill "${p}" 2>/dev/null || true; done
}
trap cleanup EXIT TERM INT

rc=0
for p in "${pids[@]}"; do
    if ! wait "${p}"; then
        this_rc=$?
        echo "[client] supernode pid=${p} exited rc=${this_rc}"
        (( rc == 0 )) && rc=${this_rc}
    fi
done

echo "[client] all supernodes finished rc=${rc}"
exit "${rc}"
