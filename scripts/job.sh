#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:?usage: job.sh <env_file>}"

source "${SCRIPTS_DIR}/conf/cluster.conf"
source "${ENV_FILE}"
source "${SCRIPTS_DIR}/lib/common.sh"
source "${SCRIPTS_DIR}/lib/backend_${SCHEDULER}.sh"

CONTAINER_CMD="$(detect_container_cmd)"
JOB_ID="${OAR_JOB_ID:-${SLURM_JOB_ID:-local}}"

mapfile -t NODES < <(backend_list_nodes)
[[ "${#NODES[@]}" -ge 2 ]] || { echo "ERROR: need >=2 nodes, got ${#NODES[@]}." >&2; exit 1; }

self_short="$(hostname -s)"
CLIENT_HOSTS=()
for n in "${NODES[@]}"; do
    [[ "${n%%.*}" == "${self_short}" ]] && continue
    CLIENT_HOSTS+=("${n}")
done
client_node_count="${#CLIENT_HOSTS[@]}"
[[ "${client_node_count}" -ge 1 ]] || { echo "ERROR: no client nodes resolved." >&2; exit 1; }

if (( NUM_CLIENTS % client_node_count != 0 )); then
    echo "ERROR: NUM_CLIENTS=${NUM_CLIENTS} not divisible by ${client_node_count} client node(s)." >&2
    exit 1
fi
CLIENTS_PER_NODE=$(( NUM_CLIENTS / client_node_count ))

SERVER_IP="$(primary_ip)"
echo "[job ${JOB_ID}] server=${self_short} ip=${SERVER_IP} clients=${CLIENT_HOSTS[*]}"
echo "[job ${JOB_ID}] NUM_CLIENTS=${NUM_CLIENTS} clients/node=${CLIENTS_PER_NODE}"

stage_project

env_prefix() {
    printf "CONTAINER_CMD='%s' " "${CONTAINER_CMD}"
    printf "NODE_PROJECT_DIR='%s' " "${NODE_PROJECT_DIR}"
    printf "HOME_PROJECT_DIR='%s' " "${HOME_PROJECT_DIR}"
    printf "SUPERNODE_SIF='%s' " "${SUPERNODE_SIF}"
    printf "SUPERLINK_IP='%s' " "${SERVER_IP}"
    printf "DATASET='%s' " "${DATASET}"
    printf "ALGO='%s' " "${ALGO}"
    printf "MODEL_NAME='%s' " "${MODEL_NAME}"
    printf "NUM_PARTITIONS='%s' " "${NUM_CLIENTS}"
    printf "CLIENTS_PER_NODE='%s' " "${CLIENTS_PER_NODE}"
    printf "PROFILER_MODE='%s' " "${PROFILER_MODE}"
    printf "JOB_ID='%s' " "${JOB_ID}"
}

CONTAINER_CMD="${CONTAINER_CMD}" \
NODE_PROJECT_DIR="${NODE_PROJECT_DIR}" \
HOME_PROJECT_DIR="${HOME_PROJECT_DIR}" \
SUPERLINK_SIF="${SUPERLINK_SIF}" \
SUPERLINK_IP="${SERVER_IP}" \
DATASET="${DATASET}" ALGO="${ALGO}" MODEL_NAME="${MODEL_NAME}" \
NUM_CLIENTS="${NUM_CLIENTS}" EXEC_ID="${EXEC_ID}" \
PROFILER_MODE="${PROFILER_MODE}" JOB_ID="${JOB_ID}" \
    bash "${SCRIPTS_DIR}/run_server.sh" &
server_pid=$!

client_pids=()
idx=0
for host in "${CLIENT_HOSTS[@]}"; do
    base=$(( idx * CLIENTS_PER_NODE ))
    remote_cmd="$(env_prefix)PARTITION_ID_BASE='${base}' bash '${SCRIPTS_DIR}/run_client.sh'"
    backend_remote_exec "${host}" "${remote_cmd}" &
    client_pids+=("$!")
    idx=$(( idx + 1 ))
done

cleanup() {
    for p in ${client_pids[@]+"${client_pids[@]}"}; do kill "${p}" 2>/dev/null || true; done
}
trap cleanup EXIT TERM INT

set +e
wait "${server_pid}"
rc=$?
set -e
echo "[job ${JOB_ID}] server finished rc=${rc}"
exit "${rc}"
