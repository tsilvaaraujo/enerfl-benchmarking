# shellcheck shell=bash

detect_container_cmd() {
    if [[ -n "${CONTAINER_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
        module load "${CONTAINER_MODULE}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${CONTAINER_CMD:-}" ]]; then
        echo "${CONTAINER_CMD}"
        return 0
    fi
    if command -v apptainer >/dev/null 2>&1; then
        echo "apptainer"
    elif command -v singularity >/dev/null 2>&1; then
        echo "singularity"
    else
        echo "ERROR: no apptainer/singularity runtime found on $(hostname)." >&2
        return 1
    fi
}

primary_ip() {
    local ips ip
    ips="$(hostname -I 2>/dev/null || true)"
    if [[ -n "${SERVER_IP_PREFIX:-}" ]]; then
        for ip in ${ips}; do
            [[ "${ip}" == "${SERVER_IP_PREFIX}"* ]] && { echo "${ip}"; return 0; }
        done
    fi
    for ip in ${ips}; do
        [[ "${ip}" != "127."* ]] && { echo "${ip}"; return 0; }
    done
    echo "ERROR: could not determine an IP for $(hostname)." >&2
    return 1
}

stage_project() {
    [[ "${NODE_PROJECT_DIR}" == "${HOME_PROJECT_DIR}" ]] && return 0
    echo "[stage] $(hostname): syncing ${HOME_PROJECT_DIR} -> ${NODE_PROJECT_DIR}"
    mkdir -p "${NODE_PROJECT_DIR}"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete --exclude 'results/' --exclude 'state/' --exclude '.git/' \
            "${HOME_PROJECT_DIR}/" "${NODE_PROJECT_DIR}/"
    else
        cp -a "${HOME_PROJECT_DIR}/." "${NODE_PROJECT_DIR}/"
    fi
}

parse_oar_job_id() {
    grep -oE 'OAR_JOB_ID=[0-9]+' | head -n1 | cut -d= -f2
}

wait_for_port() {
    local host="$1" port="$2" timeout="${3:-120}" elapsed=0
    while ! (exec 3<>"/dev/tcp/${host}/${port}") 2>/dev/null; do
        sleep 2
        elapsed=$((elapsed + 2))
        if (( elapsed >= timeout )); then
            echo "ERROR: timed out waiting for ${host}:${port}" >&2
            return 1
        fi
    done
    exec 3>&- 2>/dev/null || true
}
