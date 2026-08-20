# shellcheck shell=bash

backend_submit() {
    local dep=()
    [[ -n "${B_DEP:-}" ]] && dep=(-a "${B_DEP}")
    local jobtype=()
    [[ -n "${OAR_JOB_TYPE:-}" ]] && jobtype=(-t "${OAR_JOB_TYPE}")
    local out
    out="$(oarsub \
        -q "${OAR_QUEUE}" \
        -p "cluster='${B_PARTITION}'" \
        -l "host=${B_NODES},walltime=${B_WALLTIME}" \
        -n "${B_TAG}" \
        -O "${B_LOG_DIR}/${B_TAG}.%jobid%.out" \
        -E "${B_LOG_DIR}/${B_TAG}.%jobid%.err" \
        ${jobtype[@]+"${jobtype[@]}"} \
        ${dep[@]+"${dep[@]}"} \
        "${B_CMD}")"
    printf '%s\n' "${out}" | parse_oar_job_id
}

backend_list_nodes() {
    sort -u "${OAR_NODEFILE}"
}

backend_remote_exec() {
    local host="$1"; shift
    oarsh "${host}" "$*"
}
