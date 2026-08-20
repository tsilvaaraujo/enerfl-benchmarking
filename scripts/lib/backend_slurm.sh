# shellcheck shell=bash

backend_submit() {
    local dep=()
    [[ -n "${B_DEP:-}" ]] && dep=(--dependency=afterany:"${B_DEP}")
    sbatch --parsable \
        --job-name="${B_TAG}" \
        --partition="${B_PARTITION}" \
        --nodes="${B_NODES}" \
        --ntasks-per-node=1 \
        --time="${B_WALLTIME}" \
        --output="${B_LOG_DIR}/${B_TAG}-%j.out" \
        --error="${B_LOG_DIR}/${B_TAG}-%j.err" \
        ${dep[@]+"${dep[@]}"} \
        --wrap="${B_CMD}"
}

backend_list_nodes() {
    scontrol show hostnames "${SLURM_JOB_NODELIST}"
}

backend_remote_exec() {
    local host="$1"; shift
    srun --nodes=1 --ntasks=1 --nodelist="${host}" --overlap bash -lc "$*"
}
