#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: run_intrusion_experiment.sh [options]
  --scheduler slurm|oar   backend to use (default from cluster.conf)
  --partition NAME        Slurm partition / OAR cluster name
  --nodes N               total nodes per run (1 server + N-1 clients)
  --walltime T            walltime per run (default: backend default)
  --when day|night        OAR -t constraint (day=workhours <=1h, night=nights/weekends)
  --from N                skip the first N runs (resume)
  --limit K               submit at most K runs
EOF
}

FROM=0; LIMIT=0; WALLTIME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scheduler) export SCHEDULER="$2"; shift 2 ;;
        --partition) export PARTITION="$2"; shift 2 ;;
        --nodes)     export NUM_NODES="$2"; shift 2 ;;
        --walltime)  WALLTIME="$2"; shift 2 ;;
        --when)      export OAR_JOB_TYPE="$2"; shift 2 ;;
        --from)      FROM="$2"; shift 2 ;;
        --limit)     LIMIT="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; usage >&2; exit 2 ;;
    esac
done

source "${SCRIPTS_DIR}/conf/experiments.conf"
source "${SCRIPTS_DIR}/conf/cluster.conf"
source "${SCRIPTS_DIR}/lib/common.sh"
source "${SCRIPTS_DIR}/lib/backend_${SCHEDULER}.sh"

case "${OAR_JOB_TYPE:-}" in
    ""|day|night) ;;
    *) echo "ERROR: --when must be 'day' or 'night'." >&2; exit 2 ;;
esac

WALLTIME="${WALLTIME:-${DEFAULT_WALLTIME}}"
LOG_DIR="${SCRIPTS_DIR}/logs"
STATE_DIR="${HOME_PROJECT_DIR}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

NUM_CLIENTS=$((NUM_NODES - 1))

RUNS=()
for rep in $(seq 1 "${INTRUSION_REPS}"); do
    for combo in ${INTRUSION_COMBOS}; do
        model="${combo%%=*}"
        datasets_csv="${combo#*=}"
        IFS=',' read -ra datasets <<< "${datasets_csv}"
        for dataset_raw in "${datasets[@]}"; do
            dataset="${dataset_raw//_/ }"
            for mode in on off; do
                RUNS+=("${dataset}|${INTRUSION_ALGO}|${model}|${NUM_CLIENTS}|${rep}|${mode}")
            done
        done
    done
done

total_all="${#RUNS[@]}"
echo "EnerFL intrusion experiment"
echo "   scheduler    = ${SCHEDULER}"
echo "   partition    = ${PARTITION}"
echo "   nodes/run    = ${NUM_NODES} (1 server + $((NUM_NODES - 1)) clients)"
echo "   clients      = ${NUM_CLIENTS} (1 supernode per client node)"
echo "   walltime/run = ${WALLTIME}"
echo "   when         = ${OAR_JOB_TYPE:-auto}"
echo "   runs         = ${total_all} (profiler on/off paired)"
echo "   start line   = $((FROM + 1))   limit = ${LIMIT:-all}"

prev_job=""; run_num=0; submitted=0
for line in "${RUNS[@]}"; do
    run_num=$((run_num + 1))
    (( run_num <= FROM )) && continue
    (( LIMIT > 0 && submitted >= LIMIT )) && break
    submitted=$((submitted + 1))
    IFS='|' read -r dataset algo model c_count rep mode <<< "${line}"
    exec_id="in${run_num}r${rep}${mode}"

    env_file="${STATE_DIR}/run_${exec_id}.env"
    cat > "${env_file}" <<EOF
SCHEDULER='${SCHEDULER}'
PARTITION='${PARTITION}'
NUM_NODES='${NUM_NODES}'
DATASET='${dataset}'
ALGO='${algo}'
MODEL_NAME='${model}'
NUM_CLIENTS='${c_count}'
EXEC_ID='${exec_id}'
PROFILER_MODE='${mode}'
EOF

    B_TAG="enerfl-${exec_id}"
    B_WALLTIME="${WALLTIME}"
    B_NODES="${NUM_NODES}"
    B_PARTITION="${PARTITION}"
    B_DEP="${prev_job}"
    B_LOG_DIR="${LOG_DIR}"
    B_CMD="bash '${SCRIPTS_DIR}/job.sh' '${env_file}'"
    export B_TAG B_WALLTIME B_NODES B_PARTITION B_DEP B_LOG_DIR B_CMD

    job_id="$(backend_submit)"
    [[ -n "${job_id}" ]] || { echo "ERROR: submit failed for run ${run_num}." >&2; exit 1; }

    echo "-- Run ${run_num}/${total_all} (#${submitted}): ${dataset} | ${algo} | ${model} | c=${c_count} | rep=${rep} | ${mode}  -> ${job_id}${prev_job:+ (after ${prev_job})}"
    prev_job="${job_id}"
done

echo ""
echo "${submitted} run(s) queued (of ${total_all}), chained serially."
echo "Results: ${HOME_PROJECT_DIR}/results/"
