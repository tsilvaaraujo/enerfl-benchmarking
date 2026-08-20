#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/conf/experiments.conf"

JOB_FILE="${SCRIPT_DIR}/job_list.txt"
: > "${JOB_FILE}"

for c in ${CLIENTS_COUNT}; do
    if (( c % 2 != 0 )); then
        echo "ERROR: CLIENTS_COUNT must contain only even numbers (got ${c})." >&2
        exit 1
    fi
done

for rep in $(seq 1 "${REPETITIONS}"); do
    for c_count in ${CLIENTS_COUNT}; do
        for combo in ${COMBOS}; do
            model="${combo%%=*}"
            datasets_csv="${combo#*=}"
            IFS=',' read -ra datasets <<< "${datasets_csv}"
            for dataset_raw in "${datasets[@]}"; do
                dataset="${dataset_raw//_/ }"
                for algo in ${ALGORITHMS}; do
                    printf '%s|%s|%s|%s|%s\n' \
                        "${dataset}" "${algo}" "${model}" "${c_count}" "${rep}" \
                        >> "${JOB_FILE}"
                done
            done
        done
    done
done

total=$(wc -l < "${JOB_FILE}" | tr -d ' ')
echo "Generated ${total} job(s) in ${JOB_FILE}"

echo ""
echo "Breakdown by (model, dataset):"
awk -F'|' '{ printf "  %-16s %-30s %s reps × %s algos × %s client-counts\n", $3, $1, "'"${REPETITIONS}"'", "'"$(echo ${ALGORITHMS} | wc -w | tr -d ' ')"'", "'"$(echo ${CLIENTS_COUNT} | wc -w | tr -d ' ')"'" }' \
    "${JOB_FILE}" | sort -u
