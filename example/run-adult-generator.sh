#!/bin/bash
# run-adult-generator.sh
# Usage: ./run-adult-generator.sh [seed1 seed2 ... seedX]
# RAM limit can be overridden via RAM_LIMIT_GB env var:
#   RAM_LIMIT_GB=4 ./run-adult-generator.sh 1 2 3
# Seeds default to 20 different ones used in the paper, if none are passed:
#   ./run-adult-generator.sh

set -euo pipefail
mkdir -p log/Parallel

PYTHON_SCRIPT="./DPF_DataGeneration-Adult.py"
LOG_FILE="log/Parallel/killed_scripts-adult-gen.log"

# --- Defaults, overridable by the user ---
# Default 10GB per seed, override with env var
RAM_LIMIT_GB="${RAM_LIMIT_GB:-10}" 

# Default seeds if none given by the user
DEFAULT_SEEDS=(5 602627 767707 133843 42 153073 113647 6977 253 53453 796969 460403 4112 178753 553067 126613 32645 243421 96797 583879) 

# If no seeds were passed, fall back to defaults; otherwise use what was given
if [[ $# -eq 0 ]]; then
    SEEDS=("${DEFAULT_SEEDS[@]}")
    echo "No seeds provided — using default seeds: ${SEEDS[*]}"
else
    SEEDS=("$@")
fi

JOBS="${#SEEDS[@]}"

# Reset LOG file per run
: > "$LOG_FILE"

run_with_ram_limit() {
    local seed="$1"
    local unit_name="Adult-DataGeneration-seed-${seed}-$$"
    local PYTHON="$VIRTUAL_ENV/bin/python3"

    systemd-run --user --scope --unit="$unit_name" -p MemoryMax="${RAM_LIMIT_GB}G" \
        "$PYTHON" "$PYTHON_SCRIPT" "-s $seed" > "log/DataGeneration/Adult-seed-${seed}.out" 2> "log/DataGeneration/Adult-seed-${seed}.err"
    local exit_code=$?

    if [[ "$exit_code" -eq 137 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Seed=$seed KILLED (exceeded ${RAM_LIMIT_GB}GB RAM, exit=137)" \
            >> "$LOG_FILE"
    elif [[ "$exit_code" -ne 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Seed=$seed FAILED (exit=$exit_code, not necessarily RAM-related)" \
            >> "$LOG_FILE"
    fi

    return "$exit_code"
}

export -f run_with_ram_limit
export PYTHON_SCRIPT RAM_LIMIT_GB LOG_FILE

echo "Running $JOBS jobs in parallel with a RAM limit of ${RAM_LIMIT_GB}GB each..."

parallel -j "$JOBS" run_with_ram_limit ::: "${SEEDS[@]}"

echo "Done. Check $LOG_FILE for any killed/failed scripts."