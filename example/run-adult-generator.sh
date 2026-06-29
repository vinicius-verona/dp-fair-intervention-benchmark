#!/bin/bash
# exec-parallel.sh
# Usage: ./exec-parallel.sh [seed1 seed2 ... seedX]
# RAM limit can be overridden via RAM_LIMIT_GB env var:
#   RAM_LIMIT_GB=4 ./exec-parallel.sh 1 2 3
# Seeds default to 20 different ones used in the paper, if none are passed:
#   ./exec-parallel.sh

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

parallel -j "$JOBS" run_with_ram_limit ::: "${SEEDS[@]}"

echo "Done. Check $LOG_FILE for any killed/failed scripts."





# #!/bin/bash

# if [ $# -eq 0 ]; then
#     echo "Usage: $0 <script_number>"
#     echo "Where script_number is 1-5"
#     exit 1
# fi

# SCRIPT_NUM=$1
# PYTHON="$VIRTUAL_ENV/bin/python3"
# mkdir -p log
# mkdir -p log/DataGeneration

# case $SCRIPT_NUM in
#     1)
#         echo "Running Script 1 (batches 1-4)..."
#         $PYTHON DPF_DataGeneration-Adult.py -s 5       > log/DataGeneration/Adult-batch-1.out 2> log/DataGeneration/Adult-batch-1.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 602627  > log/DataGeneration/Adult-batch-2.out 2> log/DataGeneration/Adult-batch-2.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 767707  > log/DataGeneration/Adult-batch-3.out 2> log/DataGeneration/Adult-batch-3.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 133843  > log/DataGeneration/Adult-batch-4.out 2> log/DataGeneration/Adult-batch-4.err &
#         ;;
#     2)
#         echo "Running Script 2 (batches 5-8)..."
#         $PYTHON DPF_DataGeneration-Adult.py -s 42     > log/DataGeneration/Adult-batch-5.out 2> log/DataGeneration/Adult-batch-5.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 153073 > log/DataGeneration/Adult-batch-6.out 2> log/DataGeneration/Adult-batch-6.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 113647 > log/DataGeneration/Adult-batch-7.out 2> log/DataGeneration/Adult-batch-7.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 6977   > log/DataGeneration/Adult-batch-8.out 2> log/DataGeneration/Adult-batch-8.err &
#         ;;
#     3)
#         echo "Running Script 3 (batches 9-12)..."
#         $PYTHON DPF_DataGeneration-Adult.py -s 253    > log/DataGeneration/Adult-batch-9.out  2> log/DataGeneration/Adult-batch-9.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 53453  > log/DataGeneration/Adult-batch-10.out 2> log/DataGeneration/Adult-batch-10.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 796969 > log/DataGeneration/Adult-batch-11.out 2> log/DataGeneration/Adult-batch-11.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 460403 > log/DataGeneration/Adult-batch-12.out 2> log/DataGeneration/Adult-batch-12.err &
#         ;;
#     4)
#         echo "Running Script 4 (batches 13-16)..."
#         $PYTHON DPF_DataGeneration-Adult.py -s 4112   > log/DataGeneration/Adult-batch-13.out 2> log/DataGeneration/Adult-batch-13.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 178753 > log/DataGeneration/Adult-batch-14.out 2> log/DataGeneration/Adult-batch-14.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 553067 > log/DataGeneration/Adult-batch-15.out 2> log/DataGeneration/Adult-batch-15.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 126613 > log/DataGeneration/Adult-batch-16.out 2> log/DataGeneration/Adult-batch-16.err &
#         ;;
#     5)
#         echo "Running Script 5 (batches 17-20)..."
#         $PYTHON DPF_DataGeneration-Adult.py -s 32645  > log/DataGeneration/Adult-batch-17.out 2> log/DataGeneration/Adult-batch-17.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 243421 > log/DataGeneration/Adult-batch-18.out 2> log/DataGeneration/Adult-batch-18.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 96797  > log/DataGeneration/Adult-batch-19.out 2> log/DataGeneration/Adult-batch-19.err &
#         $PYTHON DPF_DataGeneration-Adult.py -s 583879 > log/DataGeneration/Adult-batch-20.out 2> log/DataGeneration/Adult-batch-20.err &
#         ;;
#     *)
#         echo "Error: Invalid script number. Please use 1-5."
#         exit 1
#         ;;
# esac
# wait