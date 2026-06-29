#!/bin/bash
# exec-parallel.sh
# Usage: ./exec-parallel.sh [seed1 seed2 ... seedX]
# RAM limit can be overridden via RAM_LIMIT_GB env var:
#   RAM_LIMIT_GB=4 ./exec-parallel.sh 1 2 3
# COMBO can be overridden via COMBO env var:
#   COMBO=3 ./exec-parallel.sh 1 2 3
# Seeds default to 20 different ones used in the paper, if none are passed:
#   ./exec-parallel.sh

set -euo pipefail
mkdir -p log/Parallel

PYTHON_SCRIPT="./DPF_Benchmark-BoD.py"
LOG_FILE="log/Parallel/killed_scripts-bod-benchmark.log"

# --- Defaults, overridable by the user ---
# Default 10GB per seed, override with env var
RAM_LIMIT_GB="${RAM_LIMIT_GB:-10}" 

# Default combo 5 if none given by the user, override with env var
COMBO="${COMBO:-5}"

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
    local unit_name="BoD-Benchmark-seed-${seed}-$$"
    local PYTHON="$VIRTUAL_ENV/bin/python3"

    systemd-run --user --scope --unit="$unit_name" -p MemoryMax="${RAM_LIMIT_GB}G" \
        "$PYTHON" "$PYTHON_SCRIPT" "-s $seed" --combo "$COMBO" > "log/Benchmark/BoD-seed-${seed}.out" 2> "log/Benchmark/BoD-seed-${seed}.err"
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
export PYTHON_SCRIPT RAM_LIMIT_GB LOG_FILE COMBO

parallel -j "$JOBS" run_with_ram_limit ::: "${SEEDS[@]}"

echo "Done. Check $LOG_FILE for any killed/failed scripts."








# #!/bin/bash

# if [ $# -eq 0 ]; then
#     echo "Usage: $0 <script_number>"
#     echo "Where script_number is 1-5"
#     exit 1
# fi

# SCRIPT_NUM=$1
# COMBO=$2
# PYTHON="$VIRTUAL_ENV/bin/python3"
# mkdir -p log
# mkdir -p log/Benchmark
# case $SCRIPT_NUM in
#     1)
#         echo "Running Script 1 (batches 1-4)..."
#         $PYTHON DPF_Benchmark-BoD.py -s 5 --combo $COMBO       > log/Benchmark/BoD-$COMBO-batch-1.out 2> log/Benchmark/BoD-$COMBO-batch-1.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 602627 --combo $COMBO  > log/Benchmark/BoD-$COMBO-batch-2.out 2> log/Benchmark/BoD-$COMBO-batch-2.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 767707 --combo $COMBO  > log/Benchmark/BoD-$COMBO-batch-3.out 2> log/Benchmark/BoD-$COMBO-batch-3.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 133843 --combo $COMBO  > log/Benchmark/BoD-$COMBO-batch-4.out 2> log/Benchmark/BoD-$COMBO-batch-4.err &
#         ;;
#     2)
#         echo "Running Script 2 (batches 5-8)..."
#         $PYTHON DPF_Benchmark-BoD.py -s 42 --combo $COMBO     > log/Benchmark/BoD-$COMBO-batch-5.out 2> log/Benchmark/BoD-$COMBO-batch-5.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 153073 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-6.out 2> log/Benchmark/BoD-$COMBO-batch-6.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 113647 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-7.out 2> log/Benchmark/BoD-$COMBO-batch-7.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 6977 --combo $COMBO   > log/Benchmark/BoD-$COMBO-batch-8.out 2> log/Benchmark/BoD-$COMBO-batch-8.err &
#         ;;
#     3)
#         echo "Running Script 3 (batches 9-12)..."
#         $PYTHON DPF_Benchmark-BoD.py -s 253 --combo $COMBO    > log/Benchmark/BoD-$COMBO-batch-9.out  2> log/Benchmark/BoD-$COMBO-batch-9.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 53453 --combo $COMBO  > log/Benchmark/BoD-$COMBO-batch-10.out 2> log/Benchmark/BoD-$COMBO-batch-10.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 796969 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-11.out 2> log/Benchmark/BoD-$COMBO-batch-11.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 460403 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-12.out 2> log/Benchmark/BoD-$COMBO-batch-12.err &
#         ;;
#     4)
#         echo "Running Script 4 (batches 13-16)..."
#         $PYTHON DPF_Benchmark-BoD.py -s 4112 --combo $COMBO   > log/Benchmark/BoD-$COMBO-batch-13.out 2> log/Benchmark/BoD-$COMBO-batch-13.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 178753 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-14.out 2> log/Benchmark/BoD-$COMBO-batch-14.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 553067 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-15.out 2> log/Benchmark/BoD-$COMBO-batch-15.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 126613 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-16.out 2> log/Benchmark/BoD-$COMBO-batch-16.err &
#         ;;
#     5)
#         echo "Running Script 5 (batches 17-20)..."
#         $PYTHON DPF_Benchmark-BoD.py -s 32645 --combo $COMBO  > log/Benchmark/BoD-$COMBO-batch-17.out 2> log/Benchmark/BoD-$COMBO-batch-17.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 243421 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-18.out 2> log/Benchmark/BoD-$COMBO-batch-18.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 96797 --combo $COMBO  > log/Benchmark/BoD-$COMBO-batch-19.out 2> log/Benchmark/BoD-$COMBO-batch-19.err &
#         $PYTHON DPF_Benchmark-BoD.py -s 583879 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-20.out 2> log/Benchmark/BoD-$COMBO-batch-20.err &
#         ;;
#     *)
#         echo "Error: Invalid script number. Please use 1-5."
#         exit 1
#         ;;
# esac

# wait