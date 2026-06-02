#!/bin/bash
# run_benchmark.sh

VENV_PATH="$HOME/dp-fair-intervention-benchmark/venv"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_number>"
    echo "Where script_number is 1-5"
    exit 1
fi

SCRIPT_NUM=$1
PYTHON="$VENV_PATH/bin/python3"
mkdir -p log

case $SCRIPT_NUM in
    1)
        echo "Running Script 1 (batches 1-4)..."
        $PYTHON DPF_Benchmark-ACSIncome.py 5       > log/Benchmark/ACSIncome-batch-1.out 2> log/Benchmark/ACSIncome-batch-1.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 602627  > log/Benchmark/ACSIncome-batch-2.out 2> log/Benchmark/ACSIncome-batch-2.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 767707  > log/Benchmark/ACSIncome-batch-3.out 2> log/Benchmark/ACSIncome-batch-3.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 133843  > log/Benchmark/ACSIncome-batch-4.out 2> log/Benchmark/ACSIncome-batch-4.err &
        ;;
    2)
        echo "Running Script 2 (batches 5-8)..."
        $PYTHON DPF_Benchmark-ACSIncome.py 42     > log/Benchmark/ACSIncome-batch-5.out 2> log/Benchmark/ACSIncome-batch-5.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 153073 > log/Benchmark/ACSIncome-batch-6.out 2> log/Benchmark/ACSIncome-batch-6.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 113647 > log/Benchmark/ACSIncome-batch-7.out 2> log/Benchmark/ACSIncome-batch-7.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 6977   > log/Benchmark/ACSIncome-batch-8.out 2> log/Benchmark/ACSIncome-batch-8.err &
        ;;
    3)
        echo "Running Script 3 (batches 9-12)..."
        $PYTHON DPF_Benchmark-ACSIncome.py 253    > log/Benchmark/ACSIncome-batch-9.out  2> log/Benchmark/ACSIncome-batch-9.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 53453  > log/Benchmark/ACSIncome-batch-10.out 2> log/Benchmark/ACSIncome-batch-10.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 796969 > log/Benchmark/ACSIncome-batch-11.out 2> log/Benchmark/ACSIncome-batch-11.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 460403 > log/Benchmark/ACSIncome-batch-12.out 2> log/Benchmark/ACSIncome-batch-12.err &
        ;;
    4)
        echo "Running Script 4 (batches 13-16)..."
        $PYTHON DPF_Benchmark-ACSIncome.py 4112   > log/Benchmark/ACSIncome-batch-13.out 2> log/Benchmark/ACSIncome-batch-13.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 178753 > log/Benchmark/ACSIncome-batch-14.out 2> log/Benchmark/ACSIncome-batch-14.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 553067 > log/Benchmark/ACSIncome-batch-15.out 2> log/Benchmark/ACSIncome-batch-15.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 126613 > log/Benchmark/ACSIncome-batch-16.out 2> log/Benchmark/ACSIncome-batch-16.err &
        ;;
    5)
        echo "Running Script 5 (batches 17-20)..."
        $PYTHON DPF_Benchmark-ACSIncome.py 32645  > log/Benchmark/ACSIncome-batch-17.out 2> log/Benchmark/ACSIncome-batch-17.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 243421 > log/Benchmark/ACSIncome-batch-18.out 2> log/Benchmark/ACSIncome-batch-18.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 96797  > log/Benchmark/ACSIncome-batch-19.out 2> log/Benchmark/ACSIncome-batch-19.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 583879 > log/Benchmark/ACSIncome-batch-20.out 2> log/Benchmark/ACSIncome-batch-20.err &
        ;;
    *)
        echo "Error: Invalid script number. Please use 1-5."
        exit 1
        ;;
esac


