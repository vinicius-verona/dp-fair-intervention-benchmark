#!/bin/bash

VENV_PATH="$(dirname $(dirname $(which python3)))"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_number>"
    echo "Where script_number is 1-5"
    exit 1
fi

SCRIPT_NUM=$1
COMBO=$2
PYTHON="$VENV_PATH/bin/python3"
mkdir -p log
mkdir -p log/Benchmark
case $SCRIPT_NUM in
    1)
        echo "Running Script 1 (batches 1-4)..."
        $PYTHON DPF_Benchmark-BoD.py -s 5 --combo $COMBO       > log/Benchmark/BoD-$COMBO-batch-1.out 2> log/Benchmark/BoD-$COMBO-batch-1.err &
        $PYTHON DPF_Benchmark-BoD.py -s 602627 --combo $COMBO  > log/Benchmark/BoD-$COMBO-batch-2.out 2> log/Benchmark/BoD-$COMBO-batch-2.err &
        $PYTHON DPF_Benchmark-BoD.py -s 767707 --combo $COMBO  > log/Benchmark/BoD-$COMBO-batch-3.out 2> log/Benchmark/BoD-$COMBO-batch-3.err &
        $PYTHON DPF_Benchmark-BoD.py -s 133843 --combo $COMBO  > log/Benchmark/BoD-$COMBO-batch-4.out 2> log/Benchmark/BoD-$COMBO-batch-4.err &
        ;;
    2)
        echo "Running Script 2 (batches 5-8)..."
        $PYTHON DPF_Benchmark-BoD.py -s 42 --combo $COMBO     > log/Benchmark/BoD-$COMBO-batch-5.out 2> log/Benchmark/BoD-$COMBO-batch-5.err &
        $PYTHON DPF_Benchmark-BoD.py -s 153073 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-6.out 2> log/Benchmark/BoD-$COMBO-batch-6.err &
        $PYTHON DPF_Benchmark-BoD.py -s 113647 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-7.out 2> log/Benchmark/BoD-$COMBO-batch-7.err &
        $PYTHON DPF_Benchmark-BoD.py -s 6977 --combo $COMBO   > log/Benchmark/BoD-$COMBO-batch-8.out 2> log/Benchmark/BoD-$COMBO-batch-8.err &
        ;;
    3)
        echo "Running Script 3 (batches 9-12)..."
        $PYTHON DPF_Benchmark-BoD.py -s 253 --combo $COMBO    > log/Benchmark/BoD-$COMBO-batch-9.out  2> log/Benchmark/BoD-$COMBO-batch-9.err &
        $PYTHON DPF_Benchmark-BoD.py -s 53453 --combo $COMBO  > log/Benchmark/BoD-$COMBO-batch-10.out 2> log/Benchmark/BoD-$COMBO-batch-10.err &
        $PYTHON DPF_Benchmark-BoD.py -s 796969 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-11.out 2> log/Benchmark/BoD-$COMBO-batch-11.err &
        $PYTHON DPF_Benchmark-BoD.py -s 460403 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-12.out 2> log/Benchmark/BoD-$COMBO-batch-12.err &
        ;;
    4)
        echo "Running Script 4 (batches 13-16)..."
        $PYTHON DPF_Benchmark-BoD.py -s 4112 --combo $COMBO   > log/Benchmark/BoD-$COMBO-batch-13.out 2> log/Benchmark/BoD-$COMBO-batch-13.err &
        $PYTHON DPF_Benchmark-BoD.py -s 178753 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-14.out 2> log/Benchmark/BoD-$COMBO-batch-14.err &
        $PYTHON DPF_Benchmark-BoD.py -s 553067 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-15.out 2> log/Benchmark/BoD-$COMBO-batch-15.err &
        $PYTHON DPF_Benchmark-BoD.py -s 126613 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-16.out 2> log/Benchmark/BoD-$COMBO-batch-16.err &
        ;;
    5)
        echo "Running Script 5 (batches 17-20)..."
        $PYTHON DPF_Benchmark-BoD.py -s 32645 --combo $COMBO  > log/Benchmark/BoD-$COMBO-batch-17.out 2> log/Benchmark/BoD-$COMBO-batch-17.err &
        $PYTHON DPF_Benchmark-BoD.py -s 243421 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-18.out 2> log/Benchmark/BoD-$COMBO-batch-18.err &
        $PYTHON DPF_Benchmark-BoD.py -s 96797 --combo $COMBO  > log/Benchmark/BoD-$COMBO-batch-19.out 2> log/Benchmark/BoD-$COMBO-batch-19.err &
        $PYTHON DPF_Benchmark-BoD.py -s 583879 --combo $COMBO > log/Benchmark/BoD-$COMBO-batch-20.out 2> log/Benchmark/BoD-$COMBO-batch-20.err &
        ;;
    *)
        echo "Error: Invalid script number. Please use 1-5."
        exit 1
        ;;
esac

