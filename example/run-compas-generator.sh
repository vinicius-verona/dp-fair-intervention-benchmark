#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_number>"
    echo "Where script_number is 1-5"
    exit 1
fi

SCRIPT_NUM=$1
PYTHON="$VIRTUAL_ENV/bin/python3"
mkdir -p log
mkdir -p log/DataGeneration

case $SCRIPT_NUM in
    1)
        echo "Running Script 1 (batches 1-4)..."
        $PYTHON DPF_DataGeneration-Compas.py -s 5       > log/DataGeneration/Compas-batch-1.out 2> log/DataGeneration/Compas-batch-1.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 602627  > log/DataGeneration/Compas-batch-2.out 2> log/DataGeneration/Compas-batch-2.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 767707  > log/DataGeneration/Compas-batch-3.out 2> log/DataGeneration/Compas-batch-3.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 133843  > log/DataGeneration/Compas-batch-4.out 2> log/DataGeneration/Compas-batch-4.err &
        ;;
    2)
        echo "Running Script 2 (batches 5-8)..."
        $PYTHON DPF_DataGeneration-Compas.py -s 42     > log/DataGeneration/Compas-batch-5.out 2> log/DataGeneration/Compas-batch-5.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 153073 > log/DataGeneration/Compas-batch-6.out 2> log/DataGeneration/Compas-batch-6.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 113647 > log/DataGeneration/Compas-batch-7.out 2> log/DataGeneration/Compas-batch-7.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 6977   > log/DataGeneration/Compas-batch-8.out 2> log/DataGeneration/Compas-batch-8.err &
        ;;
    3)
        echo "Running Script 3 (batches 9-12)..."
        $PYTHON DPF_DataGeneration-Compas.py -s 253    > log/DataGeneration/Compas-batch-9.out  2> log/DataGeneration/Compas-batch-9.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 53453  > log/DataGeneration/Compas-batch-10.out 2> log/DataGeneration/Compas-batch-10.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 796969 > log/DataGeneration/Compas-batch-11.out 2> log/DataGeneration/Compas-batch-11.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 460403 > log/DataGeneration/Compas-batch-12.out 2> log/DataGeneration/Compas-batch-12.err &
        ;;
    4)
        echo "Running Script 4 (batches 13-16)..."
        $PYTHON DPF_DataGeneration-Compas.py -s 4112   > log/DataGeneration/Compas-batch-13.out 2> log/DataGeneration/Compas-batch-13.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 178753 > log/DataGeneration/Compas-batch-14.out 2> log/DataGeneration/Compas-batch-14.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 553067 > log/DataGeneration/Compas-batch-15.out 2> log/DataGeneration/Compas-batch-15.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 126613 > log/DataGeneration/Compas-batch-16.out 2> log/DataGeneration/Compas-batch-16.err &
        ;;
    5)
        echo "Running Script 5 (batches 17-20)..."
        $PYTHON DPF_DataGeneration-Compas.py -s 32645  > log/DataGeneration/Compas-batch-17.out 2> log/DataGeneration/Compas-batch-17.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 243421 > log/DataGeneration/Compas-batch-18.out 2> log/DataGeneration/Compas-batch-18.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 96797  > log/DataGeneration/Compas-batch-19.out 2> log/DataGeneration/Compas-batch-19.err &
        $PYTHON DPF_DataGeneration-Compas.py -s 583879 > log/DataGeneration/Compas-batch-20.out 2> log/DataGeneration/Compas-batch-20.err &
        ;;
    *)
        echo "Error: Invalid script number. Please use 1-5."
        exit 1
        ;;
esac

wait