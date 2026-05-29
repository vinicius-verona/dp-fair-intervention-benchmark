#!/bin/bash
# run_benchmark.sh

VENV_PATH="$HOME/public/dp-fair-intervention-benchmark/venv"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_number>"
    echo "Where script_number is 1-5"
    exit 1
fi

SCRIPT_NUM=$1
PYTHON="$VENV_PATH/bin/python3"
# mkdir -p log

case $SCRIPT_NUM in
    1)
        echo "Running Script 1 (batches 1-4)..."
        $PYTHON DPF_Benchmark-ACSIncome.py 5       > log/ACSIncome-batch-1.out 2> log/ACSIncome-batch-1.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 602627  > log/ACSIncome-batch-2.out 2> log/ACSIncome-batch-2.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 767707  > log/ACSIncome-batch-3.out 2> log/ACSIncome-batch-3.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 133843  > log/ACSIncome-batch-4.out 2> log/ACSIncome-batch-4.err &
        ;;
    2)
        echo "Running Script 2 (batches 5-8)..."
        $PYTHON DPF_Benchmark-ACSIncome.py 42     > log/ACSIncome-batch-5.out 2> log/ACSIncome-batch-5.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 153073 > log/ACSIncome-batch-6.out 2> log/ACSIncome-batch-6.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 113647 > log/ACSIncome-batch-7.out 2> log/ACSIncome-batch-7.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 6977   > log/ACSIncome-batch-8.out 2> log/ACSIncome-batch-8.err &
        ;;
    3)
        echo "Running Script 3 (batches 9-12)..."
        $PYTHON DPF_Benchmark-ACSIncome.py 253    > log/ACSIncome-batch-9.out  2> log/ACSIncome-batch-9.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 53453  > log/ACSIncome-batch-10.out 2> log/ACSIncome-batch-10.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 796969 > log/ACSIncome-batch-11.out 2> log/ACSIncome-batch-11.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 460403 > log/ACSIncome-batch-12.out 2> log/ACSIncome-batch-12.err &
        ;;
    4)
        echo "Running Script 4 (batches 13-16)..."
        $PYTHON DPF_Benchmark-ACSIncome.py 4112   > log/ACSIncome-batch-13.out 2> log/ACSIncome-batch-13.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 178753 > log/ACSIncome-batch-14.out 2> log/ACSIncome-batch-14.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 553067 > log/ACSIncome-batch-15.out 2> log/ACSIncome-batch-15.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 126613 > log/ACSIncome-batch-16.out 2> log/ACSIncome-batch-16.err &
        ;;
    5)
        echo "Running Script 5 (batches 17-20)..."
        $PYTHON DPF_Benchmark-ACSIncome.py 32645  > log/ACSIncome-batch-17.out 2> log/ACSIncome-batch-17.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 243421 > log/ACSIncome-batch-18.out 2> log/ACSIncome-batch-18.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 96797  > log/ACSIncome-batch-19.out 2> log/ACSIncome-batch-19.err &
        $PYTHON DPF_Benchmark-ACSIncome.py 583879 > log/ACSIncome-batch-20.out 2> log/ACSIncome-batch-20.err &
        ;;
    *)
        echo "Error: Invalid script number. Please use 1-5."
        exit 1
        ;;
esac




# python3 DPF_Benchmark-ACSIncome.py 5       > log/ACSIncome-batch-1.out 2> log/ACSIncome-batch-1.err &
# python3 DPF_Benchmark-ACSIncome.py 602627  > log/ACSIncome-batch-2.out 2> log/ACSIncome-batch-2.err &
# python3 DPF_Benchmark-ACSIncome.py 767707  > log/ACSIncome-batch-3.out 2> log/ACSIncome-batch-3.err &
# python3 DPF_Benchmark-ACSIncome.py 133843  > log/ACSIncome-batch-4.out 2> log/ACSIncome-batch-4.err &

# python3 DPF_Benchmark-ACSIncome.py 42     > log/ACSIncome-batch-5.out 2> log/ACSIncome-batch-5.err &
# python3 DPF_Benchmark-ACSIncome.py 153073 > log/ACSIncome-batch-6.out 2> log/ACSIncome-batch-6.err &
# python3 DPF_Benchmark-ACSIncome.py 113647 > log/ACSIncome-batch-7.out 2> log/ACSIncome-batch-7.err &
# python3 DPF_Benchmark-ACSIncome.py 6977   > log/ACSIncome-batch-8.out 2> log/ACSIncome-batch-8.err &

# wait

# python3 DPF_Benchmark-ACSIncome.py 253    > log/ACSIncome-batch-9.out  2> log/ACSIncome-batch-9.err &
# python3 DPF_Benchmark-ACSIncome.py 53453  > log/ACSIncome-batch-10.out 2> log/ACSIncome-batch-10.err &
# python3 DPF_Benchmark-ACSIncome.py 796969 > log/ACSIncome-batch-11.out 2> log/ACSIncome-batch-11.err &
# python3 DPF_Benchmark-ACSIncome.py 460403 > log/ACSIncome-batch-12.out 2> log/ACSIncome-batch-12.err &

# python3 DPF_Benchmark-ACSIncome.py 4112   > log/ACSIncome-batch-13.out 2> log/ACSIncome-batch-13.err &
# python3 DPF_Benchmark-ACSIncome.py 178753 > log/ACSIncome-batch-14.out 2> log/ACSIncome-batch-14.err &
# python3 DPF_Benchmark-ACSIncome.py 553067 > log/ACSIncome-batch-15.out 2> log/ACSIncome-batch-15.err &
# python3 DPF_Benchmark-ACSIncome.py 126613 > log/ACSIncome-batch-16.out 2> log/ACSIncome-batch-16.err &

# wait

# python3 DPF_Benchmark-ACSIncome.py 32645  > log/ACSIncome-batch-17.out 2> log/ACSIncome-batch-17.err &
# python3 DPF_Benchmark-ACSIncome.py 243421 > log/ACSIncome-batch-18.out 2> log/ACSIncome-batch-18.err &
# python3 DPF_Benchmark-ACSIncome.py 96797  > log/ACSIncome-batch-19.out 2> log/ACSIncome-batch-19.err &
# python3 DPF_Benchmark-ACSIncome.py 583879 > log/ACSIncome-batch-20.out 2> log/ACSIncome-batch-20.err &

# # python3 DPF_Benchmark-ACSIncome.py 5 42 253 4112 32645 > log/ACSIncome-batch-1.out 2> log/ACSIncome-batch-1.err &
# # python3 DPF_Benchmark-ACSIncome.py 602627 153073 53453 178753 243421 > log/ACSIncome-batch-2.out 2> log/ACSIncome-batch-2.err &
# # python3 DPF_Benchmark-ACSIncome.py 767707 113647 796969 553067 96797 > log/ACSIncome-batch-3.out 2> log/ACSIncome-batch-3.err &
# # python3 DPF_Benchmark-ACSIncome.py 133843 6977 460403 126613 583879 > log/ACSIncome-batch-4.out 2> log/ACSIncome-batch-4.err &

# wait

# for MODEL in LR RF XGB; do
#     FPATH=./data/ACSIncome/output/$MODEL

#     for SYNTH in aim mst; do
#         # FILE1=$FPATH/ACSIncome/$SYNTH/results/benchmark_results_seeds_5_42_253_4112_32645_eps_0.05_0.1_0.25_0.5_0.75_1_2_3_5_10_15_20_synth_$SYNTH.csv
#         FILE1=$FPATH/ACSIncome/$SYNTH/results/benchmark_results_seeds_5_eps_0.05_0.1_0.25_0.5_0.75_1_2_3_5_10_15_20_synth_$SYNTH.csv
#         # OUTPUT=$FPATH/benchmark_ACSIncome_$SYNTH.csv
#         OUTPUT=$FPATH/ACSIncome/$SYNTH/results/${MODEL}_results_${SYNTH}_ACSIncome.csv

#         cp $FILE1 $OUTPUT

#         for file in $FPATH/ACSIncome/$SYNTH/results/*.csv; do
#             echo Checking file $file
#             if [[ "$file" != "$FILE1" && "$file" != "$OUTPUT" ]]; then
#                 tail -n +2 "$file" >> $OUTPUT
#             fi
#         done
#     done
# done