#!/bin/bash
# run_benchmark.sh

VENV_PATH="$HOME/public/dp-fair-intervention-benchmark/venv"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_number>"
    echo "Where script_number is 1-5"
    exit 1
fi

SCRIPT_NUM=$1
COMBO=$2
PYTHON="$VENV_PATH/bin/python3"

case $SCRIPT_NUM in
    1)
        echo "Running Script 1 (batches 1-4)..."
        $PYTHON DPF_Benchmark-BoD.py 5 --combo $COMBO       > log/BoD-$COMBO-batch-1.out 2> log/BoD-$COMBO-batch-1.err &
        $PYTHON DPF_Benchmark-BoD.py 602627 --combo $COMBO  > log/BoD-$COMBO-batch-2.out 2> log/BoD-$COMBO-batch-2.err &
        $PYTHON DPF_Benchmark-BoD.py 767707 --combo $COMBO  > log/BoD-$COMBO-batch-3.out 2> log/BoD-$COMBO-batch-3.err &
        $PYTHON DPF_Benchmark-BoD.py 133843 --combo $COMBO  > log/BoD-$COMBO-batch-4.out 2> log/BoD-$COMBO-batch-4.err &
        ;;
    2)
        echo "Running Script 2 (batches 5-8)..."
        $PYTHON DPF_Benchmark-BoD.py 42 --combo $COMBO     > log/BoD-$COMBO-batch-5.out 2> log/BoD-$COMBO-batch-5.err &
        $PYTHON DPF_Benchmark-BoD.py 153073 --combo $COMBO > log/BoD-$COMBO-batch-6.out 2> log/BoD-$COMBO-batch-6.err &
        $PYTHON DPF_Benchmark-BoD.py 113647 --combo $COMBO > log/BoD-$COMBO-batch-7.out 2> log/BoD-$COMBO-batch-7.err &
        $PYTHON DPF_Benchmark-BoD.py 6977 --combo $COMBO   > log/BoD-$COMBO-batch-8.out 2> log/BoD-$COMBO-batch-8.err &
        ;;
    3)
        echo "Running Script 3 (batches 9-12)..."
        $PYTHON DPF_Benchmark-BoD.py 253 --combo $COMBO    > log/BoD-$COMBO-batch-9.out  2> log/BoD-$COMBO-batch-9.err &
        $PYTHON DPF_Benchmark-BoD.py 53453 --combo $COMBO  > log/BoD-$COMBO-batch-10.out 2> log/BoD-$COMBO-batch-10.err &
        $PYTHON DPF_Benchmark-BoD.py 796969 --combo $COMBO > log/BoD-$COMBO-batch-11.out 2> log/BoD-$COMBO-batch-11.err &
        $PYTHON DPF_Benchmark-BoD.py 460403 --combo $COMBO > log/BoD-$COMBO-batch-12.out 2> log/BoD-$COMBO-batch-12.err &
        ;;
    4)
        echo "Running Script 4 (batches 13-16)..."
        $PYTHON DPF_Benchmark-BoD.py 4112 --combo $COMBO   > log/BoD-$COMBO-batch-13.out 2> log/BoD-$COMBO-batch-13.err &
        $PYTHON DPF_Benchmark-BoD.py 178753 --combo $COMBO > log/BoD-$COMBO-batch-14.out 2> log/BoD-$COMBO-batch-14.err &
        $PYTHON DPF_Benchmark-BoD.py 553067 --combo $COMBO > log/BoD-$COMBO-batch-15.out 2> log/BoD-$COMBO-batch-15.err &
        $PYTHON DPF_Benchmark-BoD.py 126613 --combo $COMBO > log/BoD-$COMBO-batch-16.out 2> log/BoD-$COMBO-batch-16.err &
        ;;
    5)
        echo "Running Script 5 (batches 17-20)..."
        $PYTHON DPF_Benchmark-BoD.py 32645 --combo $COMBO  > log/BoD-$COMBO-batch-17.out 2> log/BoD-$COMBO-batch-17.err &
        $PYTHON DPF_Benchmark-BoD.py 243421 --combo $COMBO > log/BoD-$COMBO-batch-18.out 2> log/BoD-$COMBO-batch-18.err &
        $PYTHON DPF_Benchmark-BoD.py 96797 --combo $COMBO  > log/BoD-$COMBO-batch-19.out 2> log/BoD-$COMBO-batch-19.err &
        $PYTHON DPF_Benchmark-BoD.py 583879 --combo $COMBO > log/BoD-$COMBO-batch-20.out 2> log/BoD-$COMBO-batch-20.err &
        ;;
    *)
        echo "Error: Invalid script number. Please use 1-5."
        exit 1
        ;;
esac


# COMBO=$1

# echo Executing BoD-${COMBO}

# python3 DPF_Benchmark-BoD.py 5 42 253 4112 32645 --combo $COMBO  > log/BoD-$COMBO-batch-1.out 2> log/BoD-$COMBO-batch-1.err &
# python3 DPF_Benchmark-BoD.py 602627 153073 53453 178753 243421 --combo $COMBO  > log/BoD-$COMBO-batch-2.out 2> log/BoD-$COMBO-batch-2.err &
# python3 DPF_Benchmark-BoD.py 767707 113647 796969 553067 96797 --combo $COMBO  > log/BoD-$COMBO-batch-3.out 2> log/BoD-$COMBO-batch-3.err &
# python3 DPF_Benchmark-BoD.py 133843 6977 460403 126613 583879 --combo $COMBO  > log/BoD-$COMBO-batch-4.out 2> log/BoD-$COMBO-batch-4.err &

# wait

# for MODEL in LR RF XGB; do
#     FPATH=./data/BoD/BoD-$COMBO/output/$MODEL

#     for SYNTH in aim mst; do
#         FILE1=$FPATH/BoD-$COMBO/$SYNTH/results/benchmark_results_seeds_5_42_253_4112_32645_eps_0.05_0.1_0.25_0.5_0.75_1_2_3_5_10_15_20_synth_$SYNTH.csv
#         OUTPUT=$FPATH/BoD-$COMBO/$SYNTH/results/benchmark_BoD-${COMBO}_$SYNTH.csv

#         cp $FILE1 $OUTPUT

#         for file in $FPATH/BoD-$COMBO/$SYNTH/results/*.csv; do
#             echo Checking file $file
#             if [[ "$file" != "$FILE1" && "$file" != "$OUTPUT" ]]; then
#                 tail -n +2 "$file" >> $OUTPUT
#             fi
#         done
#     done
# done