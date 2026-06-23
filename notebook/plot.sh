#!/bin/bash

type=$1
if [ -z "$type" ]; then
    echo "Usage: $0 <type>"
    echo "<type> is either ablation, output, or output-example. It specifies which type of plots to generate."
    echo "For ablation, you also need to specify which ablation results to use (in-processing or dp-split)."
    exit 1
fi

if [ "$type" != "ablation" ] && [ "$type" != "output" ] && [ "$type" != "output-example" ]; then
    echo "Invalid type: $type"
    echo "Usage: $0 <type>"
    echo "<type> is either ablation, output, or output-example. It specifies which type of plots to generate."
    exit 1
fi

which_ablation=$2
if [ "$type" == "ablation" ] && [ -z "$which_ablation" ]; then
    echo "Usage: $0 ablation <which_ablation>"
    echo "<which_ablation> is either in-processing or dp-split. It specifies which ablation results to use for plotting."
    exit 1
fi

if [ "$type" == "ablation" ] && [ "$which_ablation" != "in-processing" ] && [ "$which_ablation" != "dp-split" ]; then
    echo "Invalid which_ablation: $which_ablation"
    echo "Usage: $0 ablation <which_ablation>"
    echo "<which_ablation> is either in-processing or dp-split. It specifies which ablation results to use for plotting."
    exit 1
fi

echo "Copying results to the current directory..."
path=$(pwd)

if [ "$type" == "ablation" ]; then
    ./get-results.sh "$path/../example/ablation/$which_ablation/"
else
    ./get-results.sh "$path/../example/$type/"
fi
result=$?

if [ $result -ne 0 ]; then
    echo "Failed to copy results. Exiting."
    exit 1
fi

echo "Generating plots..."
python3 main_claim_1.py && \
python3 main_claim_2.py && \
python3 plot_main.py && \
python3 plot_main_agg_ml_models.py

if [ $? -ne 0 ]; then
    echo "Plot generation failed."
    exit 1
fi

echo "Done."