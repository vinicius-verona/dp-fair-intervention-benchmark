#!/bin/bash
# . ~/public/Experiments/exp_venv/bin/activate
# python3 dp_compas_dataset_generator.py 5 ~/public/Experiments/data/Compas &
# python3 dp_compas_dataset_generator.py 42 ~/public/Experiments/data/Compas &
# python3 dp_compas_dataset_generator.py 253 ~/public/Experiments/data/Compas &
# python3 dp_compas_dataset_generator.py 4112 ~/public/Experiments/data/Compas &
# python3 dp_compas_dataset_generator.py 32645 ~/public/Experiments/data/Compas

# SAVE_PATH=~/public/Experiments/data/Compas
SEEDS=(5 42 253 4112 32645 602627 153073 53453 178753 243421 767707 113647 796969 553067 96797 133843 6977 460403 126613 583879)
# SEEDS=(42 4112 32645 96797 126613 243421 583879 553067)
SAVE_PATH=./

echo "Generating COMPAS [DP] datasets"
for SEED in "${SEEDS[@]}"
do
    python3 dp_compas_dataset_generator.py $SEED $SAVE_PATH > compas_gen_seed_$SEED.out 2> compas_gen_seed_$SEED.err &
done