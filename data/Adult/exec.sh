#!/bin/bash
# . ~/public/Experiments/exp_venv/bin/activate
# python3 dp_adult_dataset_generator.py 5 ~/public/Experiments/data/Adult > adult_seed_5.out 2> adult_seed_5.err  &
# python3 dp_adult_dataset_generator.py 42 ~/public/Experiments/data/Adult > adult_seed_42.out 2> adult_seed_42.err  &
# python3 dp_adult_dataset_generator.py 253 ~/public/Experiments/data/Adult > adult_seed_253.out 2> adult_seed_253.err  &
# python3 dp_adult_dataset_generator.py 4112 ~/public/Experiments/data/Adult > adult_seed_4112.out 2> adult_seed_4112.err  &
# python3 dp_adult_dataset_generator.py 32645 ~/public/Experiments/data/Adult > adult_seed_32645.out 2> adult_seed_32645.err 


SEEDS=(5 42 253 4112 32645 602627 153073 53453 178753 243421 767707 113647 796969 553067 96797 133843 6977 460403 126613 583879)
SAVE_PATH=./

echo "Generating adult [DP] datasets"
for SEED in "${SEEDS[@]}"
do
    python3 dp_adult_dataset_generator.py $SEED $SAVE_PATH > adult_gen_seed_$SEED.out 2> adult_gen_seed_$SEED.err &

    # python3 dp_adult_dataset_generator_separate.py 10 $SEED $SAVE_PATH > adult_gen_seed_10_$SEED.out 2> adult_gen_seed_10_$SEED.err &
    # python3 dp_adult_dataset_generator_separate.py 15 $SEED $SAVE_PATH > adult_gen_seed_15_$SEED.out 2> adult_gen_seed_15_$SEED.err &
    # python3 dp_adult_dataset_generator_separate.py 20 $SEED $SAVE_PATH > adult_gen_seed_20_$SEED.out 2> adult_gen_seed_20_$SEED.err &
done