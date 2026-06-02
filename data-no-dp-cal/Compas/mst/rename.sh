#!/usr/bin/env bash
 
# Renames CSV files matching:
#   aim_synthetic_train_dataset_seed_[INT]_epsilon_[INT_OR_FLOAT].csv
#   mst_synthetic_train_dataset_seed_[INT]_epsilon_[INT_OR_FLOAT].csv
# To:
#   Compas_split_dataset_seed_[INT]_epsilon-[INT_OR_FLOAT].csv
#
# Searches in the current directory and 1 level of subdirectories.
 
COUNT=0
 
while IFS= read -r filepath; do
    filename=$(basename "$filepath")
    dir=$(dirname "$filepath")
 
    if [[ "$filename" =~ ^(aim|mst)_synthetic_train_dataset_seed_([0-9]+)_epsilon_([0-9]+(\.[0-9]+)?)\.csv$ ]]; then
        seed="${BASH_REMATCH[2]}"
        epsilon="${BASH_REMATCH[3]}"
        newname="Compas_split_dataset_seed_${seed}_epsilon-${epsilon}.csv"
        newpath="${dir}/${newname}"
 
        echo "Renaming: $filepath"
        echo "      To: $newpath"
        mv "$filepath" "$newpath"
        ((COUNT++))
    fi
done < <(find . -maxdepth 2 -name "*.csv" -type f | sort)
 
echo ""
echo "Done. $COUNT file(s) renamed."
